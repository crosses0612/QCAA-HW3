"""
problem4.py -- QCAA HW3 Problem 4: VQE
=======================================

(a) 四種組合的 VQE（noiseless statevector）：
        UCCSD+JW, UCCSD+parity, real-amplitude+JW, real-amplitude+parity
    全部共用同一組設定（題目要求只陳述一次）：
      * optimizer       : scipy L-BFGS-B（數值梯度）
      * tolerance       : ftol = 1e-10
      * max iterations  : 200
      * 初始化規則      : numpy.random.default_rng(13705049).normal(0, 0.1, n)
                          （每個 case 以相同 seed 重新初始化 rng）
      * seed            : 13705049（學號；VQE 初始化與 shot sampling 共用）
    回報：最佳化能量、最佳參數、電路參數個數、目標函數呼叫次數。

(b) 與同一 Hamiltonian 的精確對角化比較，回報各 case 絕對誤差。
    benchmark：UCCSD 兩種映射必須達 FCI（<1e-7 Ha）；HEA 至少達化學精度。

(c) 取最佳 2-qubit parity 電路（不重新最佳化），在固定參數下以有限
    shots（101 與 10^4）估計基態能量：
      * ideal backend : qiskit-aer AerSimulator（shot-based, noiseless）
      * noisy backend : AerSimulator.from_backend(FakeBrisbane)
        -- target QPU = ibm_brisbane（127-qubit Eagle r3），noise model 取自
           qiskit-ibm-runtime fake provider 的校準快照（gate error / readout
           error / T1 / T2；主要參數於輸出與 JSON 中列出）
      * shot 分配規則：Hamiltonian 以 qubit-wise commuting 分組
        （2 組：{I, Z0, Z1, Z0Z1} 與 {X0X1}），總 shots 平均分配到各組，
        餘數給前面的組
      * 對每組由原始測量結果計算樣本平均與標準誤（同組內共變異
        自動包含於樣本變異數；附錄 C 要求）
"""

from __future__ import annotations

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from hw3_common import (
    REFS,
    SEED,
    build_fermionic_hamiltonian,
    get_h2_integrals,
    map_jordan_wigner,
    map_parity_tapered,
    operator_matrix,
    package_versions,
    qubit_operator_to_sparse_pauli_op,
    save_results,
    spatial_to_spin_orbital,
)
from problem3 import build_hea, evolution_circuit, uccsd_generator_spos

OPTIMIZER_SETTINGS = {
    "method": "L-BFGS-B",
    "ftol": 1e-10,
    "maxiter": 200,
    "init_rule": "default_rng(13705049).normal(0, 0.1, n_params)",
    "seed": SEED,
}


# ---------------------------------------------------------------------------
# Hamiltonian 與電路
# ---------------------------------------------------------------------------

def build_hamiltonians():
    mol = get_h2_integrals()
    h_so, eri_so = spatial_to_spin_orbital(mol.h_mo, mol.eri_phys, "interleaved")
    jw_h = map_jordan_wigner(
        build_fermionic_hamiltonian(h_so, eri_so, constant=mol.e_nuc))
    h_so_b, eri_so_b = spatial_to_spin_orbital(mol.h_mo, mol.eri_phys, "blocked")
    parity_h, _ = map_parity_tapered(
        build_fermionic_hamiltonian(h_so_b, eri_so_b, constant=mol.e_nuc),
        4, n_alpha=1, n_beta=1)
    return {
        "mol": mol,
        "jw_spo": qubit_operator_to_sparse_pauli_op(jw_h, 4),
        "parity_spo": qubit_operator_to_sparse_pauli_op(parity_h, 2),
        # 精確對角化參考值（同一 Hamiltonian；JW 全空間基態即中性 sector）
        "E_exact_jw": float(np.linalg.eigvalsh(operator_matrix(jw_h, 4))[0]),
        "E_exact_parity": float(np.linalg.eigvalsh(operator_matrix(parity_h, 2))[0]),
    }


def build_cases(ham):
    h_gen_jw, h_gen_par, _, _ = uccsd_generator_spos()
    qc_ucc_jw, _ = evolution_circuit(h_gen_jw, 4, prep_x=[0, 1])
    qc_ucc_par, _ = evolution_circuit(h_gen_par, 2, prep_x=[0])
    return {
        "UCCSD+JW": (qc_ucc_jw, ham["jw_spo"], ham["E_exact_jw"]),
        "UCCSD+parity": (qc_ucc_par, ham["parity_spo"], ham["E_exact_parity"]),
        "real-amplitude+JW": (build_hea(4, 2, prep_x=[0, 1]),
                              ham["jw_spo"], ham["E_exact_jw"]),
        "real-amplitude+parity": (build_hea(2, 2, prep_x=[0]),
                                  ham["parity_spo"], ham["E_exact_parity"]),
    }


# ---------------------------------------------------------------------------
# (a)(b) VQE
# ---------------------------------------------------------------------------

def run_vqe(qc, h_spo) -> dict:
    n_params = qc.num_parameters
    rng = np.random.default_rng(SEED)            # 每個 case 相同初始化規則
    x0 = rng.normal(0.0, 0.1, n_params)
    n_evals = 0

    def objective(x: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        sv = Statevector(qc.assign_parameters(x))
        return float(np.real(sv.expectation_value(h_spo)))

    res = minimize(
        objective, x0,
        method=OPTIMIZER_SETTINGS["method"],
        options={"ftol": OPTIMIZER_SETTINGS["ftol"],
                 "maxiter": OPTIMIZER_SETTINGS["maxiter"]},
    )
    return {
        "energy": float(res.fun),
        "params": np.asarray(res.x),
        "n_params": n_params,
        "nfev": int(n_evals),
        "converged": bool(res.success),
    }


# ---------------------------------------------------------------------------
# (c) 有限 shots 估計
# ---------------------------------------------------------------------------

def allocate_shots(total: int, n_groups: int) -> list[int]:
    """總 shots 平均分配到各測量組，餘數依序給前面的組。"""
    base, rem = divmod(total, n_groups)
    return [base + (1 if i < rem else 0) for i in range(n_groups)]


def group_measurement_circuit(qc_bound, group) -> tuple:
    """附加基底旋轉與測量；回傳 (circuit, per-qubit basis)。"""
    n = qc_bound.num_qubits
    basis = ["I"] * n
    for label in group.paulis.to_labels():
        for q, ch in enumerate(reversed(label)):   # qiskit label 為 little-endian
            if ch != "I":
                assert basis[q] in ("I", ch), "group is not qubit-wise commuting"
                basis[q] = ch
    qc = qc_bound.copy()
    for q, b in enumerate(basis):
        if b == "X":
            qc.h(q)
        elif b == "Y":
            qc.sdg(q)
            qc.h(q)
    qc.measure_all()
    return qc, basis


def group_statistics(counts: dict, group) -> tuple[float, float, int]:
    """由原始計數計算該組能量貢獻的樣本平均與標準誤。

    每個 outcome 的隨機變數 Y = sum_k c_k * (-1)^(parity of bits on supp(P_k))；
    樣本變異數天然包含同組 Pauli 間的共變異。
    """
    labels = group.paulis.to_labels()
    coeffs = np.real(group.coeffs)
    values, freqs = [], []
    for bitstring, n_counts in counts.items():
        bits = bitstring.replace(" ", "")
        y = 0.0
        for label, c in zip(labels, coeffs):
            sign = 1
            for q, ch in enumerate(reversed(label)):
                if ch != "I" and bits[len(bits) - 1 - q] == "1":
                    sign = -sign
            y += float(c) * sign
        values.append(y)
        freqs.append(n_counts)
    values, freqs = np.array(values), np.array(freqs)
    n_shots = int(freqs.sum())
    mean = float(np.sum(values * freqs) / n_shots)
    if n_shots > 1:
        var = float(np.sum(freqs * (values - mean) ** 2) / (n_shots - 1))
        se = float(np.sqrt(var / n_shots))
    else:
        se = float("nan")
    return mean, se, n_shots


def estimate_energy_shots(qc_bound, h_spo, total_shots, backend,
                          layout=None, fake_backend=None) -> dict:
    """固定參數下的有限 shot 能量估計（不重新最佳化）。"""
    groups = h_spo.group_commuting(qubit_wise=True)
    shots_alloc = allocate_shots(total_shots, len(groups))
    total_mean, total_var, per_group = 0.0, 0.0, []
    for group, shots in zip(groups, shots_alloc):
        meas_qc, basis = group_measurement_circuit(qc_bound, group)
        if fake_backend is not None:
            tqc = transpile(meas_qc, backend=fake_backend, initial_layout=layout,
                            seed_transpiler=SEED, optimization_level=3)
        else:
            tqc = transpile(meas_qc, backend=backend,
                            seed_transpiler=SEED, optimization_level=1)
        counts = backend.run(tqc, shots=shots,
                             seed_simulator=SEED).result().get_counts()
        mean, se, n = group_statistics(counts, group)
        total_mean += mean
        if np.isfinite(se):
            total_var += se ** 2
        per_group.append({"basis": "".join(reversed(basis)), "shots": n,
                          "mean": mean, "se": se})
    return {"total_shots": total_shots, "mean": total_mean,
            "se": float(np.sqrt(total_var)), "groups": per_group}


def fake_backend_summary(fake) -> dict:
    """擷取 noise model 主要參數（中位數）供報告引用。"""
    t1s, t2s = [], []
    for q in range(fake.num_qubits):
        props = fake.qubit_properties(q)
        if props.t1 is not None:
            t1s.append(props.t1)
        if props.t2 is not None:
            t2s.append(props.t2)
    target = fake.target
    def median_error(op_name):
        errs = []
        if op_name in target.operation_names:
            for _, props in target[op_name].items():
                if props is not None and props.error is not None:
                    errs.append(props.error)
        return float(np.median(errs)) if errs else None
    return {
        "backend": fake.name,
        "n_qubits": fake.num_qubits,
        "median_T1_us": float(np.median(t1s) * 1e6),
        "median_T2_us": float(np.median(t2s) * 1e6),
        "median_sx_error": median_error("sx"),
        "median_ecr_error": median_error("ecr"),
        "median_readout_error": median_error("measure"),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Problem 4: VQE for H2 (statevector + finite shots)")
    print("package versions:", package_versions())
    print(f"shared settings: {OPTIMIZER_SETTINGS}")
    print("=" * 72)

    ham = build_hamiltonians()
    cases = build_cases(ham)

    # ---- (a)(b) 四種 VQE ----
    print("\n(a)(b) VQE results (noiseless statevector):")
    results = {}
    print(f"    {'case':<24s} {'E_VQE (Ha)':>14s} {'E_exact':>13s} "
          f"{'abs err':>10s} {'#par':>5s} {'nfev':>5s}")
    for name, (qc, h_spo, e_exact) in cases.items():
        r = run_vqe(qc, h_spo)
        r["E_exact"] = e_exact
        r["abs_error"] = abs(r["energy"] - e_exact)
        results[name] = r
        print(f"    {name:<24s} {r['energy']:>14.8f} {e_exact:>13.8f} "
              f"{r['abs_error']:>10.2e} {r['n_params']:>5d} {r['nfev']:>5d}")

    for name in ("UCCSD+JW", "UCCSD+parity"):
        assert results[name]["abs_error"] < 1e-7, f"{name} failed to reach FCI"
    for name in ("real-amplitude+JW", "real-amplitude+parity"):
        assert results[name]["abs_error"] < 1.6e-3, \
            f"{name} misses chemical accuracy"
    assert abs(ham["E_exact_parity"] - REFS["H2"]["E_FCI"]) < 5e-5

    print("\n    optimized parameters:")
    for name, r in results.items():
        with np.printoptions(precision=6, suppress=True):
            print(f"    {name:<24s} {np.array(r['params'])}")

    # ---- (c) 有限 shots（最佳 parity 電路，固定參數）----
    parity_cases = {k: v for k, v in results.items() if "parity" in k}
    best_name = min(parity_cases, key=lambda k: parity_cases[k]["abs_error"])
    qc_best, h_par, _ = cases[best_name]
    qc_bound = qc_best.assign_parameters(results[best_name]["params"])
    e_noiseless = results[best_name]["energy"]
    print(f"\n(c) finite-shot estimation with best parity circuit: {best_name}")
    print(f"    parameters fixed at noiseless optimum, E_ref = {e_noiseless:+.8f} Ha")

    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane

    fake = FakeBrisbane()
    noise_info = fake_backend_summary(fake)
    ideal_backend = AerSimulator()
    noisy_backend = AerSimulator.from_backend(fake)
    print(f"    ideal backend : AerSimulator (shot-based, noiseless)")
    print(f"    noisy backend : AerSimulator.from_backend(FakeBrisbane) -- "
          f"target QPU ibm_brisbane")
    print(f"    noise model   : {noise_info}")
    print(f"    shot allocation: equal split over qubit-wise commuting groups; "
          f"physical qubits [0, 1]")

    shot_results = {}
    print(f"\n    {'backend':<8s} {'shots':>7s} {'mean (Ha)':>13s} "
          f"{'SE (Ha)':>10s} {'mean-E_ref':>11s}")
    for label, backend, fake_arg in (("ideal", ideal_backend, None),
                                     ("noisy", noisy_backend, fake)):
        for shots in (101, 10_000):
            est = estimate_energy_shots(qc_bound, h_par, shots, backend,
                                        layout=[0, 1], fake_backend=fake_arg)
            shot_results[f"{label}_{shots}"] = est
            print(f"    {label:<8s} {shots:>7d} {est['mean']:>13.6f} "
                  f"{est['se']:>10.6f} {est['mean'] - e_noiseless:>+11.6f}")

    # sanity：ideal 10^4 shots 的估計應落在 5 個標準誤內
    est = shot_results["ideal_10000"]
    assert abs(est["mean"] - e_noiseless) < 5 * est["se"] + 1e-6, \
        "ideal shot estimate inconsistent with statevector optimum"

    # ---- 存檔 ----
    save_results("problem4", {
        "versions": package_versions(),
        "optimizer_settings": OPTIMIZER_SETTINGS,
        "ab_vqe": {
            name: {k: v for k, v in r.items()}
            for name, r in results.items()
        },
        "E_FCI_ref": REFS["H2"]["E_FCI"],
        "c_best_parity_case": best_name,
        "c_backends": {
            "ideal": "qiskit-aer AerSimulator (shot-based, no noise)",
            "noisy": "AerSimulator.from_backend(FakeBrisbane)",
            "target_qpu": "ibm_brisbane (127-qubit Eagle r3)",
            "noise_model_source": "qiskit-ibm-runtime fake provider "
                                  "calibration snapshot",
            "noise_model_parameters": noise_info,
        },
        "c_shot_allocation_rule": "equal split over qubit-wise commuting groups, "
                                  "remainder to leading groups",
        "c_shot_results": shot_results,
        "seed": SEED,
    })
    print("\nresults/problem4.json written. All Problem 4 checks passed.")


if __name__ == "__main__":
    main()
