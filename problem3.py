"""
problem3.py -- QCAA HW3 Problem 3: Integrals and Ansatz Circuits
=================================================================

(a) 以正則 RHF 空間軌道 (psi1=sigma_g, psi2=sigma_u) 計算
      h_pq (p,q in {1,2})、六個對稱不等價 ERI、E_nuc。
    積分慣例（報告要求明確陳述）：
      * 引擎：PennyLane 0.44 微分式 RHF（PySCF 無原生 Windows 版；
        附錄 B 之 PySCF 路徑在 Linux/WSL 可重現相同數值）
      * 題目要求的 <pq|rs> 為物理學家記號（Problem 1 Eq. 3）；
        化學家記號 (pq|rs) 經 <pq|rs> = (pr|qs) 轉換（附錄 A）
      * 驗證實軌道 8 重置換對稱（附錄 B Eq. 16）
      * 對稱性檢查：含奇數個 sigma_u 的積分（<11|12>、<12|22>）因
        g/u 宇稱而為 0

(b) 用這組積分建構 JW / BK / 2-qubit parity-tapered Hamiltonian，
    以精確對角化驗證三者在中性 N_alpha=N_beta=1 sector 給出相同基態能，
    且重現 benchmark E_FCI = -1.13727 Ha。

(c) UCCSD ansatz（一階 Trotter）：
      * 最小基底 H2 僅一個必要 double 振幅 theta（singles 因 Brillouin
        定理 + g/u 對稱而消失；程式以能量梯度 <HF|[H,G_single]|HF> = 0 驗證）
      * 產生器 G = theta (a3† a2† a1 a0 - h.c.)，JW 下為 8 條 weight-4
        Pauli 字串（係數 ±theta/8、彼此對易 -> 一階 Trotter 為精確）
      * parity-tapered 下塌縮為 2-qubit 單參數旋轉
      * 電路圖輸出 figures/problem3_uccsd_{jw,parity}.png
      * benchmark：theta 掃描的最低能量須達 E_FCI（誤差 < 1e-8 Ha）

(d) Real-amplitude hardware-efficient ansatz：
      * JW (4 qubits)：HF 初態 |1100> (X q0,q1)，RY 旋轉層 + linear CNOT
        entangler，2 層 + final rotation layer，12 參數
      * parity (2 qubits)：tapered HF 初態 |01> (X q0)，RY + 單一 CNOT
        entangler，2 層 + final rotation layer，6 參數
      * 電路圖輸出 figures/problem3_hea_{jw,parity}.png

qubit 慣例：OpenFermion 模式 p <-> Qiskit qubit p（little-endian 字串轉換
已於 hw3_common.qubit_operator_to_sparse_pauli_op 處理）。
"""

from __future__ import annotations

import numpy as np
from openfermion import FermionOperator, jordan_wigner
from scipy.optimize import minimize_scalar

from hw3_common import (
    FIGURES_DIR,
    REFS,
    build_fermionic_hamiltonian,
    check_eri_permutation_symmetry,
    fermionic_sector_eigenvalues,
    get_h2_integrals,
    map_bravyi_kitaev,
    map_jordan_wigner,
    map_parity,
    map_parity_tapered,
    operator_matrix,
    package_versions,
    parity_symmetry_eigenvalues,
    pauli_term_table,
    qubit_operator_to_sparse_pauli_op,
    save_results,
    sector_eigenvalues_penalty,
    spatial_to_spin_orbital,
    taper_qubits,
    total_number_operators,
)

N_MODES = 4
N_SPATIAL = 2


# ---------------------------------------------------------------------------
# (a) 積分報表
# ---------------------------------------------------------------------------

def part_a(mol) -> dict:
    print("\n(a) RHF canonical MO integrals (Ha)")
    print(f"    engine: PennyLane {package_versions()['pennylane']} "
          "(differentiable RHF; PySCF route of Appendix B reproduces "
          "identical values on Linux/WSL)")
    print(f"    E_nuc = {mol.e_nuc:+.8f}")
    print(f"    h_pq  = [[{mol.h_mo[0,0]:+.8f}, {mol.h_mo[0,1]:+.8f}],")
    print(f"             [{mol.h_mo[1,0]:+.8f}, {mol.h_mo[1,1]:+.8f}]]")

    # 六個對稱不等價 ERI：物理學家 <pq|rs> 與化學家 (pq|rs) 對照
    # （one-based 軌道標籤；<pq|rs> = (pr|qs)）
    unique = [
        ("<11|11>", (0, 0, 0, 0), "(11|11)"),
        ("<11|12>", (0, 0, 0, 1), "(11|12)"),
        ("<11|22>", (0, 0, 1, 1), "(12|12)"),
        ("<12|12>", (0, 1, 0, 1), "(11|22)"),
        ("<12|22>", (0, 1, 1, 1), "(12|22)"),
        ("<22|22>", (1, 1, 1, 1), "(22|22)"),
    ]
    rows = {}
    print(f"    {'physicist':<10s} {'value (Ha)':>14s}   chemist equivalent")
    for label, idx, chem_label in unique:
        v = float(mol.eri_phys[idx])
        rows[label] = v
        print(f"    {label:<10s} {v:>14.8f}   = {chem_label}")

    dev = check_eri_permutation_symmetry(mol.eri_phys)
    print(f"    8-fold permutation symmetry (Eq. 16): max deviation = {dev:.2e}")

    # g/u 宇稱：奇數個 sigma_u 的積分為 0
    assert abs(rows["<11|12>"]) < 1e-10 and abs(rows["<12|22>"]) < 1e-10, \
        "odd-u-parity integrals should vanish"
    print("    parity check: <11|12> = <12|22> = 0 (odd number of sigma_u) OK")
    assert abs(mol.e_rhf - REFS["H2"]["E_RHF"]) < 5e-5
    print(f"    E_RHF = {mol.e_rhf:+.8f} Ha (ref {REFS['H2']['E_RHF']:+.5f})")
    return {"e_nuc": mol.e_nuc, "h_mo": mol.h_mo, "unique_eri_physicist": rows,
            "perm_symmetry_dev": dev}


# ---------------------------------------------------------------------------
# (b) 三種映射 + 精確對角化
# ---------------------------------------------------------------------------

def part_b(mol) -> dict:
    h_so, eri_so = spatial_to_spin_orbital(mol.h_mo, mol.eri_phys, "interleaved")
    fop = build_fermionic_hamiltonian(h_so, eri_so, constant=mol.e_nuc)
    jw_h = map_jordan_wigner(fop)
    bk_h = map_bravyi_kitaev(fop, N_MODES)

    h_so_b, eri_so_b = spatial_to_spin_orbital(mol.h_mo, mol.eri_phys, "blocked")
    fop_b = build_fermionic_hamiltonian(h_so_b, eri_so_b, constant=mol.e_nuc)
    parity_h, taper_info = map_parity_tapered(fop_b, N_MODES, n_alpha=1, n_beta=1)

    # sector 基態能比較
    e_ferm = fermionic_sector_eigenvalues(fop, N_MODES, 1, 1, N_SPATIAL)[0]
    na_f, nb_f = total_number_operators(N_SPATIAL, "interleaved")
    e_jw = sector_eigenvalues_penalty(
        jw_h, map_jordan_wigner(na_f), map_jordan_wigner(nb_f),
        N_MODES, 1, 1, N_SPATIAL)[0]
    e_bk = sector_eigenvalues_penalty(
        bk_h, map_bravyi_kitaev(na_f, N_MODES), map_bravyi_kitaev(nb_f, N_MODES),
        N_MODES, 1, 1, N_SPATIAL)[0]
    e_par = float(np.linalg.eigvalsh(operator_matrix(parity_h, 2))[0])

    print("\n(b) exact diagonalization, neutral sector ground state (Ha):")
    for name, e in [("fermionic", e_ferm), ("JW", e_jw),
                    ("BK", e_bk), ("parity-tapered", e_par)]:
        print(f"    {name:<15s} {e:+.8f}")
    devs = max(abs(e - e_ferm) for e in (e_jw, e_bk, e_par))
    print(f"    max deviation = {devs:.3e},  "
          f"vs E_FCI ref: {abs(e_ferm - REFS['H2']['E_FCI']):.2e}")
    assert devs < 1e-9 and abs(e_ferm - REFS["H2"]["E_FCI"]) < 5e-5

    return {
        "jw_h": jw_h, "bk_h": bk_h, "parity_h": parity_h,
        "fop_b": fop_b, "taper_info": taper_info,
        "energies": {"fermionic": float(e_ferm), "JW": float(e_jw),
                     "BK": float(e_bk), "parity_tapered": e_par},
        "h_jw_spo": qubit_operator_to_sparse_pauli_op(jw_h, 4),
        "h_parity_spo": qubit_operator_to_sparse_pauli_op(parity_h, 2),
    }


# ---------------------------------------------------------------------------
# (c) UCCSD ansatz
# ---------------------------------------------------------------------------

def double_excitation_generator(ordering: str) -> FermionOperator:
    """G = a†_v_alpha a†_v_beta a_o_beta a_o_alpha - h.c.（反 Hermitian，係數 1）。

    interleaved: occupied = (0,1), virtual = (2,3)
    blocked    : occupied = (0,2), virtual = (1,3)
    """
    if ordering == "interleaved":
        oa, ob, va, vb = 0, 1, 2, 3
    else:
        oa, ob, va, vb = 0, 2, 1, 3
    t = FermionOperator(((va, 1), (vb, 1), (ob, 0), (oa, 0)), 1.0)
    return t - FermionOperator(((oa, 1), (ob, 1), (vb, 0), (va, 0)), 1.0)


def check_brillouin(jw_h_mat: np.ndarray, hf_vec: np.ndarray) -> float:
    """singles 能量梯度 <HF|[H, G_single]|HF| 的最大絕對值（應為 0）。"""
    max_grad = 0.0
    for spin in (0, 1):                      # alpha: 0->2, beta: 1->3
        o, v = spin, spin + 2
        g = FermionOperator(((v, 1), (o, 0)), 1.0) \
            - FermionOperator(((o, 1), (v, 0)), 1.0)
        g_mat = operator_matrix(jordan_wigner(g), N_MODES)
        grad = hf_vec.conj() @ (jw_h_mat @ g_mat - g_mat @ jw_h_mat) @ hf_vec
        max_grad = max(max_grad, abs(complex(grad)))
    return max_grad


def hf_index_jw() -> int:
    """JW interleaved HF |1100>：模式 0,1 佔據 -> OF 基底索引（位元 n-1-m）。"""
    return (1 << 3) | (1 << 2)


def uccsd_generator_spos():
    """UCCSD 產生器的 Hermitian SparsePauliOp（供本題與 Problem 4 共用）。

    回傳 (h_gen_jw, h_gen_parity)，皆滿足 exp(-i theta H_gen) = exp(theta G)。
    """
    g_jw = jordan_wigner(double_excitation_generator("interleaved"))
    h_gen_jw = qubit_operator_to_sparse_pauli_op(1j * g_jw, 4)

    g_par_full = map_parity(double_excitation_generator("blocked"), N_MODES)
    z_a, z_t = parity_symmetry_eigenvalues(1, 1)
    g_par = taper_qubits(g_par_full, {1: z_a, 3: z_t}, N_MODES)
    h_gen_par = qubit_operator_to_sparse_pauli_op(1j * g_par, 2)
    return h_gen_jw, h_gen_par, g_jw, g_par


def evolution_circuit(generator_spo, n_qubits: int, prep_x: list[int]):
    """HF 製備 + exp(-i theta H_gen) 電路（一階 Trotter；此處各項對易故精確）。"""
    from qiskit.circuit import Parameter, QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate

    theta = Parameter("theta")
    qc = QuantumCircuit(n_qubits)
    for q in prep_x:
        qc.x(q)
    qc.append(PauliEvolutionGate(generator_spo, time=theta), range(n_qubits))
    return qc, theta


def energy_curve(qc, theta_param, h_spo):
    """回傳 E(theta) 函數（statevector 期望值）。"""
    from qiskit.quantum_info import Statevector

    def energy(t: float) -> float:
        bound = qc.assign_parameters({theta_param: float(t)})
        return float(np.real(Statevector(bound).expectation_value(h_spo)))

    return energy


def draw_circuit(qc, fname: str, decompose_basis=None) -> str:
    """輸出電路圖 PNG（必要時先轉成基本閘）。"""
    import matplotlib

    matplotlib.use("Agg")
    from qiskit import transpile

    if decompose_basis:
        qc = transpile(qc, basis_gates=decompose_basis, optimization_level=0)
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / fname
    fig = qc.draw("mpl", fold=28)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return str(path), qc


def part_c(mol, ham) -> dict:
    print("\n(c) UCCSD ansatz (single essential double amplitude)")

    # --- Brillouin / 對稱性：singles 梯度為 0 ---
    jw_h_mat = operator_matrix(
        map_jordan_wigner(
            build_fermionic_hamiltonian(
                *spatial_to_spin_orbital(mol.h_mo, mol.eri_phys, "interleaved"),
                constant=mol.e_nuc)), N_MODES)
    hf_vec = np.zeros(2 ** N_MODES)
    hf_vec[hf_index_jw()] = 1.0
    grad = check_brillouin(jw_h_mat, hf_vec)
    print(f"    singles energy gradient at HF: max |<HF|[H,G_s]|HF>| = {grad:.3e}")
    assert grad < 1e-12, "Brillouin condition violated"

    # --- 產生器（G 反 Hermitian、純虛係數；乘 1j 得 Hermitian H_gen，
    #     exp(-i theta H_gen) = exp(theta G)）---
    h_gen_jw, h_gen_par, g_jw, g_par = uccsd_generator_spos()
    jw_strings = pauli_term_table(1j * g_jw)
    print(f"    JW generator: {len(jw_strings)} weight-4 Pauli strings, "
          "mutually commuting -> 1st-order Trotter exact")
    for r in jw_strings:
        print(f"      {r['pauli']:<14s} {r['coeff']:+.4f} * theta")

    par_strings = pauli_term_table(1j * g_par)
    print(f"    parity-tapered generator ({len(par_strings)} terms):")
    for r in par_strings:
        print(f"      {r['pauli']:<14s} {r['coeff']:+.4f} * theta")

    # --- 電路 + theta 掃描（benchmark：達到 E_FCI）---
    qc_jw, th_jw = evolution_circuit(h_gen_jw, 4, prep_x=[0, 1])
    qc_par, th_par = evolution_circuit(h_gen_par, 2, prep_x=[0])
    e_jw = energy_curve(qc_jw, th_jw, ham["h_jw_spo"])
    e_par = energy_curve(qc_par, th_par, ham["h_parity_spo"])

    # theta = 0 應回到 HF 能量
    assert abs(e_jw(0.0) - mol.e_rhf) < 1e-10
    assert abs(e_par(0.0) - mol.e_rhf) < 1e-10
    print(f"    E(theta=0) = E_RHF check passed ({e_jw(0.0):+.8f} Ha)")

    res_jw = minimize_scalar(e_jw, bounds=(-np.pi / 2, np.pi / 2), method="bounded",
                             options={"xatol": 1e-12})
    res_par = minimize_scalar(e_par, bounds=(-np.pi / 2, np.pi / 2), method="bounded",
                              options={"xatol": 1e-12})
    e_fci = ham["energies"]["fermionic"]
    print(f"    UCCSD+JW    : theta* = {res_jw.x:+.8f}, "
          f"E = {res_jw.fun:+.8f} Ha (err vs FCI {abs(res_jw.fun - e_fci):.2e})")
    print(f"    UCCSD+parity: theta* = {res_par.x:+.8f}, "
          f"E = {res_par.fun:+.8f} Ha (err vs FCI {abs(res_par.fun - e_fci):.2e})")
    assert abs(res_jw.fun - e_fci) < 1e-8, "UCCSD+JW fails to reach FCI"
    assert abs(res_par.fun - e_fci) < 1e-8, "UCCSD+parity fails to reach FCI"

    # --- 電路圖 ---
    basis = ["x", "h", "rz", "cx", "rx"]
    fig_jw, qc_jw_dec = draw_circuit(qc_jw, "problem3_uccsd_jw.png", basis)
    fig_par, qc_par_dec = draw_circuit(qc_par, "problem3_uccsd_parity.png", basis)
    ops_jw = dict(qc_jw_dec.count_ops())
    ops_par = dict(qc_par_dec.count_ops())
    print(f"    circuits saved: {fig_jw}")
    print(f"                    {fig_par}")
    print(f"    gate counts after transpile: JW {ops_jw} | parity {ops_par}")

    return {
        "brillouin_gradient": grad,
        "jw_generator_strings": jw_strings,
        "parity_generator_strings": par_strings,
        "theta_opt": {"JW": float(res_jw.x), "parity": float(res_par.x)},
        "E_opt": {"JW": float(res_jw.fun), "parity": float(res_par.fun)},
        "abs_err_vs_FCI": {"JW": float(abs(res_jw.fun - e_fci)),
                           "parity": float(abs(res_par.fun - e_fci))},
        "gate_counts": {"JW": ops_jw, "parity": ops_par},
        "figures": [fig_jw, fig_par],
    }


# ---------------------------------------------------------------------------
# (d) Real-amplitude hardware-efficient ansatz
# ---------------------------------------------------------------------------

def build_hea(n_qubits: int, reps: int, prep_x: list[int]):
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import real_amplitudes

    ansatz = real_amplitudes(n_qubits, entanglement="linear", reps=reps)
    qc = QuantumCircuit(n_qubits)
    for q in prep_x:
        qc.x(q)
    qc.compose(ansatz, inplace=True)
    return qc


def part_d(mol, ham) -> dict:
    from qiskit.quantum_info import Statevector

    print("\n(d) real-amplitude hardware-efficient ansatz")
    reps = 2
    hea_jw = build_hea(4, reps, prep_x=[0, 1])     # HF |1100>
    hea_par = build_hea(2, reps, prep_x=[0])       # tapered HF |01>

    # 初態能量檢查：HF 期望值 = E_RHF（兩種表象一致）
    sv_hf_jw = Statevector.from_label("0011")      # qiskit label: q3 q2 q1 q0
    sv_hf_par = Statevector.from_label("01")
    e0_jw = float(np.real(sv_hf_jw.expectation_value(ham["h_jw_spo"])))
    e0_par = float(np.real(sv_hf_par.expectation_value(ham["h_parity_spo"])))
    print(f"    initial-state check: <HF|H_JW|HF> = {e0_jw:+.8f}, "
          f"<HF_t|H_par|HF_t> = {e0_par:+.8f} (= E_RHF)")
    assert abs(e0_jw - mol.e_rhf) < 1e-10 and abs(e0_par - mol.e_rhf) < 1e-10

    meta = {
        "JW": {
            "initial_state": "|1100> = X(q0) X(q1) |0000>  (HF, interleaved)",
            "rotation_gates": "RY (real amplitudes)",
            "entangler": "CX linear chain q0-q1-q2-q3 (3 CX per layer)",
            "layers": reps, "final_rotation_layer": True,
            "n_parameters": hea_jw.num_parameters,
        },
        "parity": {
            "initial_state": "|01> = X(q0) |00>  (tapered HF)",
            "rotation_gates": "RY (real amplitudes)",
            "entangler": "single CX q0-q1 per layer",
            "layers": reps, "final_rotation_layer": True,
            "n_parameters": hea_par.num_parameters,
        },
    }
    for k, m in meta.items():
        print(f"    {k:>6s}: {m['n_parameters']} parameters, {m['layers']} layers, "
              f"entangler = {m['entangler']}")

    fig_jw, _ = draw_circuit(hea_jw.decompose(), "problem3_hea_jw.png")
    fig_par, _ = draw_circuit(hea_par.decompose(), "problem3_hea_parity.png")
    print(f"    circuits saved: {fig_jw}")
    print(f"                    {fig_par}")
    meta["figures"] = [fig_jw, fig_par]
    return meta


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Problem 3: STO-3G integrals and ansatz circuits for H2")
    print("package versions:", package_versions())
    print("=" * 72)

    mol = get_h2_integrals()
    res_a = part_a(mol)
    ham = part_b(mol)
    res_c = part_c(mol, ham)
    res_d = part_d(mol, ham)

    save_results("problem3", {
        "versions": package_versions(),
        "R_angstrom": REFS["H2"]["R"],
        "integral_convention": "physicist <pq|rs> (Problem 1 Eq. 3); "
                               "chemist (pq|rs) converted via <pq|rs>=(pr|qs)",
        "a_integrals": res_a,
        "b_ground_state_energies": ham["energies"],
        "b_E_FCI_ref": REFS["H2"]["E_FCI"],
        "b_taper_info": ham["taper_info"],
        "c_uccsd": res_c,
        "d_hea": res_d,
    })
    print("\nresults/problem3.json written. All Problem 3 checks passed.")


if __name__ == "__main__":
    main()
