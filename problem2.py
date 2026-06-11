"""
problem2.py -- QCAA HW3 Problem 2: Qubit Mappings (JW / BK / parity-tapered)
=============================================================================

(a) 數值驗證 JW 變換保持費米子反交換關係
        {a_p, a†_q} = delta_pq,  {a_p, a_q} = {a†_p, a†_q} = 0
    並驗證數算符的 JW 表式  n_p = a†_p a_p -> (I - Z_p)/2
    （Z 字串互相抵消；解析推導見報告）。

(b) 由 Problem 3 的 STO-3G 積分（R = 0.7414 A）產生 JW、BK 與
    2-qubit parity-tapered Hamiltonian，列出全部 Pauli 項與係數，
    並驗證 JW Hamiltonian 具有作業 Eq. (10) 的結構：
        I + 4 個 Z + 6 個 ZZ + b11 (X0 Y1 Y2 X3 + Y0 X1 X2 Y3
                                     - X0 X1 Y2 Y3 - Y0 Y1 X2 X3)

(c) 在中性 N_alpha = N_beta = 1 sector 比較四個本徵值：
    fermionic（行列式基底投影，定義上的 sector 能譜）、
    JW / BK（懲罰法投影）、parity-tapered（2-qubit 全譜即該 sector）。
    最低本徵值須重現 E_FCI benchmark = -1.13727 Ha。

(d) 比較表：qubit 數、非恆等 Pauli 項數、最大 Pauli weight。

慣例（報告要求明確陳述）：
  * JW / BK：interleaved 自旋軌道順序，qubit p <-> 模式 p，
    模式 (0,1,2,3) = (sigma_g alpha, sigma_g beta, sigma_u alpha, sigma_u beta)
    = 作業的 (chi1, chi2, chi3, chi4)；Pauli 字串依 qubit index 升冪。
  * parity-tapered：blocked 順序（alpha 模式 0,1 = sigma_g/sigma_u alpha，
    beta 模式 2,3），parity 編碼後移除 qubit 1（alpha 宇稱）與
    qubit 3（總宇稱），固定本徵值 Z1 = (-1)^N_alpha = -1、
    Z3 = (-1)^(N_alpha+N_beta) = +1。
"""

from __future__ import annotations

import numpy as np
from openfermion import FermionOperator, QubitOperator, jordan_wigner

from hw3_common import (
    REFS,
    build_fermionic_hamiltonian,
    get_h2_integrals,
    map_bravyi_kitaev,
    map_jordan_wigner,
    map_parity_tapered,
    mapping_summary,
    operator_matrix,
    package_versions,
    pauli_term_table,
    fermionic_sector_eigenvalues,
    save_results,
    sector_eigenvalues_penalty,
    spatial_to_spin_orbital,
    total_number_operators,
)

N_MODES = 4
N_SPATIAL = 2


# ---------------------------------------------------------------------------
# (a) JW 反交換關係與數算符的數值驗證
# ---------------------------------------------------------------------------

def _qop_max_coeff(op: QubitOperator) -> float:
    return max((abs(c) for c in op.terms.values()), default=0.0)


def verify_jw_anticommutation(n_modes: int = N_MODES) -> float:
    """檢查全部模式對的反交換關係，回傳最大偏差。

    JW 映射後 {A,B} = AB + BA 以 QubitOperator 代數展開：
      {a_p, a†_q} - delta_pq I = 0,  {a_p, a_q} = 0,  {a†_p, a†_q} = 0
    """
    max_dev = 0.0
    for p in range(n_modes):
        for q in range(n_modes):
            a_p = jordan_wigner(FermionOperator(((p, 0),)))
            a_q = jordan_wigner(FermionOperator(((q, 0),)))
            ad_p = jordan_wigner(FermionOperator(((p, 1),)))
            ad_q = jordan_wigner(FermionOperator(((q, 1),)))

            dev1 = a_p * ad_q + ad_q * a_p - QubitOperator((), 1.0 if p == q else 0.0)
            dev2 = a_p * a_q + a_q * a_p
            dev3 = ad_p * ad_q + ad_q * ad_p
            max_dev = max(max_dev, *(_qop_max_coeff(d) for d in (dev1, dev2, dev3)))
    return max_dev


def verify_jw_number_operator(n_modes: int = N_MODES) -> float:
    """驗證 n_p = a†_p a_p 的 JW 表式為 (I - Z_p)/2（Z 字串抵消）。"""
    max_dev = 0.0
    for p in range(n_modes):
        n_p = jordan_wigner(FermionOperator(((p, 1), (p, 0))))
        expected = QubitOperator((), 0.5) - QubitOperator(((p, "Z"),), 0.5)
        max_dev = max(max_dev, _qop_max_coeff(n_p - expected))
    return max_dev


# ---------------------------------------------------------------------------
# (b) JW Hamiltonian 結構檢查（作業 Eq. 10）
# ---------------------------------------------------------------------------

def check_jw_structure(jw_h: QubitOperator) -> dict:
    """驗證 JW Hamiltonian 恰為 Eq. (10) 的 15 項結構並取出 b 係數。"""
    def coeff(*paulis) -> float:
        term = tuple(sorted(paulis))
        c = jw_h.terms.get(term, 0.0)
        return float(np.real(c))

    rows = pauli_term_table(jw_h)
    by_weight = {w: [r for r in rows if r["weight"] == w] for w in (0, 1, 2, 4)}
    assert len(rows) == 15, f"expect 15 terms (Eq. 10), got {len(rows)}"
    assert len(by_weight[1]) == 4 and all(
        r["pauli"].startswith("Z") for r in by_weight[1]
    ), "expect single-qubit terms Z0..Z3"
    assert len(by_weight[2]) == 6 and all(
        set(p[0] for p in r["pauli"].split()) == {"Z"} for r in by_weight[2]
    ), "expect six ZZ terms"
    assert len(by_weight[4]) == 4, "expect four weight-4 XY terms"

    # b11 符號模式：XYYX = YXXY = -XXYY = -YYXX
    c_xyyx = coeff((0, "X"), (1, "Y"), (2, "Y"), (3, "X"))
    c_yxxy = coeff((0, "Y"), (1, "X"), (2, "X"), (3, "Y"))
    c_xxyy = coeff((0, "X"), (1, "X"), (2, "Y"), (3, "Y"))
    c_yyxx = coeff((0, "Y"), (1, "Y"), (2, "X"), (3, "X"))
    assert abs(c_xyyx - c_yxxy) < 1e-12, "XYYX and YXXY coefficients differ"
    assert abs(c_xxyy - c_yyxx) < 1e-12, "XXYY and YYXX coefficients differ"
    assert abs(c_xyyx + c_xxyy) < 1e-12, "sign pattern of Eq.(10) violated"

    b = {
        "b0 (I)": coeff(),
        "b1 (Z0)": coeff((0, "Z")),
        "b2 (Z1)": coeff((1, "Z")),
        "b3 (Z2)": coeff((2, "Z")),
        "b4 (Z3)": coeff((3, "Z")),
        "b5 (Z0Z1)": coeff((0, "Z"), (1, "Z")),
        "b6 (Z0Z2)": coeff((0, "Z"), (2, "Z")),
        "b7 (Z0Z3)": coeff((0, "Z"), (3, "Z")),
        "b8 (Z1Z2)": coeff((1, "Z"), (2, "Z")),
        "b9 (Z1Z3)": coeff((1, "Z"), (3, "Z")),
        "b10 (Z2Z3)": coeff((2, "Z"), (3, "Z")),
        "b11 (XYYX block)": c_xyyx,
    }
    return b


def print_pauli_table(label: str, rows: list[dict]) -> None:
    print(f"\n  {label} Pauli terms ({len(rows)} total):")
    print(f"    {'pauli':<16s} {'weight':>6s} {'coefficient (Ha)':>18s}")
    for r in rows:
        print(f"    {r['pauli']:<16s} {r['weight']:>6d} {r['coeff']:>18.8f}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Problem 2: Qubit mappings for minimal-basis H2")
    print("package versions:", package_versions())
    print("=" * 72)

    # ---- (a) JW 代數驗證 ----
    dev_acr = verify_jw_anticommutation()
    dev_num = verify_jw_number_operator()
    print("\n(a) JW algebra verification (4 modes):")
    print(f"    anticommutation relations: max deviation = {dev_acr:.3e}")
    print(f"    n_p -> (I - Z_p)/2       : max deviation = {dev_num:.3e}")
    assert dev_acr < 1e-12 and dev_num < 1e-12

    # ---- Hamiltonian 建構（與 Problem 3 相同積分）----
    mol = get_h2_integrals()
    print(f"\nH2/STO-3G integrals at R = {REFS['H2']['R']} A, "
          f"E_nuc = {mol.e_nuc:+.8f} Ha (included as identity offset)")

    # interleaved（作業 chi 順序）：JW / BK
    h_so, eri_so = spatial_to_spin_orbital(mol.h_mo, mol.eri_phys, "interleaved")
    fop = build_fermionic_hamiltonian(h_so, eri_so, constant=mol.e_nuc)
    jw_h = map_jordan_wigner(fop)
    bk_h = map_bravyi_kitaev(fop, N_MODES)

    # blocked：parity + 2-qubit tapering
    h_so_b, eri_so_b = spatial_to_spin_orbital(mol.h_mo, mol.eri_phys, "blocked")
    fop_b = build_fermionic_hamiltonian(h_so_b, eri_so_b, constant=mol.e_nuc)
    parity_h, taper_info = map_parity_tapered(fop_b, N_MODES, n_alpha=1, n_beta=1)

    # ---- (b) Pauli 項與係數 ----
    print("\n(b) Pauli terms and coefficients")
    jw_rows = pauli_term_table(jw_h)
    bk_rows = pauli_term_table(bk_h)
    parity_rows = pauli_term_table(parity_h)
    print_pauli_table("Jordan-Wigner (4 qubits, interleaved ordering)", jw_rows)
    print_pauli_table("Bravyi-Kitaev (4 qubits, interleaved ordering)", bk_rows)
    print_pauli_table("parity-tapered (2 qubits, blocked ordering)", parity_rows)

    b_coeffs = check_jw_structure(jw_h)
    print("\n  JW structure matches assignment Eq. (10); coefficients b_i:")
    for k, v in b_coeffs.items():
        print(f"    {k:<18s} = {v:+.8f}")

    # ---- (c) N_alpha = N_beta = 1 sector 的四個本徵值 ----
    print("\n(c) four eigenvalues in the neutral N_alpha = N_beta = 1 sector")
    print("    Z2 sectors fixed for parity tapering: "
          f"removed qubits {taper_info['removed_qubits']}, "
          f"eigenvalues {taper_info['fixed_eigenvalues']}")

    e_fermion = fermionic_sector_eigenvalues(fop, N_MODES, 1, 1, N_SPATIAL)

    na_f, nb_f = total_number_operators(N_SPATIAL, "interleaved")
    e_jw = sector_eigenvalues_penalty(
        jw_h, map_jordan_wigner(na_f), map_jordan_wigner(nb_f),
        N_MODES, 1, 1, N_SPATIAL,
    )
    e_bk = sector_eigenvalues_penalty(
        bk_h, map_bravyi_kitaev(na_f, N_MODES), map_bravyi_kitaev(nb_f, N_MODES),
        N_MODES, 1, 1, N_SPATIAL,
    )
    e_parity = np.linalg.eigvalsh(operator_matrix(parity_h, 2))

    print(f"\n    {'#':>3s} {'fermionic':>14s} {'JW':>14s} {'BK':>14s} {'parity-2q':>14s}")
    for i in range(4):
        print(f"    {i:>3d} {e_fermion[i]:>14.8f} {e_jw[i]:>14.8f} "
              f"{e_bk[i]:>14.8f} {e_parity[i]:>14.8f}")

    spectra = np.vstack([e_fermion, e_jw, e_bk, e_parity])
    sector_dev = float(np.max(np.abs(spectra - spectra[0])))
    e_fci = float(e_fermion[0])
    print(f"\n    max deviation across mappings = {sector_dev:.3e}")
    print(f"    ground state = {e_fci:+.8f} Ha "
          f"(ref {REFS['H2']['E_FCI']:+.5f}, "
          f"diff {abs(e_fci - REFS['H2']['E_FCI']):.2e})")
    assert sector_dev < 1e-8, "sector spectra disagree across mappings"
    assert abs(e_fci - REFS["H2"]["E_FCI"]) < 5e-5, "E_FCI benchmark failed"

    # ---- (d) 映射比較表 ----
    print("\n(d) mapping comparison")
    summary = {
        "JW": mapping_summary(jw_h, 4),
        "BK": mapping_summary(bk_h, 4),
        "parity-tapered": mapping_summary(parity_h, 2),
    }
    print(f"    {'mapping':<16s} {'qubits':>6s} {'non-I terms':>12s} {'max weight':>11s}")
    for name, s in summary.items():
        print(f"    {name:<16s} {s['n_qubits']:>6d} "
              f"{s['n_non_identity_terms']:>12d} {s['max_pauli_weight']:>11d}")

    # ---- 存檔 ----
    save_results("problem2", {
        "versions": package_versions(),
        "conventions": {
            "jw_bk_ordering": "interleaved: mode (0,1,2,3) = "
                              "(sigma_g a, sigma_g b, sigma_u a, sigma_u b) "
                              "= (chi1, chi2, chi3, chi4); qubit p <-> mode p",
            "parity": taper_info,
            "pauli_string_format": "ascending qubit index",
        },
        "a_anticommutation_max_dev": dev_acr,
        "a_number_operator_max_dev": dev_num,
        "b_pauli_terms": {
            "JW": jw_rows, "BK": bk_rows, "parity_tapered": parity_rows,
        },
        "b_jw_eq10_coefficients": b_coeffs,
        "c_sector_eigenvalues": {
            "fermionic": e_fermion, "JW": e_jw, "BK": e_bk,
            "parity_tapered": e_parity,
        },
        "c_max_deviation": sector_dev,
        "c_E_FCI": e_fci,
        "c_E_FCI_ref": REFS["H2"]["E_FCI"],
        "c_abs_error_vs_ref": abs(e_fci - REFS["H2"]["E_FCI"]),
        "d_summary": summary,
    })
    print("\nresults/problem2.json written. All Problem 2 checks passed.")


if __name__ == "__main__":
    main()
