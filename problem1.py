"""
problem1.py -- QCAA HW3 Problem 1: Minimal-basis H2
====================================================

(a)(b)(c) 的解析推導寫在報告中；本程式的角色是以 STO-3G 數值
**驗證**理論結果，並為 (d) 提供定量佐證：

  (a) 由自旋軌道積分組裝兩行列式閉殼層 FCI block（作業 Eq. 5）
        H_FCI = [[ h11+h22+<12|12>-<12|21> ,  <12|34>-<12|43> ],
                 [ <34|12>-<34|21>         ,  h33+h44+<34|34>-<34|43> ]]
      並檢查：
        * Hermitian（off-diagonal 相等，源自實軌道 ERI 置換對稱 <12|34>=<34|12>）
        * 與二次量子化 Hamiltonian（Problem 1 Eq. 6）在行列式基底
          { |1 1bar> = a†0 a†1 |vac>,  |2 2bar> = a†2 a†3 |vac> }
          下逐元素一致（Slater-Condon 規則的數值驗證）
        * N_alpha=N_beta=1 sector 的完整 4x4 矩陣中，兩個開殼層行列式
          |1 2bar>, |2 1bar> 與閉殼層 block 解耦（g/u 空間對稱），
          故 2x2 block 的最低本徵值即為 FCI 基態能
        * E_FCI 與 benchmark -1.13727 Ha 相符

  (d) 鍵長拉伸掃描：E_RHF / E_FCI / 雙激發行列式權重 |c2|^2 對 R 作圖，
      顯示 R 增大時靜態相關（double excitation）權重上升、
      RHF 單行列式誤差發散。

自旋軌道慣例（與作業 Eq. 4 一致，zero-based）：
    mode (0,1,2,3) = (chi1,chi2,chi3,chi4) = (sigma_g alpha, sigma_g beta,
                                              sigma_u alpha, sigma_u beta)
ERI 一律為物理學家記號 <pq|rs>（Problem 1 Eq. 3）。
"""

from __future__ import annotations

import numpy as np

from hw3_common import (
    FIGURES_DIR,
    REFS,
    build_fermionic_hamiltonian,
    check_eri_permutation_symmetry,
    get_h2_integrals,
    get_sparse_operator,
    package_versions,
    save_results,
    spatial_to_spin_orbital,
)

N_MODES = 4          # H2 / STO-3G: 4 個自旋軌道
HARTREE_FMT = "+.8f"


# ---------------------------------------------------------------------------
# (a) 由積分組裝 2x2 閉殼層 FCI block（作業 Eq. 5）
# ---------------------------------------------------------------------------

def build_fci_block(h_so: np.ndarray, eri_so: np.ndarray) -> np.ndarray:
    """直接照抄 Eq. (5)，one-based chi_i -> zero-based mode i-1。"""
    e_hf = h_so[0, 0] + h_so[1, 1] + eri_so[0, 1, 0, 1] - eri_so[0, 1, 1, 0]
    e_dd = h_so[2, 2] + h_so[3, 3] + eri_so[2, 3, 2, 3] - eri_so[2, 3, 3, 2]
    off_01 = eri_so[0, 1, 2, 3] - eri_so[0, 1, 3, 2]   # <12|34> - <12|43>
    off_10 = eri_so[2, 3, 0, 1] - eri_so[2, 3, 1, 0]   # <34|12> - <34|21>
    return np.array([[e_hf, off_01], [off_10, e_dd]])


# ---------------------------------------------------------------------------
# 與二次量子化 Hamiltonian 的交叉驗證（Slater-Condon 數值版）
# ---------------------------------------------------------------------------

def determinant_index(occupied_modes: tuple[int, ...]) -> int:
    """佔據模式 -> OpenFermion 計算基底索引（模式 m <-> 位元 n-1-m）。

    基底態 = a†_{m1} a†_{m2} ... |vac>，模式依升冪排列（與 |1 1bar> 等
    行列式的標準排序一致，故符號慣例相同）。
    """
    idx = 0
    for m in occupied_modes:
        idx |= 1 << (N_MODES - 1 - m)
    return idx


def sector_matrix(h_elec_sparse, determinants: list[tuple[int, ...]]) -> np.ndarray:
    """在給定行列式基底下取出 Hamiltonian 子矩陣。"""
    idx = [determinant_index(d) for d in determinants]
    return h_elec_sparse[np.ix_(idx, idx)].toarray().real


# ---------------------------------------------------------------------------
# (d) 鍵長拉伸掃描
# ---------------------------------------------------------------------------

def stretch_scan(r_values: np.ndarray) -> dict:
    """對每個鍵長計算 E_RHF、E_FCI（2x2 block）與雙激發權重 |c2|^2。"""
    out = {"R": [], "E_RHF": [], "E_FCI": [], "c2_sq": [], "E_corr": []}
    for r in r_values:
        mol = get_h2_integrals(float(r))
        h_so, eri_so = spatial_to_spin_orbital(mol.h_mo, mol.eri_phys, "interleaved")
        block = build_fci_block(h_so, eri_so)
        vals, vecs = np.linalg.eigh(block)
        e_fci = vals[0] + mol.e_nuc
        c2 = vecs[1, 0]  # 基態中 |2 2bar> 的係數
        out["R"].append(float(r))
        out["E_RHF"].append(mol.e_rhf)
        out["E_FCI"].append(float(e_fci))
        out["c2_sq"].append(float(c2**2))
        out["E_corr"].append(float(e_fci - mol.e_rhf))
    return out


def plot_stretch(scan: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(scan["R"], scan["E_RHF"], "o-", label=r"$E_\mathrm{RHF}$")
    ax1.plot(scan["R"], scan["E_FCI"], "s-", label=r"$E_\mathrm{FCI}$ (2-det)")
    ax1.axvline(REFS["H2"]["R"], color="gray", ls=":", lw=1,
                label=rf"$R_e$ = {REFS['H2']['R']} $\AA$")
    ax1.set_xlabel(r"$R$ ($\AA$)")
    ax1.set_ylabel("Energy (Ha)")
    ax1.set_title(r"H$_2$/STO-3G: RHF vs FCI dissociation")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(scan["R"], scan["c2_sq"], "d-", color="tab:red")
    ax2.axhline(0.5, color="gray", ls=":", lw=1)
    ax2.set_xlabel(r"$R$ ($\AA$)")
    ax2.set_ylabel(r"$|c_2|^2$  (weight of $|2\bar{2}\rangle$)")
    ax2.set_title("Double-excitation weight vs bond stretching")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / "problem1_stretch.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Problem 1: Minimal-basis H2 -- numerical verification")
    print("package versions:", package_versions())
    print("=" * 72)

    # ---- 積分（平衡鍵長）----
    mol = get_h2_integrals()
    check_eri_permutation_symmetry(mol.eri_phys)
    h_so, eri_so = spatial_to_spin_orbital(mol.h_mo, mol.eri_phys, "interleaved")
    print(f"\nR = {REFS['H2']['R']} A,  E_nuc = {mol.e_nuc:{HARTREE_FMT}} Ha")
    print(f"E_RHF = {mol.e_rhf:{HARTREE_FMT}} Ha  (ref {REFS['H2']['E_RHF']:+.5f})")

    # ---- (a) 2x2 FCI block ----
    block = build_fci_block(h_so, eri_so)
    print("\n(a) closed-shell FCI block (electronic, Ha):")
    print(f"    [[{block[0,0]:{HARTREE_FMT}}, {block[0,1]:{HARTREE_FMT}}],")
    print(f"     [{block[1,0]:{HARTREE_FMT}}, {block[1,1]:{HARTREE_FMT}}]]")

    herm_dev = abs(block[0, 1] - np.conj(block[1, 0]))
    print(f"    Hermiticity |H01 - H10*| = {herm_dev:.3e}")
    assert herm_dev < 1e-12, "FCI block is not Hermitian"

    # off-diagonal 的積分恆等式：<12|34> = <34|12>（實軌道置換對稱）
    sym_dev = abs(eri_so[0, 1, 2, 3] - eri_so[2, 3, 0, 1])
    assert sym_dev < 1e-12
    print(f"    <12|34> = <34|12> check: deviation = {sym_dev:.3e}")

    # ---- Slater-Condon 交叉驗證：行列式基底下的二次量子化矩陣元 ----
    fop_elec = build_fermionic_hamiltonian(h_so, eri_so)        # 不含 E_nuc
    h_sparse = get_sparse_operator(fop_elec, n_qubits=N_MODES).tocsc()

    closed_shell = [(0, 1), (2, 3)]                  # |1 1bar>, |2 2bar>
    block_2nd_q = sector_matrix(h_sparse, closed_shell)
    dev = float(np.max(np.abs(block - block_2nd_q)))
    print("\n    cross-check vs second-quantized H in determinant basis:")
    print(f"    max |Eq.(5) - <det_i|H|det_j>| = {dev:.3e}")
    assert dev < 1e-12, "Eq.(5) and second-quantized matrix elements disagree"

    # ---- 開殼層行列式解耦（4x4 sector 矩陣）----
    sector_dets = [(0, 1), (2, 3), (0, 3), (1, 2)]   # 後兩個為開殼層 |1 2bar>, |2 1bar>
    full_sector = sector_matrix(h_sparse, sector_dets)
    coupling = float(np.max(np.abs(full_sector[:2, 2:])))
    print(f"    closed/open-shell coupling (should vanish by g/u symmetry): "
          f"{coupling:.3e}")
    assert coupling < 1e-10, "open-shell determinants unexpectedly couple"

    # ---- 對角化與 benchmark ----
    vals, vecs = np.linalg.eigh(block)
    e_fci = float(vals[0] + mol.e_nuc)
    c1, c2 = float(vecs[0, 0]), float(vecs[1, 0])
    e_corr = e_fci - mol.e_rhf
    print(f"\n    eigenvalues (electronic): {vals[0]:{HARTREE_FMT}}, "
          f"{vals[1]:{HARTREE_FMT}} Ha")
    print(f"    E_FCI total = {e_fci:{HARTREE_FMT}} Ha  "
          f"(ref {REFS['H2']['E_FCI']:+.5f}, "
          f"diff {abs(e_fci - REFS['H2']['E_FCI']):.2e})")
    print(f"    ground state = {c1:+.6f} |1 1bar> {c2:+.6f} |2 2bar>,  "
          f"|c2|^2 = {c2**2:.6f}")
    print(f"    E_corr = {e_corr:{HARTREE_FMT}} Ha")
    assert abs(e_fci - REFS["H2"]["E_FCI"]) < 5e-5, "E_FCI does not match benchmark"
    assert abs(mol.e_rhf - REFS["H2"]["E_RHF"]) < 5e-5, "E_RHF does not match benchmark"

    # 2x2 block 最低本徵值 = HF 態能量上界檢查（變分原理）
    assert vals[0] <= block[0, 0] + 1e-12

    # ---- (d) 鍵長拉伸掃描 ----
    print("\n(d) bond-stretching scan:")
    r_values = np.array([0.5, 0.6, 0.7, 0.7414, 0.9, 1.1, 1.3,
                         1.5, 1.8, 2.1, 2.5, 3.0])
    scan = stretch_scan(r_values)
    fig_path = plot_stretch(scan)
    print(f"    {'R (A)':>7s} {'E_RHF':>12s} {'E_FCI':>12s} "
          f"{'E_corr':>10s} {'|c2|^2':>8s}")
    for i, r in enumerate(scan["R"]):
        print(f"    {r:7.4f} {scan['E_RHF'][i]:12.6f} {scan['E_FCI'][i]:12.6f} "
              f"{scan['E_corr'][i]:10.6f} {scan['c2_sq'][i]:8.4f}")
    print(f"    figure saved: {fig_path}")

    # 物理檢查：|c2|^2 隨 R 單調上升、解離極限趨近 0.5（完全靜態相關）
    assert all(np.diff(scan["c2_sq"]) > 0), "|c2|^2 should grow with R"
    assert scan["c2_sq"][-1] > 0.3, "|c2|^2 should approach 0.5 at dissociation"

    # ---- 存檔供報告引用 ----
    save_results("problem1", {
        "versions": package_versions(),
        "R_angstrom": REFS["H2"]["R"],
        "e_nuc": mol.e_nuc,
        "e_rhf": mol.e_rhf,
        "h_mo": mol.h_mo,
        "spin_orbital_h_diag": np.diag(h_so),
        "fci_block_electronic": block,
        "fci_block_secondquant_dev": dev,
        "hermiticity_dev": herm_dev,
        "open_shell_coupling": coupling,
        "eigenvalues_electronic": vals,
        "E_FCI_total": e_fci,
        "E_FCI_ref": REFS["H2"]["E_FCI"],
        "abs_error_vs_ref": abs(e_fci - REFS["H2"]["E_FCI"]),
        "ground_state_coefficients": {"c1": c1, "c2": c2, "c2_sq": c2**2},
        "E_corr": e_corr,
        "stretch_scan": scan,
        "figure": fig_path,
    })
    print("\nresults/problem1.json written. All Problem 1 checks passed.")


if __name__ == "__main__":
    main()
