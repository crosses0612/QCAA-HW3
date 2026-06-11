"""
problem5.py -- QCAA HW3 Problem 5: Sample-based Quantum Diagonalization on LiH
===============================================================================

方法宣告（題目要求 state which）：採 **QSCI**（quantum-selected CI，
Kanno et al. 2023）。取樣為無噪聲統計向量取樣，樣本天然落在正確的
(N_alpha, N_beta) sector，故 **不需 SQD 的 configuration recovery**
（對應題目「state whether you applied any configuration recovery」：否）。

實作路線（題目允許 "a custom build on PySCF + OpenFermion + NumPy"；
PySCF 無原生 Windows 版，本實作為 OpenFermion + NumPy custom build，
積分引擎為 PennyLane，數值已對 benchmark 驗證）：

  1. LiH/STO-3G, R = 1.5957 A，全空間即 (4e, 6o) = 12 spin orbitals = 12 qubits
  2. UCCSD 振幅來源（題目要求 state which）：**古典 CCSD**（自寫 spin-orbital
     CCSD，Stanton et al. JCP 94, 4334 (1991) 方程），收斂能量對 benchmark
     E_CCSD = -7.88238 Ha 驗證
  3. 態製備：|psi> = exp(T - T†)|HF>，以稀疏矩陣指數精確作用於統計向量
     （UCCSD ansatz 的零 Trotter 誤差參考實作）
  4. 取樣：機率 |psi_x|^2，rng = default_rng(13705049)（學號 seed）
     抽一條 10^5 shots 的樣本流，shot budgets 10^2..10^5 取其前綴
     （巢狀子空間 -> 變分單調收斂）
  5. 過濾：僅保留 2 alpha + 2 beta 佔據的 bitstring（閉殼層 singlet sector）
  6. 子空間：unique 行列式張成 CI 子空間，古典對角化 (4e,6o) Hamiltonian

bitstring / qubit 順序慣例（題目要求 state）：
  * interleaved spin orbitals：qubit p <-> 自旋軌道 p，
    p = 2k 為第 k 個 spatial MO 的 alpha、p = 2k+1 為 beta
  * OpenFermion 整數基底索引：模式 p <-> 位元 (11 - p)（最高位 = 模式 0）
  * HF = 模式 {0,1,2,3} 佔據（兩個最低 spatial MO 雙佔據）

benchmark（作業 PDF）：
  E_nuc = 0.99488, E_HF = -7.86200, E_CCSD = -7.88238, E_FCI = -7.88239 Ha
"""

from __future__ import annotations

import numpy as np
from numpy import einsum
from openfermion import FermionOperator
from scipy.sparse.linalg import expm_multiply

from hw3_common import (
    FIGURES_DIR,
    REFS,
    SEED,
    build_fermionic_hamiltonian,
    fermionic_sector_eigenvalues,
    get_lih_integrals,
    get_sparse_operator,
    map_jordan_wigner,
    package_versions,
    save_results,
    spatial_to_spin_orbital,
)

N_SPATIAL = 6
N_SO = 12               # spin orbitals = qubits
N_OCC = 4               # 4 electrons -> interleaved 模式 0..3 佔據
N_ALPHA = N_BETA = 2
SHOT_BUDGETS = [100, 1_000, 10_000, 100_000]


# ---------------------------------------------------------------------------
# 古典 spin-orbital CCSD（Stanton et al. 1991 方程）
# ---------------------------------------------------------------------------

def spin_orbital_fock(h_so: np.ndarray, eri_so: np.ndarray) -> np.ndarray:
    """f_pq = h_pq + sum_{i in occ} <pi||qi>。"""
    occ = range(N_OCC)
    f = h_so.copy()
    for i in occ:
        f += eri_so[:, i, :, i] - eri_so[:, i, i, :]
    return f


def ccsd_solve(f: np.ndarray, asym: np.ndarray,
               max_iter: int = 200, tol: float = 1e-11) -> tuple:
    """回傳 (t1, t2, e_corr, n_iter, e_mp2)。

    asym[p,q,r,s] = <pq||rs> = <pq|rs> - <pq|sr>（物理學家記號）。
    o = 佔據 spin orbitals (0..3)，v = 虛 spin orbitals (4..11)。
    """
    o, v = slice(0, N_OCC), slice(N_OCC, N_SO)
    eps = np.diag(f)
    d_ia = eps[o, None] - eps[None, v]
    d_ijab = (eps[o, None, None, None] + eps[None, o, None, None]
              - eps[None, None, v, None] - eps[None, None, None, v])

    fov = f[o, v]
    t1 = np.zeros_like(fov)
    t2 = asym[o, o, v, v] / d_ijab
    e_mp2 = 0.25 * einsum("ijab,ijab->", asym[o, o, v, v], t2)

    def energy(t1, t2):
        e = einsum("ia,ia->", fov, t1)
        e += 0.25 * einsum("ijab,ijab->", asym[o, o, v, v], t2)
        e += 0.5 * einsum("ijab,ia,jb->", asym[o, o, v, v], t1, t1)
        return float(e)

    e_old = energy(t1, t2)
    for it in range(1, max_iter + 1):
        taut = t2 + 0.5 * (einsum("ia,jb->ijab", t1, t1)
                           - einsum("ib,ja->ijab", t1, t1))
        tau = t2 + (einsum("ia,jb->ijab", t1, t1)
                    - einsum("ib,ja->ijab", t1, t1))

        # --- intermediates ---
        Fae = (f[v, v] - np.diag(np.diag(f[v, v]))
               - 0.5 * einsum("me,ma->ae", fov, t1)
               + einsum("mf,mafe->ae", t1, asym[o, v, v, v])
               - 0.5 * einsum("mnaf,mnef->ae", taut, asym[o, o, v, v]))
        Fmi = (f[o, o] - np.diag(np.diag(f[o, o]))
               + 0.5 * einsum("ie,me->mi", t1, fov)
               + einsum("ne,mnie->mi", t1, asym[o, o, o, v])
               + 0.5 * einsum("inef,mnef->mi", taut, asym[o, o, v, v]))
        Fme = fov + einsum("nf,mnef->me", t1, asym[o, o, v, v])

        Wmnij = (asym[o, o, o, o]
                 + einsum("je,mnie->mnij", t1, asym[o, o, o, v])
                 - einsum("ie,mnje->mnij", t1, asym[o, o, o, v])
                 + 0.25 * einsum("ijef,mnef->mnij", tau, asym[o, o, v, v]))
        Wabef = (asym[v, v, v, v]
                 - einsum("mb,amef->abef", t1, asym[v, o, v, v])
                 + einsum("ma,bmef->abef", t1, asym[v, o, v, v])
                 + 0.25 * einsum("mnab,mnef->abef", tau, asym[o, o, v, v]))
        Wmbej = (asym[o, v, v, o]
                 + einsum("jf,mbef->mbej", t1, asym[o, v, v, v])
                 - einsum("nb,mnej->mbej", t1, asym[o, o, v, o])
                 - einsum("jnfb,mnef->mbej",
                          0.5 * t2 + einsum("jf,nb->jnfb", t1, t1),
                          asym[o, o, v, v]))

        # --- T1 ---
        rhs1 = (fov
                + einsum("ie,ae->ia", t1, Fae)
                - einsum("ma,mi->ia", t1, Fmi)
                + einsum("imae,me->ia", t2, Fme)
                - einsum("nf,naif->ia", t1, asym[o, v, o, v])
                - 0.5 * einsum("imef,maef->ia", t2, asym[o, v, v, v])
                - 0.5 * einsum("mnae,nmei->ia", t2, asym[o, o, v, o]))
        t1_new = rhs1 / d_ia

        # --- T2 ---
        Fbe_eff = Fae - 0.5 * einsum("mb,me->be", t1, Fme)
        Fmj_eff = Fmi + 0.5 * einsum("je,me->mj", t1, Fme)
        rhs2 = asym[o, o, v, v].copy()
        tmp = einsum("ijae,be->ijab", t2, Fbe_eff)
        rhs2 += tmp - tmp.swapaxes(2, 3)                      # P_(ab)
        tmp = einsum("imab,mj->ijab", t2, Fmj_eff)
        rhs2 -= tmp - tmp.swapaxes(0, 1)                      # P_(ij)
        rhs2 += 0.5 * einsum("mnab,mnij->ijab", tau, Wmnij)
        rhs2 += 0.5 * einsum("ijef,abef->ijab", tau, Wabef)
        tmp = (einsum("imae,mbej->ijab", t2, Wmbej)
               - einsum("ie,ma,mbej->ijab", t1, t1, asym[o, v, v, o]))
        tmp = tmp - tmp.swapaxes(0, 1)                        # P_(ij)
        rhs2 += tmp - tmp.swapaxes(2, 3)                      # P_(ab)
        tmp = einsum("ie,abej->ijab", t1, asym[v, v, v, o])
        rhs2 += tmp - tmp.swapaxes(0, 1)                      # P_(ij)
        tmp = einsum("ma,mbij->ijab", t1, asym[o, v, o, o])
        rhs2 -= tmp - tmp.swapaxes(2, 3)                      # P_(ab)
        t2_new = rhs2 / d_ijab

        t1, t2 = t1_new, t2_new
        e_new = energy(t1, t2)
        if abs(e_new - e_old) < tol:
            return t1, t2, e_new, it, e_mp2
        e_old = e_new
    raise RuntimeError("CCSD did not converge")


# ---------------------------------------------------------------------------
# UCCSD 態製備
# ---------------------------------------------------------------------------

def cluster_operator(t1: np.ndarray, t2: np.ndarray,
                     tol: float = 1e-12) -> FermionOperator:
    """T - T†（反 Hermitian），佔據/虛指標轉成全域模式編號。"""
    op = FermionOperator()
    for i in range(N_OCC):
        for a in range(N_SO - N_OCC):
            c = float(t1[i, a])
            if abs(c) > tol:
                A = a + N_OCC
                op += FermionOperator(((A, 1), (i, 0)), c)
                op += FermionOperator(((i, 1), (A, 0)), -c)
    for i in range(N_OCC):
        for j in range(N_OCC):
            for a in range(N_SO - N_OCC):
                for b in range(N_SO - N_OCC):
                    c = 0.25 * float(t2[i, j, a, b])
                    if abs(c) > tol:
                        A, B = a + N_OCC, b + N_OCC
                        op += FermionOperator(
                            ((A, 1), (B, 1), (j, 0), (i, 0)), c)
                        op += FermionOperator(
                            ((i, 1), (j, 1), (B, 0), (A, 0)), -c)
    return op


def hf_basis_index() -> int:
    """HF 行列式（模式 0..3 佔據）的 OpenFermion 基底索引。"""
    idx = 0
    for m in range(N_OCC):
        idx |= 1 << (N_SO - 1 - m)
    return idx


def prepare_uccsd_state(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    gen = map_jordan_wigner(cluster_operator(t1, t2))
    g_sparse = get_sparse_operator(gen, n_qubits=N_SO)
    hf = np.zeros(2 ** N_SO, dtype=complex)
    hf[hf_basis_index()] = 1.0
    psi = expm_multiply(g_sparse, hf)
    return psi / np.linalg.norm(psi)


# ---------------------------------------------------------------------------
# QSCI：取樣 -> 過濾 -> 子空間對角化
# ---------------------------------------------------------------------------

def occupation_counts(basis_index: int) -> tuple[int, int]:
    """基底索引 -> (N_alpha, N_beta)。interleaved：偶數模式 = alpha。"""
    n_a = n_b = 0
    for m in range(N_SO):
        if (basis_index >> (N_SO - 1 - m)) & 1:
            if m % 2 == 0:
                n_a += 1
            else:
                n_b += 1
    return n_a, n_b


def run_qsci(psi: np.ndarray, h_sparse, e_fci: float) -> list[dict]:
    """單一 10^5-shot 樣本流，budgets 取前綴（巢狀 -> 變分單調）。"""
    probs = np.abs(psi) ** 2
    probs /= probs.sum()
    rng = np.random.default_rng(SEED)
    stream = rng.choice(len(probs), size=max(SHOT_BUDGETS), p=probs)

    h_csr = h_sparse.tocsr()
    results = []
    for shots in SHOT_BUDGETS:
        sample = stream[:shots]
        kept = [int(x) for x in np.unique(sample)
                if occupation_counts(int(x)) == (N_ALPHA, N_BETA)]
        n_discarded = int(shots - sum(np.isin(sample, kept)))
        d = len(kept)
        h_sub = h_csr[kept][:, kept].toarray()
        e_sqd = float(np.linalg.eigvalsh(h_sub)[0])
        results.append({
            "shots": shots,
            "d_unique_kept": d,
            "n_discarded_shots": n_discarded,
            "E_SQD": e_sqd,
            "abs_error_vs_FCI": abs(e_sqd - e_fci),
        })
    return results


def plot_convergence(results: list[dict], e_fci: float) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = [r["d_unique_kept"] for r in results]
    shots = [r["shots"] for r in results]
    err = [max(r["abs_error_vs_FCI"], 1e-12) for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.semilogy(d, err, "o-")
    ax1.axhline(1.6e-3, color="gray", ls=":", lw=1, label="chemical accuracy")
    ax1.set_xlabel("subspace dimension $d$")
    ax1.set_ylabel(r"$|E_\mathrm{SQD} - E_\mathrm{FCI}|$ (Ha)")
    ax1.set_title("QSCI error vs subspace dimension")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend()

    ax2.loglog(shots, err, "s-", color="tab:red")
    ax2.axhline(1.6e-3, color="gray", ls=":", lw=1, label="chemical accuracy")
    ax2.set_xlabel("shot budget")
    ax2.set_ylabel(r"$|E_\mathrm{SQD} - E_\mathrm{FCI}|$ (Ha)")
    ax2.set_title("QSCI error vs shot count")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()

    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / "problem5_convergence.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Problem 5: QSCI on LiH (4e, 6o) / STO-3G, R = 1.5957 A")
    print("package versions:", package_versions())
    print(f"seed for all sampling: {SEED}")
    print("=" * 72)

    refs = REFS["LiH"]
    mol = get_lih_integrals()
    h_so, eri_so = spatial_to_spin_orbital(mol.h_mo, mol.eri_phys, "interleaved")
    print(f"\nE_nuc = {mol.e_nuc:+.6f} Ha (ref {refs['E_nuc']:+.5f}), "
          f"E_RHF = {mol.e_rhf:+.6f} Ha (ref {refs['E_HF']:+.5f})")
    assert abs(mol.e_rhf - refs["E_HF"]) < 5e-5

    # ---- HF 一致性 + 古典 CCSD ----
    f = spin_orbital_fock(h_so, eri_so)
    asym = eri_so - eri_so.transpose(0, 1, 3, 2)
    occ = slice(0, N_OCC)
    e_hf_check = (float(np.trace(h_so[occ, occ]))
                  + 0.5 * float(einsum("ijij->", asym[occ, occ, occ, occ]))
                  + mol.e_nuc)
    assert abs(e_hf_check - mol.e_rhf) < 1e-10, "spin-orbital HF inconsistency"

    t1, t2, e_corr, n_iter, e_mp2 = ccsd_solve(f, asym)
    e_ccsd = mol.e_rhf + e_corr
    print(f"\nclassical CCSD (custom spin-orbital implementation, "
          f"Stanton et al. equations):")
    print(f"    E_MP2  = {mol.e_rhf + e_mp2:+.6f} Ha")
    print(f"    E_CCSD = {e_ccsd:+.8f} Ha (ref {refs['E_CCSD']:+.5f}, "
          f"diff {abs(e_ccsd - refs['E_CCSD']):.2e}), {n_iter} iterations")
    assert abs(e_ccsd - refs["E_CCSD"]) < 5e-5, "CCSD benchmark failed"
    print(f"    max |t1| = {np.max(np.abs(t1)):.6f}, "
          f"max |t2| = {np.max(np.abs(t2)):.6f}")

    # ---- Hamiltonian 與 FCI 參考 ----
    fop = build_fermionic_hamiltonian(h_so, eri_so, constant=mol.e_nuc)
    e_fci = float(fermionic_sector_eigenvalues(
        fop, N_SO, N_ALPHA, N_BETA, N_SPATIAL, k=1)[0])
    print(f"\nE_FCI (sector ED) = {e_fci:+.8f} Ha (ref {refs['E_FCI']:+.5f})")
    assert abs(e_fci - refs["E_FCI"]) < 5e-5
    h_sparse = get_sparse_operator(map_jordan_wigner(fop), n_qubits=N_SO)

    # ---- UCCSD 態製備（CCSD 振幅）----
    psi = prepare_uccsd_state(t1, t2)
    e_ucc = float(np.real(psi.conj() @ (h_sparse @ psi)))
    print(f"UCCSD(CCSD amplitudes) state energy <psi|H|psi> = {e_ucc:+.8f} Ha "
          f"(err vs FCI {abs(e_ucc - e_fci):.2e})")
    # 取樣分布的 sector 純度（UCCSD 保粒子數 -> 應為 1）
    probs = np.abs(psi) ** 2
    sector_prob = sum(p for x, p in enumerate(probs)
                      if occupation_counts(x) == (N_ALPHA, N_BETA))
    print(f"probability weight in (2a,2b) sector = {sector_prob:.12f}")
    assert sector_prob > 1.0 - 1e-10

    # ---- QSCI ----
    print(f"\nQSCI (no configuration recovery; noiseless sampling, "
          f"nested shot prefixes of one 10^5-shot stream):")
    results = run_qsci(psi, h_sparse, e_fci)
    print(f"    {'shots':>8s} {'d':>5s} {'discarded':>9s} "
          f"{'E_SQD (Ha)':>14s} {'|err| (Ha)':>12s}")
    for r in results:
        print(f"    {r['shots']:>8d} {r['d_unique_kept']:>5d} "
              f"{r['n_discarded_shots']:>9d} {r['E_SQD']:>14.8f} "
              f"{r['abs_error_vs_FCI']:>12.3e}")

    # benchmark / 物理性檢查
    errs = [r["abs_error_vs_FCI"] for r in results]
    for r in results:
        assert r["E_SQD"] >= e_fci - 1e-9, "variational bound violated"
        assert r["d_unique_kept"] <= 225, "subspace exceeds sector dimension"
    assert all(e2 <= e1 + 1e-12 for e1, e2 in zip(errs, errs[1:])), \
        "nested subspaces must converge monotonically"
    assert errs[-1] < 1.6e-3, "10^5 shots should reach chemical accuracy"

    fig_path = plot_convergence(results, e_fci)
    print(f"    convergence figure: {fig_path}")

    # ---- 存檔 ----
    save_results("problem5", {
        "versions": package_versions(),
        "method": "QSCI (Kanno et al. 2023); custom OpenFermion+NumPy build; "
                  "no configuration recovery (noiseless sampling, samples "
                  "natively in the correct sector)",
        "amplitude_source": "classical CCSD (custom spin-orbital NumPy "
                            "implementation, verified against E_CCSD benchmark)",
        "state_preparation": "exact statevector application of exp(T - T+) "
                             "to |HF> (zero-Trotter-error UCCSD reference)",
        "conventions": {
            "ordering": "interleaved: qubit p <-> spin orbital p; even p = "
                        "alpha, odd p = beta; OpenFermion basis bit (11-p)",
            "hf_determinant": "modes 0..3 occupied (basis index 3840)",
            "sampling": f"default_rng({SEED}); single 1e5-shot stream, "
                        "budgets are prefixes (nested subspaces)",
        },
        "benchmarks": {
            "E_nuc": mol.e_nuc, "E_RHF": mol.e_rhf,
            "E_MP2": mol.e_rhf + e_mp2, "E_CCSD": e_ccsd,
            "E_FCI_sector_ED": e_fci,
            "E_UCCSD_state": e_ucc,
            "refs": refs,
        },
        "ccsd_n_iter": n_iter,
        "sector_probability": float(sector_prob),
        "qsci_results": results,
        "sector_dimension_max": 225,
        "figure": fig_path,
        "seed": SEED,
    })
    print("\nresults/problem5.json written. All Problem 5 checks passed.")


if __name__ == "__main__":
    main()
