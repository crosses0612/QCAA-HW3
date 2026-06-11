"""
hw3_common.py -- QCAA HW3 共用模組
===================================

集中處理所有「容易寫錯」的慣例，供 problem1.py ~ problem5.py 共用：

  1. 分子積分        : PennyLane 微分式 Hartree-Fock 引擎（Windows 上取代 PySCF）
  2. ERI 記號轉換    : PennyLane 內部慣例 -> 化學家 (pq|rs) -> 物理學家 <pq|rs>
  3. 自旋軌道展開    : interleaved（作業慣例）與 blocked（parity tapering 用）
  4. 二次量子化      : H = sum h_pq a†p aq + 1/2 sum <pq|rs> a†p a†q a_s a_r
  5. 費米子->量子位元: Jordan-Wigner / Bravyi-Kitaev / Parity + 2-qubit Z2 tapering
  6. 精確對角化      : 全空間、(N_alpha, N_beta) 粒子數 sector 投影
  7. 工具            : OpenFermion <-> Qiskit 轉換、Pauli 項列表、結果存檔

積分記號（本作業最重要的陷阱）
------------------------------
* 物理學家記號  <pq|rs> = ∫ dx1 dx2  χp*(x1) χq*(x2) r12^-1 χr(x1) χs(x2)
  （作業 Problem 1 Eq.(3)；電子 1 配 p,r、電子 2 配 q,s）
* 化學家記號    (pq|rs) = ∫ dr1 dr2  ψp(r1)ψq(r1) r12^-1 ψr(r2)ψs(r2)
* 實軌道轉換    <pq|rs> = (pr|qs)        （附錄 A）
* PennyLane `electron_integrals` 回傳張量 eri_pl 滿足
      H = sum h[p,q] a†p aq + 1/2 sum eri_pl[p,q,r,s] a†p a†q a_r a_s
  亦即 eri_pl[p,q,r,s] = <pq|sr>_phys = (ps|qr)_chem
  （已用 H2/STO-3G 數值驗證：J=0.6635、K=0.1813、E_RHF=-1.11668 Ha）

自旋軌道順序
------------
* interleaved（作業 Problem 1/2 的 (χ1,χ2,χ3,χ4)，zero-based 模式 0..3）：
      mode 2p + sigma，sigma=0 為 alpha、1 為 beta
      H2: (0,1,2,3) = (ψ1α, ψ1β, ψ2α, ψ2β)
* blocked（parity 2-qubit reduction 需要：先全部 alpha、再全部 beta）：
      alpha: mode p (p=0..n-1)，beta: mode n+p
      parity 編碼下 qubit n-1 儲存 N_alpha 宇稱、qubit 2n-1 儲存總宇稱

隨機種子：SEED = 13705049（學號）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path

import numpy as np
from openfermion import (
    FermionOperator,
    QubitOperator,
    bravyi_kitaev,
    get_sparse_operator,
    jordan_wigner,
)

# ---------------------------------------------------------------------------
# 常數與路徑
# ---------------------------------------------------------------------------

SEED = 13705049                       # 學號，所有隨機步驟共用
ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# 作業 PDF 給定的 benchmark（total energy, Hartree）
REFS = {
    "H2": {"R": 0.7414, "E_FCI": -1.13727, "E_RHF": -1.11668},
    "LiH": {
        "R": 1.5957,
        "E_nuc": 0.99488,
        "E_HF": -7.86200,
        "E_CCSD": -7.88238,
        "E_FCI": -7.88239,
    },
}


# ---------------------------------------------------------------------------
# 1. 分子積分（PennyLane 引擎）
# ---------------------------------------------------------------------------

@dataclass
class MolecularIntegrals:
    """RHF 正則分子軌道下的積分集合（spatial-orbital 層級）。

    eri_chem / eri_phys 均為 4-index MO 張量：
        eri_chem[p,q,r,s] = (pq|rs)   化學家記號
        eri_phys[p,q,r,s] = <pq|rs>   物理學家記號 = (pr|qs)
    """

    name: str
    symbols: list
    geometry_angstrom: np.ndarray
    basis: str
    n_orbitals: int          # spatial MO 數
    n_electrons: int
    e_nuc: float             # 核排斥能 (Ha)
    h_mo: np.ndarray         # (n,n)   一電子積分 h_pq
    eri_chem: np.ndarray     # (n,n,n,n) 化學家記號
    eri_phys: np.ndarray = field(init=False)
    e_rhf: float = field(init=False)

    def __post_init__(self):
        self.eri_phys = chemist_to_physicist(self.eri_chem)
        self.e_rhf = rhf_total_energy(
            self.h_mo, self.eri_chem, self.n_electrons, self.e_nuc
        )


def compute_rhf_integrals(
    name: str,
    symbols: list,
    geometry_angstrom,
    basis: str = "sto-3g",
) -> MolecularIntegrals:
    """以 PennyLane 微分式 HF 計算 RHF 正則 MO 積分。

    回傳的 ERI 已轉成化學家記號 (pq|rs)，與作業附錄 B 的
    PySCF ``ao2mo`` 輸出語意一致（PySCF 不支援原生 Windows，
    故以 PennyLane 為積分引擎；兩者皆為 STO-3G/RHF，數值一致）。
    """
    import pennylane as qml  # 延遲載入，避免非積分用途時的 import 開銷

    geometry_angstrom = np.asarray(geometry_angstrom, dtype=float)
    geometry_bohr = geometry_angstrom * ANGSTROM_TO_BOHR

    mol = qml.qchem.Molecule(symbols, geometry_bohr, basis_name=basis)
    core, h_mo, eri_pl = qml.qchem.electron_integrals(mol)()

    h_mo = np.array(h_mo, dtype=float)
    eri_pl = np.array(eri_pl, dtype=float)

    return MolecularIntegrals(
        name=name,
        symbols=list(symbols),
        geometry_angstrom=geometry_angstrom,
        basis=basis,
        n_orbitals=h_mo.shape[0],
        n_electrons=mol.n_electrons,
        e_nuc=float(np.atleast_1d(core)[0]),
        h_mo=h_mo,
        eri_chem=pennylane_to_chemist(eri_pl),
    )


def get_h2_integrals(r_angstrom: float = REFS["H2"]["R"]) -> MolecularIntegrals:
    """H2 / STO-3G（Problem 1-4）。2 個 spatial MO：ψ1=σg、ψ2=σu。"""
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, r_angstrom]]
    return compute_rhf_integrals("H2", ["H", "H"], geometry)


def get_lih_integrals(r_angstrom: float = REFS["LiH"]["R"]) -> MolecularIntegrals:
    """LiH / STO-3G（Problem 5）。

    STO-3G 下 LiH 共 6 個基底函數（Li 1s,2s,2p×3 + H 1s），
    全空間即作業要求的 (4e, 6o) active space，無須再做軌道篩選。
    """
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, r_angstrom]]
    return compute_rhf_integrals("LiH", ["Li", "H"], geometry)


def rhf_total_energy(h_mo, eri_chem, n_electrons, e_nuc) -> float:
    """閉殼層 RHF 能量：E = 2 Σi h_ii + Σij (2 J_ij - K_ij) + E_nuc。

    J_ij=(ii|jj)、K_ij=(ij|ji)，i,j 跑過佔據 spatial MO（前 n_elec/2 個）。
    用作積分正確性的自我檢查（應重現 benchmark E_RHF）。
    """
    n_occ = n_electrons // 2
    e = e_nuc
    for i in range(n_occ):
        e += 2.0 * h_mo[i, i]
        for j in range(n_occ):
            e += 2.0 * eri_chem[i, i, j, j] - eri_chem[i, j, j, i]
    return float(e)


# ---------------------------------------------------------------------------
# 2. ERI 記號轉換
# ---------------------------------------------------------------------------

def pennylane_to_chemist(eri_pl: np.ndarray) -> np.ndarray:
    """PennyLane 張量 -> 化學家記號。

    eri_pl[p,q,r,s] = (ps|qr)_chem，故 (ab|cd) = eri_pl[a,c,d,b]，
    即輸出軸 (a,b,c,d) 取自輸入軸 (0,3,1,2)。
    """
    return np.ascontiguousarray(np.transpose(eri_pl, (0, 3, 1, 2)))


def chemist_to_physicist(eri_chem: np.ndarray) -> np.ndarray:
    """化學家 -> 物理學家：<pq|rs> = (pr|qs)（實軌道，附錄 A）。"""
    return np.ascontiguousarray(np.transpose(eri_chem, (0, 2, 1, 3)))


def check_eri_permutation_symmetry(eri_phys: np.ndarray, atol: float = 1e-10) -> float:
    """檢查實軌道物理學家記號 ERI 的 8 重置換對稱（作業 Eq. 16）。

    <pq|rs> = <qp|sr> = <rs|pq> = <sr|qp> = <rq|ps> = <ps|rq> = <qr|sp> = <sp|qr>
    回傳最大偏差；超過 atol 則丟出 AssertionError。
    """
    perms = [
        (1, 0, 3, 2),  # <qp|sr>
        (2, 3, 0, 1),  # <rs|pq>
        (3, 2, 1, 0),  # <sr|qp>
        (2, 1, 0, 3),  # <rq|ps>
        (0, 3, 2, 1),  # <ps|rq>
        (1, 2, 3, 0),  # <qr|sp>
        (3, 0, 1, 2),  # <sp|qr>
    ]
    max_dev = 0.0
    for perm in perms:
        dev = float(np.max(np.abs(eri_phys - np.transpose(eri_phys, perm))))
        max_dev = max(max_dev, dev)
    assert max_dev < atol, f"ERI 置換對稱破壞：max deviation = {max_dev:.3e}"
    return max_dev


# ---------------------------------------------------------------------------
# 3. spatial -> spin-orbital 展開
# ---------------------------------------------------------------------------

def spin_orbital_index(p: int, sigma: int, n_spatial: int, ordering: str) -> int:
    """spatial MO p、自旋 sigma (0=alpha, 1=beta) 對應的 spin-orbital 模式編號。"""
    if ordering == "interleaved":
        return 2 * p + sigma
    if ordering == "blocked":
        return p + sigma * n_spatial
    raise ValueError(f"unknown ordering: {ordering}")


def spatial_to_spin_orbital(
    h_mo: np.ndarray,
    eri_phys: np.ndarray,
    ordering: str = "interleaved",
) -> tuple[np.ndarray, np.ndarray]:
    """spatial 積分展開到 spin-orbital（物理學家記號）。

    h_so[P,Q]      = h[p,q]       δ(σP,σQ)
    eri_so[P,Q,R,S]= <pq|rs>      δ(σP,σR) δ(σQ,σS)
    （<PQ|RS> 中電子 1 連 P,R、電子 2 連 Q,S，自旋積分給出兩個 delta）
    """
    n = h_mo.shape[0]
    n_so = 2 * n
    # 模式編號 -> (spatial, spin) 對照表
    spatial = np.empty(n_so, dtype=int)
    spin = np.empty(n_so, dtype=int)
    for p in range(n):
        for sigma in range(2):
            m = spin_orbital_index(p, sigma, n, ordering)
            spatial[m], spin[m] = p, sigma

    h_so = np.zeros((n_so, n_so))
    eri_so = np.zeros((n_so, n_so, n_so, n_so))
    for P in range(n_so):
        for Q in range(n_so):
            if spin[P] == spin[Q]:
                h_so[P, Q] = h_mo[spatial[P], spatial[Q]]
    for P in range(n_so):
        for Q in range(n_so):
            for R in range(n_so):
                if spin[P] != spin[R]:
                    continue
                for S in range(n_so):
                    if spin[Q] != spin[S]:
                        continue
                    eri_so[P, Q, R, S] = eri_phys[
                        spatial[P], spatial[Q], spatial[R], spatial[S]
                    ]
    return h_so, eri_so


# ---------------------------------------------------------------------------
# 4. 二次量子化 Hamiltonian（OpenFermion）
# ---------------------------------------------------------------------------

def build_fermionic_hamiltonian(
    h_so: np.ndarray,
    eri_so_phys: np.ndarray,
    constant: float = 0.0,
    tol: float = 1e-12,
) -> FermionOperator:
    """H = constant + Σ h_PQ a†P aQ + (1/2) Σ <PQ|RS> a†P a†Q a_S a_R。

    與作業 Problem 1 Eq.(6) 完全一致（物理學家記號；注意湮滅算符
    次序為 a_S a_R）。constant 通常放 E_nuc。
    """
    n_so = h_so.shape[0]
    op = FermionOperator((), constant) if constant else FermionOperator()
    for p in range(n_so):
        for q in range(n_so):
            c = h_so[p, q]
            if abs(c) > tol:
                op += FermionOperator(((p, 1), (q, 0)), c)
    for p in range(n_so):
        for q in range(n_so):
            for r in range(n_so):
                for s in range(n_so):
                    c = eri_so_phys[p, q, r, s]
                    if abs(c) > tol:
                        op += FermionOperator(
                            ((p, 1), (q, 1), (s, 0), (r, 0)), 0.5 * c
                        )
    return op


def total_number_operators(
    n_spatial: int, ordering: str
) -> tuple[FermionOperator, FermionOperator]:
    """N_alpha、N_beta 粒子數算符（FermionOperator）。"""
    n_a, n_b = FermionOperator(), FermionOperator()
    for p in range(n_spatial):
        ma = spin_orbital_index(p, 0, n_spatial, ordering)
        mb = spin_orbital_index(p, 1, n_spatial, ordering)
        n_a += FermionOperator(((ma, 1), (ma, 0)))
        n_b += FermionOperator(((mb, 1), (mb, 0)))
    return n_a, n_b


# ---------------------------------------------------------------------------
# 5. 費米子 -> 量子位元映射
# ---------------------------------------------------------------------------

def map_jordan_wigner(fop: FermionOperator) -> QubitOperator:
    """Jordan-Wigner（作業 Eq. 8-9）。qubit p <-> spin-orbital 模式 p。"""
    return chop(jordan_wigner(fop))


def map_bravyi_kitaev(fop: FermionOperator, n_modes: int) -> QubitOperator:
    """Bravyi-Kitaev（Seeley-Richard-Love 慣例）。"""
    return chop(bravyi_kitaev(fop, n_qubits=n_modes))


def _parity_ladder(j: int, dagger: bool, n_modes: int) -> QubitOperator:
    """Parity 編碼下的階梯算符（Seeley-Richard-Love, JCP 137, 224109）。

    qubit j 儲存累積宇稱 p_j = (n_0 + ... + n_j) mod 2，因此

        a†_j = 1/2 · X_{j+1} ... X_{n-1} · (X_j Z_{j-1} - i Y_j)
        a_j  = 1/2 · X_{j+1} ... X_{n-1} · (X_j Z_{j-1} + i Y_j)

    （j=0 時 Z_{j-1} 視為 I。）update set X_{j+1..n-1} 負責翻轉後續
    累積宇稱，Z_{j-1} 提供 JW 相位 (-1)^{Σ_{i<j} n_i}。
    自我檢查：a†_j a_j = (1 - Z_{j-1} Z_j)/2，即 parity 編碼的數算符。
    """
    update = tuple((q, "X") for q in range(j + 1, n_modes))
    xz = ((j - 1, "Z"), (j, "X")) if j > 0 else ((j, "X"),)
    sign = -1.0j if dagger else 1.0j
    return QubitOperator(tuple(sorted(update + xz)), 0.5) + QubitOperator(
        tuple(sorted(update + ((j, "Y"),))), 0.5 * sign
    )


def map_parity(fop: FermionOperator, n_modes: int) -> QubitOperator:
    """Parity 編碼（不 taper）：qubit j 儲存模式 0..j 的累積佔據宇稱。

    註：不使用 openfermion.binary_code_transform（其在 numpy>=2.4 有
    np.int64 係數的相容性 bug），改以階梯算符逐項替換實作。
    """
    ladders = {
        (j, d): _parity_ladder(j, bool(d), n_modes)
        for j in range(n_modes)
        for d in (0, 1)
    }
    out = QubitOperator()
    for term, coeff in fop.terms.items():
        q_term = QubitOperator((), coeff)
        for j, d in term:
            q_term *= ladders[(j, d)]
        out += q_term
    return chop(out)


def parity_symmetry_eigenvalues(n_alpha: int, n_beta: int) -> tuple[int, int]:
    """blocked 順序 parity 編碼下兩個 Z2 對稱 qubit 的固定本徵值。

    qubit n-1   儲存 alpha 宇稱  -> Z 本徵值 (-1)^N_alpha
    qubit 2n-1  儲存總宇稱       -> Z 本徵值 (-1)^(N_alpha+N_beta)
    中性 H2 (N_alpha=N_beta=1)：(-1, +1)。
    """
    return (-1) ** n_alpha, (-1) ** (n_alpha + n_beta)


def taper_qubits(
    qop: QubitOperator, fixed: dict[int, int], n_qubits: int
) -> QubitOperator:
    """以固定 Z 本徵值移除對稱 qubit，並將剩餘 qubit 依升冪重新編號。

    fixed: {qubit_index: ±1}。要求 Hamiltonian 在這些 qubit 上只含 I/Z
    （粒子數守恆 Hamiltonian 在 parity 編碼的對稱 qubit 上必然如此），
    否則丟出 ValueError。
    """
    remaining = [q for q in range(n_qubits) if q not in fixed]
    relabel = {q: i for i, q in enumerate(remaining)}
    out = QubitOperator()
    for term, coeff in qop.terms.items():
        new_term, c = [], coeff
        for q, pauli in term:
            if q in fixed:
                if pauli != "Z":
                    raise ValueError(
                        f"tapered qubit {q} 上出現 {pauli}（非 Z），無法替換本徵值"
                    )
                c *= fixed[q]
            else:
                new_term.append((relabel[q], pauli))
        out += QubitOperator(tuple(new_term), c)
    return chop(out)


def map_parity_tapered(
    fop_blocked: FermionOperator,
    n_modes: int,
    n_alpha: int,
    n_beta: int,
) -> tuple[QubitOperator, dict]:
    """Parity 映射 + 2-qubit Z2 tapering（作業 Problem 2/3 的第三種映射）。

    參數 fop_blocked 必須以 *blocked* 自旋軌道順序建立
    （先全部 alpha 模式、再全部 beta 模式）。

    回傳 (tapered_hamiltonian, info)；info 記錄移除的 qubit、
    固定的本徵值與順序慣例，供報告引用。
    """
    z_alpha, z_total = parity_symmetry_eigenvalues(n_alpha, n_beta)
    q_alpha, q_total = n_modes // 2 - 1, n_modes - 1
    parity_h = map_parity(fop_blocked, n_modes)
    tapered = taper_qubits(parity_h, {q_alpha: z_alpha, q_total: z_total}, n_modes)
    info = {
        "ordering": "blocked (all alpha modes first, then all beta modes)",
        "removed_qubits": [q_alpha, q_total],
        "fixed_eigenvalues": {
            f"Z{q_alpha} (alpha-number parity)": z_alpha,
            f"Z{q_total} (total-number parity)": z_total,
        },
        "n_qubits_after": n_modes - 2,
    }
    return tapered, info


def chop(qop: QubitOperator, tol: float = 1e-12) -> QubitOperator:
    """移除數值雜訊：刪掉 |係數| < tol 的項並丟棄殘餘虛部。"""
    out = QubitOperator()
    for term, coeff in qop.terms.items():
        if abs(coeff.imag) < tol:
            coeff = coeff.real
        if abs(coeff) > tol:
            out += QubitOperator(term, coeff)
    return out


# ---------------------------------------------------------------------------
# 6. 精確對角化與 sector 分析
# ---------------------------------------------------------------------------

def operator_matrix(op, n_qubits: int) -> np.ndarray:
    """FermionOperator / QubitOperator -> 稠密矩陣（小系統用）。"""
    return get_sparse_operator(op, n_qubits=n_qubits).toarray()


def exact_ground_energy(op, n_qubits: int) -> float:
    """全 Hilbert 空間基態能量。"""
    return float(np.linalg.eigvalsh(operator_matrix(op, n_qubits))[0])


def sector_dimension(n_spatial: int, n_alpha: int, n_beta: int) -> int:
    from math import comb

    return comb(n_spatial, n_alpha) * comb(n_spatial, n_beta)


def sector_eigenvalues_penalty(
    h_qop: QubitOperator,
    n_a_qop: QubitOperator,
    n_b_qop: QubitOperator,
    n_qubits: int,
    n_alpha: int,
    n_beta: int,
    n_spatial: int,
    penalty: float = 1.0e3,
) -> np.ndarray:
    """以懲罰法取出 (N_alpha, N_beta) sector 的全部本徵值。

    H' = H + λ(N̂a - n_a)² + λ(N̂b - n_b)²
    sector 內本徵值不變、sector 外被推高 ≥ λ；取 H' 最低的
    dim(sector) 個本徵值即為所求。對 JW/BK/parity 一視同仁
    （只需把粒子數算符用同一映射轉換）。
    """
    dim = sector_dimension(n_spatial, n_alpha, n_beta)
    h = operator_matrix(h_qop, n_qubits)
    na = operator_matrix(n_a_qop, n_qubits)
    nb = operator_matrix(n_b_qop, n_qubits)
    eye = np.eye(h.shape[0])
    h_pen = (
        h
        + penalty * (na - n_alpha * eye) @ (na - n_alpha * eye)
        + penalty * (nb - n_beta * eye) @ (nb - n_beta * eye)
    )
    return np.linalg.eigvalsh(h_pen)[:dim]


def fermionic_sector_eigenvalues(
    fop: FermionOperator,
    n_modes: int,
    n_alpha: int,
    n_beta: int,
    n_spatial: int,
    ordering: str = "interleaved",
    k: int | None = None,
) -> np.ndarray:
    """直接在佔據數基底投影出 (N_alpha, N_beta) sector 並對角化。

    對費米子 Hamiltonian 而言這是「定義上」的 sector 能譜，
    用來與各 qubit 映射的結果交叉驗證。JW 下佔據數基底 = 計算基底，
    模式 m 佔據 <-> 整數索引的第 m 個位元（OpenFermion 慣例：
    basis index 的最高位是模式 0）。
    """
    alpha_modes = [spin_orbital_index(p, 0, n_spatial, ordering) for p in range(n_spatial)]
    beta_modes = [spin_orbital_index(p, 1, n_spatial, ordering) for p in range(n_spatial)]

    indices = []
    for occ_a in combinations(alpha_modes, n_alpha):
        for occ_b in combinations(beta_modes, n_beta):
            idx = 0
            for m in (*occ_a, *occ_b):
                # OpenFermion get_sparse_operator: 模式 m <-> 位元 (n_modes-1-m)
                idx |= 1 << (n_modes - 1 - m)
            indices.append(idx)
    indices = sorted(indices)

    h = get_sparse_operator(fop, n_qubits=n_modes).toarray()
    sub = h[np.ix_(indices, indices)]
    vals = np.linalg.eigvalsh(sub)
    return vals if k is None else vals[:k]


# ---------------------------------------------------------------------------
# 7. OpenFermion <-> Qiskit 與 Pauli 項工具
# ---------------------------------------------------------------------------

def qubit_operator_to_sparse_pauli_op(qop: QubitOperator, n_qubits: int):
    """QubitOperator -> qiskit SparsePauliOp。

    Qiskit 的 Pauli 標籤是 little-endian：字串最右邊的字元對應 qubit 0。
    """
    from qiskit.quantum_info import SparsePauliOp

    labels, coeffs = [], []
    for term, coeff in qop.terms.items():
        chars = ["I"] * n_qubits
        for q, pauli in term:
            chars[q] = pauli
        labels.append("".join(reversed(chars)))
        coeffs.append(complex(coeff))
    return SparsePauliOp(labels, coeffs=np.array(coeffs)).simplify()


def pauli_term_table(qop: QubitOperator, tol: float = 1e-12) -> list[dict]:
    """Pauli 項列表（字串依 qubit index 升冪，如 'X0 Y1 Y2 X3'；恆等項為 'I'）。

    依 (weight, 字串) 排序，方便報告呈現與跨映射比較。
    """
    rows = []
    for term, coeff in qop.terms.items():
        coeff = coeff.real if abs(coeff.imag) < tol else coeff
        label = " ".join(f"{p}{q}" for q, p in sorted(term)) if term else "I"
        rows.append({"pauli": label, "weight": len(term), "coeff": float(np.real(coeff))})
    rows.sort(key=lambda r: (r["weight"], r["pauli"]))
    return rows


def mapping_summary(qop: QubitOperator, n_qubits: int) -> dict:
    """Problem 2(d) 比較指標：qubit 數、非恆等 Pauli 項數、最大 Pauli weight。"""
    weights = [len(term) for term in qop.terms if term]
    return {
        "n_qubits": n_qubits,
        "n_non_identity_terms": len(weights),
        "max_pauli_weight": max(weights) if weights else 0,
    }


# ---------------------------------------------------------------------------
# 8. 結果存檔
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return {"re": obj.real, "im": obj.imag}
        return super().default(obj)


def save_results(name: str, data: dict) -> Path:
    """寫入 results/<name>.json，供 build_report.py 自動插入報告。"""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    return path


def package_versions() -> dict:
    """記錄主要套件版本（報告要求）。"""
    import openfermion
    import pennylane
    import qiskit

    return {
        "pennylane": pennylane.__version__,
        "openfermion": openfermion.__version__,
        "qiskit": qiskit.__version__,
        "numpy": np.__version__,
    }


# ---------------------------------------------------------------------------
# 自我測試：python hw3_common.py
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # Windows 主控台預設 CP950，中文輸出易成亂碼，自我測試訊息採 ASCII
    np.set_printoptions(precision=8, suppress=True)
    print("=" * 70)
    print("hw3_common.py self-test (benchmark verification)")
    print("package versions:", package_versions())
    print("=" * 70)

    # ---- H2 ----
    h2 = get_h2_integrals()
    dev = check_eri_permutation_symmetry(h2.eri_phys)
    print(f"\n[H2/STO-3G, R={REFS['H2']['R']} A]")
    print(f"  E_nuc          = {h2.e_nuc:+.6f} Ha")
    print(f"  E_RHF          = {h2.e_rhf:+.6f} Ha   (ref {REFS['H2']['E_RHF']:+.5f})")
    print(f"  ERI 8-fold sym : max deviation = {dev:.2e}")
    assert abs(h2.e_rhf - REFS["H2"]["E_RHF"]) < 5e-5, "H2 RHF 與 benchmark 不符"

    n_modes = 2 * h2.n_orbitals
    # interleaved（作業慣例）：JW / BK
    h_so, eri_so = spatial_to_spin_orbital(h2.h_mo, h2.eri_phys, "interleaved")
    fop = build_fermionic_hamiltonian(h_so, eri_so, constant=h2.e_nuc)
    jw = map_jordan_wigner(fop)
    bk = map_bravyi_kitaev(fop, n_modes)
    # blocked：parity + 2-qubit tapering
    h_so_b, eri_so_b = spatial_to_spin_orbital(h2.h_mo, h2.eri_phys, "blocked")
    fop_b = build_fermionic_hamiltonian(h_so_b, eri_so_b, constant=h2.e_nuc)
    parity2, taper_info = map_parity_tapered(fop_b, n_modes, n_alpha=1, n_beta=1)

    e_fci_sector = fermionic_sector_eigenvalues(fop, n_modes, 1, 1, h2.n_orbitals)[0]
    e_jw = exact_ground_energy(jw, n_modes)        # 全空間基態即落在中性 sector
    e_bk = exact_ground_energy(bk, n_modes)
    e_parity2 = exact_ground_energy(parity2, 2)    # tapered 空間 = 中性 sector
    print(f"  E_FCI fermionic sector = {e_fci_sector:+.6f} Ha (ref {REFS['H2']['E_FCI']:+.5f})")
    print(f"  E_FCI JW   ED          = {e_jw:+.6f} Ha")
    print(f"  E_FCI BK   ED          = {e_bk:+.6f} Ha")
    print(f"  E_FCI parity-tapered ED= {e_parity2:+.6f} Ha")
    print(f"  tapering: removed qubits {taper_info['removed_qubits']}, "
          f"eigenvalues {taper_info['fixed_eigenvalues']}")
    for e in (e_jw, e_bk, e_parity2):
        assert abs(e - e_fci_sector) < 1e-9, "各映射基態能不一致"
    assert abs(e_fci_sector - REFS["H2"]["E_FCI"]) < 5e-5, "H2 FCI 與 benchmark 不符"

    for label, op, nq in (("JW", jw, 4), ("BK", bk, 4), ("parity-tapered", parity2, 2)):
        print(f"  {label:15s}: {mapping_summary(op, nq)}")

    # ---- LiH ----
    lih = get_lih_integrals()
    dev = check_eri_permutation_symmetry(lih.eri_phys)
    print(f"\n[LiH/STO-3G, R={REFS['LiH']['R']} A]  (4e, 6o) = 12 spin orbitals")
    print(f"  E_nuc = {lih.e_nuc:+.6f} Ha (ref {REFS['LiH']['E_nuc']:+.5f})")
    print(f"  E_RHF = {lih.e_rhf:+.6f} Ha (ref {REFS['LiH']['E_HF']:+.5f})")
    print(f"  ERI 8-fold sym : max deviation = {dev:.2e}")
    assert abs(lih.e_nuc - REFS["LiH"]["E_nuc"]) < 5e-5
    assert abs(lih.e_rhf - REFS["LiH"]["E_HF"]) < 5e-5

    h_so, eri_so = spatial_to_spin_orbital(lih.h_mo, lih.eri_phys, "interleaved")
    fop = build_fermionic_hamiltonian(h_so, eri_so, constant=lih.e_nuc)
    e_fci = fermionic_sector_eigenvalues(fop, 12, 2, 2, 6, k=1)[0]
    print(f"  E_FCI (N_a=N_b=2 sector ED) = {e_fci:+.6f} Ha (ref {REFS['LiH']['E_FCI']:+.5f})")
    assert abs(e_fci - REFS["LiH"]["E_FCI"]) < 5e-5, "LiH FCI 與 benchmark 不符"

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _self_test()
