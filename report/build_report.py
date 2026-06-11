"""
build_report.py -- QCAA HW3 報告產生器
=======================================

流程：
  1. 依序執行 problem1.py ~ problem5.py（--skip-run 可跳過，直接用現有
     results/*.json）
  2. 由 results/*.json 產生 report/generated/ 下的 LaTeX 巨集與表格
  3. 以 XeLaTeX（MiKTeX，--enable-installer 自動補套件）編譯 main.tex 兩次
  4. 複製輸出為 HW3_13705049.pdf（HW3 根目錄）

用法：
  python report/build_report.py            # 全流程
  python report/build_report.py --skip-run # 只重建表格與 PDF
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
BASE_DIR = REPORT_DIR.parent
RESULTS_DIR = BASE_DIR / "results"
GENERATED_DIR = REPORT_DIR / "generated"
OUTPUT_PDF = BASE_DIR / "HW3_13705049.pdf"
PROBLEMS = [f"problem{i}.py" for i in range(1, 6)]


# ---------------------------------------------------------------------------
# step 1: 執行五題
# ---------------------------------------------------------------------------

def run_problems() -> None:
    for script in PROBLEMS:
        print(f"[run] {script} ...")
        proc = subprocess.run([sys.executable, str(BASE_DIR / script)],
                              cwd=BASE_DIR, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-3000:])
            print(proc.stderr[-3000:])
            raise RuntimeError(f"{script} failed (exit {proc.returncode})")
        tail = [ln for ln in proc.stdout.splitlines() if "checks passed" in ln]
        print(f"      {tail[-1] if tail else 'done'}")


def load(name: str) -> dict:
    with open(RESULTS_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# step 2: LaTeX 產生工具
# ---------------------------------------------------------------------------

def fnum(x: float, nd: int = 8, signed: bool = True) -> str:
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def fsci(x: float) -> str:
    """1.75e-07 -> \\ensuremath{1.8\\times10^{-7}}（文字/數學模式皆可用）。"""
    if x == 0:
        return r"\ensuremath{0}"
    m, e = f"{x:.1e}".split("e")
    return rf"\ensuremath{{{m}\times10^{{{int(e)}}}}}"


def write(name: str, content: str) -> None:
    GENERATED_DIR.mkdir(exist_ok=True)
    (GENERATED_DIR / name).write_text(content, encoding="utf-8")
    print(f"[gen] generated/{name}")


def table(body: str, caption: str, colspec: str, header: str) -> str:
    return (
        "\\begin{table}[H]\n\\centering\\small\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
        f"{header} \\\\\n\\midrule\n{body}\\bottomrule\n\\end{{tabular}}\n"
        f"\\caption{{{caption}}}\n\\end{{table}}\n"
    )


def pauli_table(rows: list[dict], caption: str) -> str:
    body = "".join(
        f"\\texttt{{{r['pauli'].replace(' ', '\\,')}}} & {r['weight']} & "
        f"{fnum(r['coeff'])} \\\\\n"
        for r in rows)
    return table(body, caption, "lrr",
                 "Pauli string & weight & coefficient (Ha)")


# ---------------------------------------------------------------------------
# step 2: 巨集與表格
# ---------------------------------------------------------------------------

def generate_latex() -> None:
    p1, p2 = load("problem1"), load("problem2")
    p3, p4, p5 = load("problem3"), load("problem4"), load("problem5")

    # ---------------- macros ----------------
    v = p1["versions"]
    blk = p1["fci_block_electronic"]
    scan = p1["stretch_scan"]
    noise = p4["c_backends"]["noise_model_parameters"]
    e_ref_parity = p4["ab_vqe"]["UCCSD+parity"]["energy"]
    noisy_bias = p4["c_shot_results"]["noisy_10000"]["mean"] - e_ref_parity
    bm5 = p5["benchmarks"]

    macros = {
        "PkgVersions": (f"PennyLane {v['pennylane']}, OpenFermion "
                        f"{v['openfermion']}, Qiskit {v['qiskit']}, "
                        f"NumPy {v['numpy']}"),
        "PLVersion": v["pennylane"],
        # benchmarks
        "Enuc": fnum(p1["e_nuc"]),
        "ErhfHtwo": fnum(p1["e_rhf"], 5),
        "EfciHtwo": fnum(p1["E_FCI_total"], 5),
        "EfciHtwoLong": fnum(p1["E_FCI_total"]),
        "ErhfLih": fnum(bm5["E_RHF"], 5),
        "EfciLih": fnum(bm5["E_FCI_sector_ED"], 5),
        "EfciLihLong": fnum(bm5["E_FCI_sector_ED"]),
        # problem 1
        "PoneHaa": fnum(blk[0][0]), "PoneHab": fnum(blk[0][1]),
        "PoneHbb": fnum(blk[1][1]),
        "PoneHermDev": fsci(p1["hermiticity_dev"]),
        "PoneSqDev": fsci(p1["fci_block_secondquant_dev"]),
        "PoneCoupling": fsci(max(p1["open_shell_coupling"], 1e-16)),
        "PoneFciErr": fsci(p1["abs_error_vs_ref"]),
        "PoneCa": fnum(p1["ground_state_coefficients"]["c1"], 4),
        "PoneCb": fnum(p1["ground_state_coefficients"]["c2"], 4),
        "PoneCtwoSq": fnum(p1["ground_state_coefficients"]["c2_sq"], 4, False),
        "PoneEcorr": fnum(p1["E_corr"], 4),
        "PoneCorrLast": fnum(abs(scan["E_corr"][-1]), 3, False),
        # problem 2
        "PtwoAcrDev": fsci(max(p2["a_anticommutation_max_dev"], 1e-16)),
        "PtwoNumDev": fsci(max(p2["a_number_operator_max_dev"], 1e-16)),
        "PtwoBeleven": fnum(p2["b_jw_eq10_coefficients"]["b11 (XYYX block)"]),
        "PtwoSectorDev": fsci(p2["c_max_deviation"]),
        "PtwoFciErr": fsci(p2["c_abs_error_vs_ref"]),
        # problem 3
        "PthreePermDev": fsci(p3["a_integrals"]["perm_symmetry_dev"]),
        "PthreeBrillouin": fsci(max(p3["c_uccsd"]["brillouin_gradient"], 1e-16)),
        "PthreeThetaOpt": fnum(p3["c_uccsd"]["theta_opt"]["JW"], 6),
        "PthreeCxJw": str(p3["c_uccsd"]["gate_counts"]["JW"]["cx"]),
        "PthreeCxPar": str(p3["c_uccsd"]["gate_counts"]["parity"]["cx"]),
        # problem 4
        "PfourTone": f"{noise['median_T1_us']:.0f}",
        "PfourTtwo": f"{noise['median_T2_us']:.0f}",
        "PfourSxErr": fsci(noise["median_sx_error"]),
        "PfourEcrErr": fsci(noise["median_ecr_error"]),
        "PfourRoErr": fsci(noise["median_readout_error"]),
        "PfourNoisyBias": fnum(noisy_bias, 3, False),
        # problem 5
        "PfiveSectorProb": fnum(p5["sector_probability"], 12, False),
        "PfiveEnuc": fnum(bm5["E_nuc"], 5),
        "PfiveCcsdIter": str(p5["ccsd_n_iter"]),
        "PfiveEccsd": fnum(bm5["E_CCSD"]),
        "PfiveCcsdErr": fsci(abs(bm5["E_CCSD"] - bm5["refs"]["E_CCSD"])),
        "PfiveEucc": fnum(bm5["E_UCCSD_state"]),
        "PfiveUccErr": fsci(abs(bm5["E_UCCSD_state"] - bm5["E_FCI_sector_ED"])),
        "PfiveFinalErr": fsci(p5["qsci_results"][-1]["abs_error_vs_FCI"]),
    }
    write("macros.tex", "".join(
        f"\\newcommand{{\\{k}}}{{{val}}}\n" for k, val in macros.items()))

    # ---------------- problem 1: stretch scan ----------------
    body = "".join(
        f"{scan['R'][i]:.4f} & {fnum(scan['E_RHF'][i], 6)} & "
        f"{fnum(scan['E_FCI'][i], 6)} & {fnum(scan['E_corr'][i], 6)} & "
        f"{scan['c2_sq'][i]:.4f} \\\\\n"
        for i in range(len(scan["R"])))
    write("tab_p1_scan.tex", table(
        body, "Bond-stretching scan of H$_2$/STO-3G (\\texttt{problem1.py}).",
        "rrrrr",
        "$R$ (\\AA) & $E_\\mathrm{RHF}$ (Ha) & $E_\\mathrm{FCI}$ (Ha) & "
        "$E_\\mathrm{corr}$ (Ha) & $|c_2|^2$"))

    # ---------------- problem 2: Pauli tables ----------------
    write("tab_p2_jw.tex", pauli_table(
        p2["b_pauli_terms"]["JW"],
        "Jordan--Wigner Hamiltonian (4 qubits, interleaved ordering; "
        "$E_\\mathrm{nuc}$ in the identity term)."))
    write("tab_p2_bk.tex", pauli_table(
        p2["b_pauli_terms"]["BK"],
        "Bravyi--Kitaev Hamiltonian (4 qubits, interleaved ordering)."))
    write("tab_p2_parity.tex", pauli_table(
        p2["b_pauli_terms"]["parity_tapered"],
        "Two-qubit parity-tapered Hamiltonian (blocked ordering; removed "
        "qubits 1 and 3 with $Z_1=-1$, $Z_3=+1$)."))

    sect = p2["c_sector_eigenvalues"]
    body = "".join(
        f"{i} & {fnum(sect['fermionic'][i])} & {fnum(sect['JW'][i])} & "
        f"{fnum(sect['BK'][i])} & {fnum(sect['parity_tapered'][i])} \\\\\n"
        for i in range(4))
    write("tab_p2_sector.tex", table(
        body, "The four eigenvalues (Ha) of the neutral "
        "$N_\\alpha{=}N_\\beta{=}1$ sector across representations.",
        "crrrr", "\\# & fermionic & JW & BK & parity-tapered"))

    body = "".join(
        f"{name} & {s['n_qubits']} & {s['n_non_identity_terms']} & "
        f"{s['max_pauli_weight']} \\\\\n"
        for name, s in p2["d_summary"].items())
    write("tab_p2_summary.tex", table(
        body, "Mapping comparison for minimal-basis H$_2$.",
        "lccc", "mapping & qubits & non-identity terms & max Pauli weight"))

    # ---------------- problem 3: integrals ----------------
    h = p3["a_integrals"]["h_mo"]
    eri = p3["a_integrals"]["unique_eri_physicist"]
    chem = {"<11|11>": "(11|11)", "<11|12>": "(11|12)", "<11|22>": "(12|12)",
            "<12|12>": "(11|22)", "<12|22>": "(12|22)", "<22|22>": "(22|22)"}
    body = (f"$E_\\mathrm{{nuc}}$ & {fnum(p3['a_integrals']['e_nuc'])} & \\\\\n"
            f"$h_{{11}}$ & {fnum(h[0][0])} & \\\\\n"
            f"$h_{{12}}=h_{{21}}$ & {fnum(h[0][1])} & (symmetry) \\\\\n"
            f"$h_{{22}}$ & {fnum(h[1][1])} & \\\\\n")
    for k, val in eri.items():
        kk = k.replace("<", "\\langle ").replace("|", "\\vert ").replace(">", "\\rangle")
        body += f"${kk}$ & {fnum(val)} & $= {chem[k]}$ \\\\\n"
    write("tab_p3_integrals.tex", table(
        body, "RHF/STO-3G integrals of H$_2$ at $R=0.7414$\\,\\AA\\ "
        "(physicists' notation; chemists' equivalent in the last column).",
        "lrl", "quantity & value (Ha) & chemists'"))

    # ---------------- problem 4 ----------------
    body = "".join(
        f"{name} & {fnum(r['energy'])} & {fnum(r['E_exact'])} & "
        f"{fsci(max(r['abs_error'], 1e-16))} & {r['n_params']} & {r['nfev']} \\\\\n"
        for name, r in p4["ab_vqe"].items())
    write("tab_p4_vqe.tex", table(
        body, "VQE results on the noiseless statevector simulator "
        "(shared settings in the text).",
        "lrrrcc",
        "case & $E_\\mathrm{VQE}$ (Ha) & $E_\\mathrm{exact}$ (Ha) & "
        "abs.\\ error & \\#params & \\#evals"))

    body = "".join(
        f"{name} & \\texttt{{{', '.join(f'{x:+.4f}' for x in r['params'])}}} \\\\\n"
        for name, r in p4["ab_vqe"].items())
    write("tab_p4_params.tex", table(
        body, "Optimized variational parameters.",
        "l>{\\raggedright\\arraybackslash}p{10.5cm}", "case & parameters"))

    body = ""
    for key, label in (("ideal_101", "ideal"), ("ideal_10000", "ideal"),
                       ("noisy_101", "noisy"), ("noisy_10000", "noisy")):
        r = p4["c_shot_results"][key]
        body += (f"{label} & {r['total_shots']} & {fnum(r['mean'], 6)} & "
                 f"{r['se']:.6f} & {fnum(r['mean'] - e_ref_parity, 6)} \\\\\n")
    write("tab_p4_shots.tex", table(
        body, "Finite-shot energy estimates of the frozen UCCSD+parity "
        "circuit ($E_\\mathrm{ref}=" + fnum(e_ref_parity) + "$\\,Ha).",
        "lrrrr",
        "backend & shots & sample mean (Ha) & std.\\ error (Ha) & "
        "mean $-$ $E_\\mathrm{ref}$"))

    # ---------------- problem 5 ----------------
    body = "".join(
        f"$10^{len(str(r['shots'])) - 1}$ & {r['d_unique_kept']} & "
        f"{r['n_discarded_shots']} & {fnum(r['E_SQD'])} & "
        f"{fsci(r['abs_error_vs_FCI'])} \\\\\n"
        for r in p5["qsci_results"])
    write("tab_p5_qsci.tex", table(
        body, "QSCI on LiH (4e,\\,6o): unique kept determinants $d$, total "
        "energy and error vs.\\ FCI for each shot budget "
        "(sector dimension 225).",
        "rrrrr",
        "shots & $d$ & discarded & $E_\\mathrm{SQD}$ (Ha) & "
        "$|E_\\mathrm{SQD}-E_\\mathrm{FCI}|$"))


# ---------------------------------------------------------------------------
# step 3-4: 編譯
# ---------------------------------------------------------------------------

def compile_pdf() -> None:
    cmd = ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
           "--enable-installer", "main.tex"]
    for run in (1, 2):
        print(f"[tex] XeLaTeX pass {run} ...")
        proc = subprocess.run(cmd, cwd=REPORT_DIR, capture_output=True,
                              text=True, errors="replace")
        if proc.returncode != 0:
            tail = proc.stdout[-4000:]
            print(tail)
            raise RuntimeError("XeLaTeX failed")
    pages = [ln for ln in proc.stdout.splitlines() if "Output written" in ln]
    print(f"[tex] {pages[-1] if pages else 'compiled'}")
    shutil.copyfile(REPORT_DIR / "main.pdf", OUTPUT_PDF)
    print(f"[out] {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    if "--skip-run" not in sys.argv:
        run_problems()
    else:
        print("[run] skipped (--skip-run)")
    generate_latex()
    compile_pdf()


if __name__ == "__main__":
    main()
