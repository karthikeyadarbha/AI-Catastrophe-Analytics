#!/usr/bin/env python3
"""Gold, Experiment 2 (Data-First), stage 5: re-run the 54 treatise-defined bins and check them against Silver.

Run from the repository root, after build_silver.py has made silver.duckdb:

    python3 gold_experiment2/pipeline/run_stage5.py --inputs analysis/workdir [--null]

What it does, in plain words
  1. Reads the 54 bins (Silver rules S5.*) and the published results from Silver.
  2. Runs analysis/experiment2_data_first/stage5_treatise_bin_scan.py, never edited, in a temporary
     copy that only redirects where the results are written.
  3. Checks:
       A. Silver's published p-values, hold-out counts and expected counts equal the committed CSV.
       B. The re-run reproduces the committed CSV exactly: counts identical, z-scores within 1e-9,
          for all four periods (1990-2007, 2008-2026, Older set 1, Older set 2).
       C. The pass rule gives the same answer: a bin is a finding only if p < 0.05/54 in both
          1990-2007 and 2008-2026, in the same direction.
       D. With --null (about a minute): the cross-era agreement test (3,000 redraws, fixed seed)
          reproduces the committed agreement CSV.
Unlike Experiment 1 this stage starts from already-declustered input files, so nothing in it depends on
the order of tied events and an exact match is expected. Writes only to the output folder.
Needs: python3, duckdb, pandas, numpy, scipy.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path.cwd()
EXP2 = REPO / "analysis/experiment2_data_first"
SCRIPT = EXP2 / "stage5_treatise_bin_scan.py"
SHARED = REPO / "analysis/shared"
COMMITTED = REPO / "docs/audit/data/data_first_treatise_bin_scan.csv"
COMMITTED_AGREE = REPO / "docs/audit/data/data_first_treatise_scan_agreement.csv"
PATH_LINE = "_d=_o.path.dirname(_o.path.abspath(__file__))"
OUT_BIN = '"results/data_first_treatise_bin_scan.csv"'
OUT_AGREE = '"results/data_first_treatise_scan_agreement.csv"'
NEEDED = ["quakes_m55_input.csv", "m55_features_sid.csv", "bg_sid.csv",
          "histA_declustered.csv", "histB_declustered.csv"]
N_BINS = 54
TOL = 1e-9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_silver(db: Path):
    import duckdb
    con = duckdb.connect(str(db), read_only=True)
    rules = con.execute("SELECT rule_id, preregistration, review_status FROM rules "
                        "WHERE rule_kind='testable_claim' AND source_file LIKE '%experiment2_stage5_rules.json' "
                        "ORDER BY rule_id").fetchall()
    rep = {r[0]: r[1:] for r in con.execute(
        "SELECT rule_id, discovery_p, holdout_p, holdout_observed, holdout_expected FROM reported_results").fetchall()}
    con.close()
    if len(rules) != N_BINS:
        sys.exit(f"Silver holds {len(rules)} stage-5 bins, expected {N_BINS}. Rebuild it with build_silver.py.")
    key_of = {rid: tuple(json.loads(pre)["test_id"].split(" || ")) for rid, pre, _ in rules}
    return key_of, rep, sorted({rv for _, _, rv in rules})


def run_script(inputs: Path, work: Path, null: bool):
    src = SCRIPT.read_text(encoding="utf-8")
    for needle in (PATH_LINE, OUT_BIN, OUT_AGREE):
        if src.count(needle) != 1:
            sys.exit(f"stage5_treatise_bin_scan.py no longer contains exactly one: {needle}\n"
                     "It has changed since this runner was written; stopping.")
    out = work / "out"
    out.mkdir()
    src = (src.replace(PATH_LINE, f"_d={str(EXP2)!r}")
              .replace(OUT_BIN, repr(str(out / "bins.csv"))).replace(OUT_AGREE, repr(str(out / "agree.csv"))))
    script = work / "stage5_copy.py"
    script.write_text(src, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(SHARED) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, str(script)] + (["null"] if null else []),
                       cwd=inputs, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit("stage 5 script failed:\n" + r.stderr[-1200:])
    return out, r.stdout


def make_features(inputs: Path):
    """m55_features_sid.csv is made from the other inputs by the repository's own script."""
    print("making m55_features_sid.csv with analysis/shared/big55.py ...", flush=True)
    env = dict(os.environ, PYTHONPATH=str(SHARED) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, str(SHARED / "big55.py")], cwd=inputs, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit("big55.py failed:\n" + r.stderr[-800:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="silver_text_rules/silver.duckdb")
    ap.add_argument("--inputs", default="analysis/workdir")
    ap.add_argument("--out", default="gold_experiment2/results")
    ap.add_argument("--null", action="store_true", help="also reproduce the cross-era agreement test (about a minute)")
    args = ap.parse_args()

    inputs = (REPO / args.inputs).resolve()
    base_needed = [f for f in NEEDED if f != "m55_features_sid.csv"]
    missing = [f for f in base_needed if not (inputs / f).is_file()]
    if missing:
        sys.exit(f"Missing in {inputs}: {', '.join(missing)}")
    if not Path(args.db).is_file():
        sys.exit(f"{args.db} not found. Run silver_text_rules/pipeline/build_silver.py first.")
    if not (inputs / "m55_features_sid.csv").is_file():
        make_features(inputs)

    key_of, reported, reviews = load_silver(Path(args.db))
    committed = {(r["feature"], r["bin"]): r for r in read_csv(COMMITTED)}
    problems = []
    if set(key_of.values()) != set(committed):
        problems.append("Silver's 54 bins and the committed CSV do not cover the same bins")

    with tempfile.TemporaryDirectory() as tmp:
        out, _ = run_script(inputs, Path(tmp), args.null)
        rerun = {(r["feature"], r["bin"]): r for r in read_csv(out / "bins.csv")}
        agree = read_csv(out / "agree.csv") if args.null else None

    periods = ("disc", "hold", "older1", "older2")
    rows = []
    for rid in sorted(key_of):
        k = key_of[rid]
        c, n = committed[k], rerun.get(k)
        sp = reported.get(rid)
        a_ok = (sp is not None and abs(round(float(c["p_disc"]), 3) - sp[0]) < 1e-9
                and abs(round(float(c["p_hold"]), 3) - sp[1]) < 1e-9
                and sp[2] == c["obs_hold"] and sp[3] == f'{float(c["exp_hold"]):.1f}')
        b_ok, worst = n is not None, 0.0
        if n is not None:
            for p in periods:
                if n[f"obs_{p}"] != c[f"obs_{p}"] or n[f"n_{p}"] != c[f"n_{p}"]:
                    b_ok = False
                d = abs(float(n[f"z_{p}"]) - float(c[f"z_{p}"]))
                worst = max(worst, d)
                if d > TOL:
                    b_ok = False
        for flag, text in ((a_ok, "Silver's published values differ from the committed CSV"),
                           (b_ok, "re-run does not reproduce the committed CSV")):
            if not flag:
                problems.append(f"{rid}: {text}")
        rows.append({"rule_id": rid, "feature": k[0], "bin": k[1], "silver_matches_csv": a_ok,
                     "rerun_reproduces_csv": b_ok, "max_abs_z_difference": worst,
                     "z_disc": c["z_disc"], "z_hold": c["z_hold"]})

    def findings(table):
        bar = 0.05 / N_BINS
        return sorted(k for k, r in table.items()
                      if float(r["p_disc"]) < bar and float(r["p_hold"]) < bar
                      and (float(r["z_disc"]) > 0) == (float(r["z_hold"]) > 0))
    f_committed, f_rerun = findings(committed), findings(rerun)
    c_ok = f_committed == f_rerun
    if not c_ok:
        problems.append("the pass rule gives a different set of findings after the re-run")
    nominal = sum(float(r["p_disc"]) < 0.05 for r in rerun.values())

    d_ok = None
    if agree is not None:
        ref = {r["era_compared_with_1990_2007"]: r for r in read_csv(COMMITTED_AGREE)}
        d_ok = (len(agree) == len(ref) and all(
            abs(float(a["observed_r"]) - float(ref[a["era_compared_with_1990_2007"]]["observed_r"])) < 1e-3
            and abs(float(a["p_two_sided"]) - float(ref[a["era_compared_with_1990_2007"]]["p_two_sided"])) < 1e-9
            for a in agree))
        if not d_ok:
            problems.append("the cross-era agreement test does not reproduce the committed file")

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "stage5_gold_check.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "bins_checked": len(rows), "bonferroni_threshold": 0.05 / N_BINS,
        "findings_committed": [list(k) for k in f_committed], "findings_rerun": [list(k) for k in f_rerun],
        "nominal_discovery_hits": nominal, "agreement_test_reproduced": d_ok,
        "silver_rule_review_status": reviews,
        "inputs_sha256": {f: sha256(inputs / f) for f in NEEDED},
        "script_sha256": sha256(SCRIPT),
        "hard_checks_passed": not problems, "problems": problems,
    }
    (out_dir / "stage5_gold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{len(rows)} bins from Silver")
    print(f"  A  Silver matches the committed CSV:      {sum(r['silver_matches_csv'] for r in rows)} of {len(rows)}")
    print(f"  B  re-run reproduces the committed CSV:   {sum(r['rerun_reproduces_csv'] for r in rows)} of {len(rows)}"
          f" (largest z difference {max(r['max_abs_z_difference'] for r in rows):.1e})")
    print(f"  C  findings under the pass rule:          {len(f_rerun)} (same as committed: {c_ok}); "
          f"{nominal} bins nominally significant in 1990-2007 (about 2.7 expected by chance)")
    if d_ok is not None:
        print(f"  D  cross-era agreement test reproduced:   {d_ok}")
    if set(reviews) != {"approved"}:
        print(f"  note: Silver review status of these bins is {reviews}; they are not yet human-reviewed.")
    for p in problems:
        print("  PROBLEM:", p)
    print(f"written: {out_dir}/stage5_gold_check.csv and stage5_gold_summary.json")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
