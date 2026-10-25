#!/usr/bin/env python3
"""Gold, Experiment 1 (Rule-First): re-run the 23 pre-registered tests and check them against Silver.

Run from the repository root, after build_silver.py has made silver.duckdb:

    python3 gold_experiment1/pipeline/run_experiment1.py --inputs analysis/workdir --seeds 200

What it does, in plain words
  1. Reads the 23 testable rules and the published results from Silver (the DuckDB file).
  2. Runs the pre-registered script analysis/experiment1_rule_first/02_rules_large_sets/big_test.py
     once exactly as committed, then --seeds more times. The script itself is never edited.
     Each extra run only changes one thing: the order in which earthquakes of EQUAL magnitude
     are visited when clusters are removed (the script leaves that order to chance).
  3. Checks, for each of the 46 results (23 rules x discovery/holdout):
       HARD checks (the run fails if any breaks)
         A. Silver's published p-values match the committed results CSV (no transcription slip).
         B. Every run agrees with the published verdict at the Bonferroni threshold 0.05/23.
         C. Every run finds the same declustered event counts as the committed CSV (n per set).
       SOFT check (reported, not failing)
         D. The published p-value lies inside the range the runs produced.
Nothing is written into the repository except the output folder (default gold_experiment1/results/).
Needs: python3, duckdb, pandas, numpy, scipy. The input folder must hold big.csv, quakes_m65_sid.csv,
quakes_m65_trop.csv, bg_sid.csv, bg_trop.csv (see gold_experiment1/README.md).
"""
import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path.cwd()
SCRIPT = REPO / "analysis/experiment1_rule_first/02_rules_large_sets/big_test.py"
COMMITTED = REPO / "docs/audit/data/prereg_large_sample_results.csv"
DECLUSTER_LINE = 'q=q.sort_values("mag",ascending=False).reset_index(drop=True)'
PATH_LINE = "_d=_o.path.dirname(_o.path.abspath(__file__))"
OUT_LINES = ('"results/prereg_large_sample_results.csv"', '"results/declustered_events_M65.csv"')
NEEDED = ["big.csv", "quakes_m65_sid.csv", "quakes_m65_trop.csv", "bg_sid.csv", "bg_trop.csv"]
BONF = 0.05 / 23


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_silver(db: Path):
    import duckdb
    con = duckdb.connect(str(db), read_only=True)
    rules = con.execute("SELECT rule_id, preregistration, review_status FROM rules "
                        "WHERE rule_kind='testable_claim' AND source_file LIKE '%experiment1_rules.json' ORDER BY rule_id").fetchall()
    rep = {r[0]: r[1:] for r in con.execute(
        "SELECT rule_id, discovery_p, holdout_p FROM reported_results").fetchall()}
    con.close()
    if not rules:
        sys.exit("Silver holds no Experiment 1 rules. Rebuild it with build_silver.py.")
    test_of = {rid: json.loads(pre)["test_id"] for rid, pre, _ in rules}
    reviews = sorted({rv for _, _, rv in rules})
    return test_of, rep, reviews


def variant_source(seed):
    """Source of big_test.py with only the tie order changed (seed None = exactly as committed)."""
    src = SCRIPT.read_text(encoding="utf-8")
    for needle in (PATH_LINE, DECLUSTER_LINE, *OUT_LINES):
        if src.count(needle) != 1:
            sys.exit(f"big_test.py no longer contains exactly one: {needle}\n"
                     "It has changed since this runner was written; stopping.")
    src = src.replace(PATH_LINE, f"_d={str(SCRIPT.parent)!r}")
    if seed is not None:
        src = src.replace(DECLUSTER_LINE,
                          f'q=q.sample(frac=1,random_state={seed}).sort_values("mag",ascending=False,'
                          'kind="mergesort").reset_index(drop=True)')
    return src


def one_run(seed, inputs: Path, work: Path, cache: Path):
    """One run of the test script in its own process. Results are cached per seed, so a repeat
    run (or a larger --seeds) only computes what has not been computed before."""
    tag = "committed" if seed is None else f"seed{seed:04d}"
    cached = cache / f"{tag}.json"
    if cached.is_file():
        return {tuple(k.split("|")): tuple(v) for k, v in json.loads(cached.read_text()).items()}, True
    out = work / tag
    out.mkdir(parents=True)
    src = variant_source(seed)
    src = src.replace(OUT_LINES[0], repr(str(out / "results.csv"))).replace(OUT_LINES[1], repr(str(out / "events.csv")))
    script = work / f"{tag}.py"
    script.write_text(src, encoding="utf-8")
    # one thread per process: with many processes at once, extra library threads only slow things down
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, str(script)], cwd=inputs, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"run {tag} failed:\n{r.stderr[-800:]}")
    with (out / "results.csv").open(newline="", encoding="utf-8") as fh:
        res = {(d["test"], d["set"]): (float(d["p"]), int(d["n"])) for d in csv.DictReader(fh)}
    tmp = cached.with_suffix(".tmp")
    tmp.write_text(json.dumps({"|".join(k): v for k, v in res.items()}))
    tmp.replace(cached)
    return res, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="silver_text_rules/silver.duckdb")
    ap.add_argument("--inputs", default="analysis/workdir", help="folder holding the five input CSVs")
    ap.add_argument("--seeds", type=int, default=200, help="extra runs with shuffled tie order (default 200)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="runs in parallel (default: CPU cores minus one)")
    ap.add_argument("--out", default="gold_experiment1/results")
    args = ap.parse_args()

    inputs = (REPO / args.inputs).resolve()
    missing = [f for f in NEEDED if not (inputs / f).is_file()]
    if missing:
        sys.exit(f"Missing in {inputs}: {', '.join(missing)}")
    if not Path(args.db).is_file():
        sys.exit(f"{args.db} not found. Run silver_text_rules/pipeline/build_silver.py first.")

    test_of, reported, reviews = load_silver(Path(args.db))
    with COMMITTED.open(newline="", encoding="utf-8") as fh:
        committed = {(d["test"], d["set"]): (float(d["p"]), int(d["n"])) for d in csv.DictReader(fh)}
    problems = []
    if set(test_of.values()) != {t for t, _ in committed}:
        problems.append("Silver's rules and the committed results do not cover the same 23 tests")
    missing_rep = sorted(set(test_of) - set(reported))
    if missing_rep:
        problems.append("no published result in Silver for: " + ", ".join(missing_rep))

    # Cache key: any change to the script, an input file or the patched line starts a fresh cache.
    key = hashlib.sha256(("".join(sha256(inputs / f) for f in NEEDED) + sha256(SCRIPT)
                          + DECLUSTER_LINE).encode()).hexdigest()[:16]
    cache = REPO / args.out / "cache" / key
    cache.mkdir(parents=True, exist_ok=True)
    seeds = [None] + list(range(args.seeds))
    print(f"{len(seeds)} run(s) (1 as committed + {args.seeds} shuffled tie orders), "
          f"{args.jobs} at a time; cache: {cache}", flush=True)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:   # each task is a separate process
            futs = {pool.submit(one_run, s, inputs, work, cache): s for s in seeds}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    results[futs[fut]], _ = fut.result()
                except Exception as exc:        # stop at once; finished runs stay cached
                    for f in futs:
                        f.cancel()
                    sys.exit(str(exc))
                if i % 20 == 0 or i == len(seeds):
                    print(f"  {i}/{len(seeds)} runs done", flush=True)
    runs = [results[s] for s in seeds]
    as_committed, shuffled = runs[0], runs[1:]

    rows, soft_out = [], 0
    for rid in sorted(test_of):
        test = test_of[rid]
        for k, label in ((1, "discovery"), (2, "holdout")):
            key = (test, label)
            csv_p, csv_n = committed[key]
            silver_p = reported.get(rid, (None, None, None))[k - 1]
            ps = [r[key][0] for r in runs]
            lo, hi, med = min(ps), max(ps), statistics.median(ps)
            a_ok = silver_p is not None and abs(round(csv_p, 3) - silver_p) < 1e-9
            b_ok = all((p < BONF) == (csv_p < BONF) for p in ps)
            c_ok = all(r[key][1] == csv_n for r in runs)
            d_ok = lo - 1e-9 <= csv_p <= hi + 1e-9
            soft_out += not d_ok
            for flag, text in ((a_ok, "Silver's published p differs from the committed CSV"),
                               (b_ok, "a run disagrees with the published Bonferroni verdict"),
                               (c_ok, "declustered event count differs from the committed CSV")):
                if not flag:
                    problems.append(f"{rid} {label}: {text}")
            rows.append({"rule_id": rid, "test": test, "set": label, "silver_p": silver_p,
                         "committed_p": round(csv_p, 6), "as_committed_script_p": round(as_committed[key][0], 6),
                         "min_p": round(lo, 6), "median_p": round(med, 6), "max_p": round(hi, 6),
                         "silver_matches_csv": a_ok, "same_verdict_all_runs": b_ok,
                         "same_event_count": c_ok, "published_inside_range": d_ok})

    out = (REPO / args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "experiment1_gold_check.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    smallest = min(r["min_p"] for r in rows if r["set"] == "holdout")
    summary = {
        "runs": {"as_committed": 1, "shuffled_tie_order": args.seeds},
        "bonferroni_threshold": BONF,
        "smallest_holdout_p_over_all_runs": smallest,
        "rules_checked": len(test_of), "results_checked": len(rows),
        "as_committed_script_identical_rows": sum(
            abs(r["as_committed_script_p"] - r["committed_p"]) < 5e-7 for r in rows),
        "published_inside_range": len(rows) - soft_out,
        "silver_rule_review_status": reviews,
        "inputs_sha256": {f: sha256(inputs / f) for f in NEEDED},
        "big_test_py_sha256": sha256(SCRIPT),
        "hard_checks_passed": not problems, "problems": problems,
    }
    (out / "experiment1_gold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{len(rows)} results from {len(test_of)} rules")
    print(f"  A/B/C hard checks: {'all passed' if not problems else str(len(problems)) + ' problem(s)'}")
    print(f"  D published p-value inside the range of the runs: {len(rows) - soft_out} of {len(rows)}")
    print(f"  as-committed script reproduces the CSV exactly in {summary['as_committed_script_identical_rows']} of {len(rows)} rows")
    print(f"  smallest holdout p over all runs: {smallest:.4f} (Bonferroni threshold {BONF:.4f})")
    if set(reviews) != {"approved"}:
        print(f"  note: Silver review status of these rules is {reviews}; they are not yet human-reviewed.")
    for p in problems:
        print("  PROBLEM:", p)
    print(f"written: {out}/experiment1_gold_check.csv and experiment1_gold_summary.json")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
