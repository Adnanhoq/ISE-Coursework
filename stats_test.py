import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiment_utils import PROJECTS, ensure_results_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wilcoxon signed-rank test on positive-class F1")
    parser.add_argument("--project", type=str, default="pytorch", help="project name or 'all'")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def _compute_effect_size(stat: float, n: int) -> float:
    expected_w = n * (n + 1) / 4
    sd_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (stat - expected_w) / sd_w
    return float(abs(z) / np.sqrt(n))


def run_test_for_project(project: str, results_dir: Path, alpha: float) -> pd.DataFrame:
    baseline_path = results_dir / f"{project}_NB_pos_f1.npy"
    proposed_path = results_dir / f"{project}_LR_pos_f1.npy"

    if not baseline_path.exists() or not proposed_path.exists():
        missing = [str(p) for p in [baseline_path, proposed_path] if not p.exists()]
        raise FileNotFoundError(f"Missing required .npy files for '{project}': {missing}")

    baseline = np.load(baseline_path)
    proposed = np.load(proposed_path)

    if len(baseline) != len(proposed):
        raise ValueError(f"Mismatch in number of runs for '{project}': {len(baseline)} vs {len(proposed)}")

    stat, p_value = wilcoxon(baseline, proposed)
    effect_size = _compute_effect_size(float(stat), len(baseline))

    row = {
        "Project": project,
        "Metric": "Positive-class F1",
        "Runs": int(len(baseline)),
        "Statistic": float(stat),
        "p_value": float(p_value),
        "Effect_size_r": effect_size,
        "Significant_at_alpha": bool(p_value < alpha),
        "Alpha": float(alpha),
        "Baseline_PosF1_mean": float(np.mean(baseline)),
        "Proposed_PosF1_mean": float(np.mean(proposed)),
        "Delta_Proposed_minus_Baseline": float(np.mean(proposed - baseline)),
    }
    return pd.DataFrame([row])


def main() -> None:
    args = parse_args()
    results_dir = ensure_results_dir(args.results_dir)

    if args.project == "all":
        frames = [run_test_for_project(project, results_dir, args.alpha) for project in PROJECTS]
        out_df = pd.concat(frames, ignore_index=True)
        out_path = results_dir / "all_projects_wilcoxon_test.csv"
    else:
        out_df = run_test_for_project(args.project, results_dir, args.alpha)
        out_path = results_dir / f"{args.project}_wilcoxon_test.csv"

    out_df.to_csv(out_path, index=False)

    print("\\n=== Wilcoxon Signed-Rank Test (Positive-class F1) ===\\n")
    print(out_df.to_string(index=False))
    print(f"\\nSaved statistical results to {out_path}")


if __name__ == "__main__":
    main()
