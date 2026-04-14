import argparse
from pathlib import Path

import pandas as pd

from experiment_utils import PROJECTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze baseline vs proposed results across projects")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--projects", nargs="+", default=PROJECTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    rows = []
    for project in args.projects:
        nb_path = results_dir / f"{project}_NB.csv"
        lr_path = results_dir / f"{project}_LR.csv"

        if not nb_path.exists() or not lr_path.exists():
            print(f"Skipping {project}: missing {nb_path.name} or {lr_path.name}")
            continue

        nb = pd.read_csv(nb_path).iloc[0]
        lr = pd.read_csv(lr_path).iloc[0]

        delta_pos_f1 = float(lr["PosF1_mean"] - nb["PosF1_mean"])
        delta_auc = float(lr["AUC_mean"] - nb["AUC_mean"])
        delta_f1 = float(lr["F1_mean"] - nb["F1_mean"])

        rows.append(
            {
                "Project": project,
                "Baseline_PosF1": float(nb["PosF1_mean"]),
                "Proposed_PosF1": float(lr["PosF1_mean"]),
                "Delta_PosF1": delta_pos_f1,
                "Baseline_AUC": float(nb["AUC_mean"]),
                "Proposed_AUC": float(lr["AUC_mean"]),
                "Delta_AUC": delta_auc,
                "Baseline_F1": float(nb["F1_mean"]),
                "Proposed_F1": float(lr["F1_mean"]),
                "Delta_F1": delta_f1,
                "PosF1_Winner": "Proposed" if delta_pos_f1 > 0 else "Baseline",
            }
        )

    if not rows:
        raise SystemExit("No comparable results found")

    out_df = pd.DataFrame(rows)

    summary = pd.DataFrame(
        [
            {
                "Projects_Evaluated": int(len(out_df)),
                "Proposed_Wins_on_PosF1": int((out_df["Delta_PosF1"] > 0).sum()),
                "Baseline_Wins_on_PosF1": int((out_df["Delta_PosF1"] <= 0).sum()),
                "Mean_Delta_PosF1": float(out_df["Delta_PosF1"].mean()),
                "Mean_Delta_AUC": float(out_df["Delta_AUC"].mean()),
                "Mean_Delta_F1": float(out_df["Delta_F1"].mean()),
            }
        ]
    )

    detail_path = results_dir / "all_projects_metric_deltas.csv"
    summary_path = results_dir / "all_projects_summary_stats.csv"

    out_df.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\\n=== Per-Project Deltas ===\\n")
    print(out_df.to_string(index=False))
    print("\\n=== Overall Summary ===\\n")
    print(summary.to_string(index=False))
    print(f"\\nSaved: {detail_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
