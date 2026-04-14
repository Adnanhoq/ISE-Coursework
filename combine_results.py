import argparse
from pathlib import Path

import pandas as pd

from experiment_utils import PROJECTS, ensure_results_dir, format_report_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine model result CSVs into clean report tables")
    parser.add_argument("--project", type=str, default="pytorch", help="project name or 'all'")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def _load_pair(results_dir: Path, project: str) -> pd.DataFrame:
    nb_path = results_dir / f"{project}_NB.csv"
    lr_path = results_dir / f"{project}_LR.csv"
    if not nb_path.exists() or not lr_path.exists():
        missing = [str(p) for p in [nb_path, lr_path] if not p.exists()]
        raise FileNotFoundError(f"Missing required result files for '{project}': {missing}")
    return pd.concat([pd.read_csv(nb_path), pd.read_csv(lr_path)], ignore_index=True)


def _save_single_project(results_dir: Path, project: str) -> Path:
    combined = _load_pair(results_dir, project)
    final_table = format_report_metrics(combined)
    out_path = results_dir / f"{project}_combined_results.csv"
    final_table.to_csv(out_path, index=False)
    print("\\n=== Combined Results ===\\n")
    print(final_table.to_string(index=False))
    print(f"\\nSaved combined table to {out_path}")
    return out_path


def _save_all_projects(results_dir: Path) -> Path:
    rows = []
    for project in PROJECTS:
        rows.append(_load_pair(results_dir, project))

    combined = pd.concat(rows, ignore_index=True)
    final_table = format_report_metrics(combined)
    out_path = results_dir / "all_projects_combined_results.csv"
    final_table.to_csv(out_path, index=False)
    print("\\n=== Combined Results (All Projects) ===\\n")
    print(final_table.to_string(index=False))
    print(f"\\nSaved combined table to {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    results_dir = ensure_results_dir(args.results_dir)

    if args.project == "all":
        _save_all_projects(results_dir)
    else:
        _save_single_project(results_dir, args.project)


if __name__ == "__main__":
    main()
