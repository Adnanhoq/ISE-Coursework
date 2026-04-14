import argparse
import subprocess
import sys
from pathlib import Path

from experiment_utils import PROJECTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete experimental pipeline")
    parser.add_argument("--projects", nargs="+", default=PROJECTS, help="List of projects to run")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    py = sys.executable

    for project in args.projects:
        run_cmd(
            [
                py,
                "baseline_nb.py",
                "--project",
                project,
                "--repeats",
                str(args.repeats),
                "--test-size",
                str(args.test_size),
                "--max-features",
                str(args.max_features),
                "--seed-base",
                str(args.seed_base),
                "--results-dir",
                args.results_dir,
            ]
        )

        run_cmd(
            [
                py,
                "classification.py",
                "--project",
                project,
                "--repeats",
                str(args.repeats),
                "--test-size",
                str(args.test_size),
                "--max-features",
                str(args.max_features),
                "--seed-base",
                str(args.seed_base),
                "--results-dir",
                args.results_dir,
            ]
        )

        run_cmd([py, "combine_results.py", "--project", project, "--results-dir", args.results_dir])
        run_cmd([py, "stats_test.py", "--project", project, "--results-dir", args.results_dir])

    if set(args.projects) == set(PROJECTS):
        run_cmd([py, "combine_results.py", "--project", "all", "--results-dir", args.results_dir])
        run_cmd([py, "stats_test.py", "--project", "all", "--results-dir", args.results_dir])

    print("\\nPipeline completed successfully.")
    print(f"Results written under: {Path(args.results_dir).resolve()}")


if __name__ == "__main__":
    main()
