import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression

from experiment_utils import ExperimentConfig, ensure_results_dir, load_project_data, preprocess_text, run_repeated_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Logistic Regression model for bug report classification")
    parser.add_argument("--project", type=str, default="pytorch")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def fit_and_predict_lr(x_train, y_train, x_test, repeat):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=repeat,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_scores = model.predict_proba(x_test)[:, 1]
    return y_pred, y_scores


def main() -> None:
    args = parse_args()
    results_dir = ensure_results_dir(args.results_dir)

    config = ExperimentConfig(
        project=args.project,
        repeats=args.repeats,
        test_size=args.test_size,
        max_features=args.max_features,
        random_seed_base=args.seed_base,
    )

    data = preprocess_text(load_project_data(config.project))
    data.to_csv("Title+Body.csv", index=False)

    summary, metrics = run_repeated_experiment(data, config, fit_and_predict_lr)
    summary.insert(0, "Model", "Logistic Regression")

    summary_path = results_dir / f"{config.project}_LR.csv"
    summary.to_csv(summary_path, index=False)

    np.save(results_dir / f"{config.project}_LR_pos_f1.npy", np.array(metrics["f1_pos"], dtype=float))

    if config.project == "pytorch":
        np.save(results_dir / "proposed_f1_pos.npy", np.array(metrics["f1_pos"], dtype=float))

    print("=== Logistic Regression + TF-IDF Results ===")
    print(summary.to_string(index=False))
    print(f"\\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
