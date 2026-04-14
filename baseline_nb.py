import argparse

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

from experiment_utils import ExperimentConfig, ensure_results_dir, load_project_data, preprocess_text, run_repeated_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Naive Bayes baseline for bug report classification")
    parser.add_argument("--project", type=str, default="pytorch")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def fit_and_predict_nb(x_train, y_train, x_test, _repeat):
    params = {"var_smoothing": np.logspace(-12, 0, 13)}
    clf = GaussianNB()
    grid = GridSearchCV(clf, params, cv=5, scoring="roc_auc")
    grid.fit(x_train.toarray(), y_train)
    best = grid.best_estimator_
    y_pred = best.predict(x_test.toarray())
    y_scores = best.predict_proba(x_test.toarray())[:, 1]
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

    summary, metrics = run_repeated_experiment(data, config, fit_and_predict_nb)
    summary.insert(0, "Model", "Naive Bayes")

    summary_path = results_dir / f"{config.project}_NB.csv"
    summary.to_csv(summary_path, index=False)

    np.save(results_dir / f"{config.project}_NB_pos_f1.npy", np.array(metrics["f1_pos"], dtype=float))

    if config.project == "pytorch":
        np.save(results_dir / "baseline_f1_pos.npy", np.array(metrics["f1_pos"], dtype=float))

    print("=== Naive Bayes + TF-IDF Results ===")
    print(summary.to_string(index=False))
    print(f"\\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
