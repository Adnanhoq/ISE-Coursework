import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

PROJECTS = ["pytorch", "tensorflow", "keras", "incubator-mxnet", "caffe"]

FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
    "with", "this", "these", "those", "or", "if", "but", "not", "we", "you", "they",
    "i", "me", "my", "our", "us", "your", "their", "them", "can", "could", "should",
    "would", "do", "does", "did", "done", "have", "had", "having", "than", "then",
    "there", "here", "when", "where", "why", "how", "what", "which", "who", "whom",
}


@dataclass
class ExperimentConfig:
    project: str
    repeats: int = 30
    test_size: float = 0.3
    max_features: int = 1000
    random_seed_base: int = 0
    ngram_min: int = 1
    ngram_max: int = 2


def ensure_results_dir(results_dir: str = "results") -> Path:
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _remove_html(text: str) -> str:
    html = re.compile(r"<.*?>")
    return html.sub("", text)


def _remove_emoji(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        + "\U0001F600-\U0001F64F"
        + "\U0001F300-\U0001F5FF"
        + "\U0001F680-\U0001F6FF"
        + "\U0001F1E0-\U0001F1FF"
        + "\U00002702-\U000027B0"
        + "\U000024C2-\U0001F251"
        + "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def _remove_stopwords(text: str, stop_words: set[str]) -> str:
    return " ".join([word for word in str(text).split() if word not in stop_words])


def _clean_str(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", text)
    text = re.sub(r"\'s", " 's", text)
    text = re.sub(r"\'ve", " 've", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r'"', "", text)
    return text.strip().lower()


def load_project_data(project: str) -> pd.DataFrame:
    dataset_path = Path("datasets") / f"{project}.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = pd.read_csv(dataset_path).sample(frac=1, random_state=999)
    data["Title+Body"] = data.apply(
        lambda row: row["Title"] + ". " + row["Body"] if pd.notna(row["Body"]) else row["Title"],
        axis=1,
    )
    prepared = data.rename(
        columns={"Unnamed: 0": "id", "class": "sentiment", "Title+Body": "text"}
    )[["id", "Number", "sentiment", "text"]]
    return prepared


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().fillna("")
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        warnings.warn(
            "NLTK stopwords corpus not available. Falling back to built-in stopword list.",
            RuntimeWarning,
            stacklevel=2,
        )
        stop_words = set(FALLBACK_STOPWORDS)
    stop_words.add("...")

    df["text"] = df["text"].apply(_remove_html)
    df["text"] = df["text"].apply(_remove_emoji)
    df["text"] = df["text"].apply(lambda x: _remove_stopwords(x, stop_words))
    df["text"] = df["text"].apply(_clean_str)
    return df


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def run_repeated_experiment(
    data: pd.DataFrame,
    config: ExperimentConfig,
    fit_and_predict: Callable,
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    metrics = {
        "accuracy": [],
        "precision_macro": [],
        "recall_macro": [],
        "f1_macro": [],
        "auc": [],
        "precision_pos": [],
        "recall_pos": [],
        "f1_pos": [],
    }

    text_col = "text"
    indices = np.arange(data.shape[0])

    for repeat in range(config.repeats):
        train_idx, test_idx = train_test_split(
            indices,
            test_size=config.test_size,
            random_state=config.random_seed_base + repeat,
            stratify=data["sentiment"],
        )

        train_text = data[text_col].iloc[train_idx]
        test_text = data[text_col].iloc[test_idx]
        y_train = data["sentiment"].iloc[train_idx]
        y_test = data["sentiment"].iloc[test_idx]

        tfidf = TfidfVectorizer(
            ngram_range=(config.ngram_min, config.ngram_max),
            max_features=config.max_features,
        )
        x_train = tfidf.fit_transform(train_text)
        x_test = tfidf.transform(test_text)

        y_pred, y_scores = fit_and_predict(x_train, y_train, x_test, repeat)

        metrics["accuracy"].append(float(accuracy_score(y_test, y_pred)))
        metrics["precision_macro"].append(float(precision_score(y_test, y_pred, average="macro", zero_division=0)))
        metrics["recall_macro"].append(float(recall_score(y_test, y_pred, average="macro", zero_division=0)))
        metrics["f1_macro"].append(float(f1_score(y_test, y_pred, average="macro", zero_division=0)))

        metrics["precision_pos"].append(float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)))
        metrics["recall_pos"].append(float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)))
        metrics["f1_pos"].append(float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)))

        fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=1)
        metrics["auc"].append(float(auc(fpr, tpr)))

    summary = pd.DataFrame(
        {
            "Runs": [config.repeats],
            "Accuracy_mean": [_mean_std(metrics["accuracy"])[0]],
            "Accuracy_std": [_mean_std(metrics["accuracy"])[1]],
            "Precision_mean": [_mean_std(metrics["precision_macro"])[0]],
            "Precision_std": [_mean_std(metrics["precision_macro"])[1]],
            "Recall_mean": [_mean_std(metrics["recall_macro"])[0]],
            "Recall_std": [_mean_std(metrics["recall_macro"])[1]],
            "F1_mean": [_mean_std(metrics["f1_macro"])[0]],
            "F1_std": [_mean_std(metrics["f1_macro"])[1]],
            "AUC_mean": [_mean_std(metrics["auc"])[0]],
            "AUC_std": [_mean_std(metrics["auc"])[1]],
            "PosPrecision_mean": [_mean_std(metrics["precision_pos"])[0]],
            "PosPrecision_std": [_mean_std(metrics["precision_pos"])[1]],
            "PosRecall_mean": [_mean_std(metrics["recall_pos"])[0]],
            "PosRecall_std": [_mean_std(metrics["recall_pos"])[1]],
            "PosF1_mean": [_mean_std(metrics["f1_pos"])[0]],
            "PosF1_std": [_mean_std(metrics["f1_pos"])[1]],
            "Project": [config.project],
            "TestSize": [config.test_size],
            "MaxFeatures": [config.max_features],
        }
    )

    return summary, metrics


def format_report_metrics(df: pd.DataFrame) -> pd.DataFrame:
    def fmt(mean: float, std: float) -> str:
        return f"{mean:.4f} +/- {std:.4f}"

    table = df.copy()
    table["Accuracy"] = table.apply(lambda r: fmt(r["Accuracy_mean"], r["Accuracy_std"]), axis=1)
    table["Precision"] = table.apply(lambda r: fmt(r["Precision_mean"], r["Precision_std"]), axis=1)
    table["Recall"] = table.apply(lambda r: fmt(r["Recall_mean"], r["Recall_std"]), axis=1)
    table["F1"] = table.apply(lambda r: fmt(r["F1_mean"], r["F1_std"]), axis=1)
    table["AUC"] = table.apply(lambda r: fmt(r["AUC_mean"], r["AUC_std"]), axis=1)
    table["PosRecall"] = table.apply(lambda r: fmt(r["PosRecall_mean"], r["PosRecall_std"]), axis=1)
    table["PosF1"] = table.apply(lambda r: fmt(r["PosF1_mean"], r["PosF1_std"]), axis=1)

    preferred_order = [
        "Project",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC",
        "PosRecall",
        "PosF1",
    ]
    return table[preferred_order]
