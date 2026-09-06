"""
Build the final comparative analysis artifacts for the thesis.

Consumes the per-sample prediction files that already exist in results/ for the
four systems under study:

    - BERT baseline      (results/bert/error_analysis/all_predictions.csv)
    - BERT V2            (results/bert/binary_v2/all_predictions.csv)
    - LLaMA 3.2 3B zero-shot  (results/llama/zero-shot/llama_openrouter_zeroshot_full_test.csv)
    - LLaMA 3.2 3B few-shot   (results/llama/few-shot/llama_openrouter_fewshot_full_test.csv)

All files are aligned row-by-row with data/processed/binary/test.csv and with the
raw LIAR test split (data/raw/liar_dataset/test.tsv), which is verified with
assertions before anything is computed.

Outputs (results/comparison/):
    predictions_merged.csv
    metrics_overall.csv
    metrics_per_class.csv
    confusion_matrices.csv / confusion_matrices.txt
    error_directions.csv
    performance_by_original_label.csv
    confidence_summary.csv
    plots/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from itertools import combinations
from scipy.stats import chi2

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

TEST_CSV = ROOT / "data" / "processed" / "binary" / "test.csv"
TEST_TSV = ROOT / "data" / "raw" / "liar_dataset" / "test.tsv"

SRC = {
    "BERT baseline": ROOT / "results" / "bert" / "error_analysis" / "all_predictions.csv",
    "BERT V2": ROOT / "results" / "bert" / "binary_v2" / "all_predictions.csv",
    "LLaMA 3.2 3B zero-shot": ROOT
    / "results"
    / "llama"
    / "zero-shot"
    / "llama_openrouter_zeroshot_full_test.csv",
    "LLaMA 3.2 3B few-shot": ROOT
    / "results"
    / "llama"
    / "few-shot"
    / "llama_openrouter_fewshot_full_test.csv",
}

OUT = ROOT / "results" / "comparison"
PLOTS = OUT / "plots"

MODELS = list(SRC)  # canonical order used everywhere
CLASSES = ["FAKE", "REAL"]
ORIGINAL_LABELS = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]

# LIAR six-way label -> binary target used throughout the project.
BINARY_MAPPING = {
    "pants-fire": "FAKE",
    "false": "FAKE",
    "barely-true": "FAKE",
    "half-true": "FAKE",
    "mostly-true": "REAL",
    "true": "REAL",
}

# Colour per model (colour-blind friendly, consistent across every figure).
MODEL_COLOURS = {
    "BERT baseline": "#4C72B0",
    "BERT V2": "#55A868",
    "LLaMA 3.2 3B zero-shot": "#C44E52",
    "LLaMA 3.2 3B few-shot": "#8172B2",
}


# ------------------------------------------------------------------
# Load + align
# ------------------------------------------------------------------


def load_aligned() -> pd.DataFrame:
    """Return one dataframe with the gold labels, the original LIAR label and
    every model's per-sample prediction / confidence, aligned by row position."""

    test = pd.read_csv(TEST_CSV)
    tsv = pd.read_csv(TEST_TSV, sep="\t", header=None)

    n = len(test)
    assert len(tsv) == n, f"test.csv has {n} rows, test.tsv has {len(tsv)}"

    true_binary = test["label"].astype(str).str.strip().str.upper()
    original_label = tsv[1].astype(str).str.strip()

    # Sanity: raw statements and derived binary labels line up position-wise.
    assert (
        test["statement"].astype(str).str.strip().values
        == tsv[2].astype(str).str.strip().values
    ).all(), "statement order differs between test.csv and test.tsv"
    assert (
        original_label.map(BINARY_MAPPING).str.upper().values == true_binary.values
    ).all(), "binary mapping of the raw LIAR label does not match test.csv"

    merged = pd.DataFrame(
        {
            "idx": np.arange(n),
            "statement": test["statement"].astype(str),
            "original_label": original_label,
            "true_binary": true_binary,
            "token_length": np.nan,
        }
    )

    for name, path in SRC.items():
        df = pd.read_csv(path)
        assert len(df) == n, f"{name}: expected {n} rows, got {len(df)}"

        if name.startswith("BERT"):
            file_true = df["true_label"].astype(str).str.upper()
            pred = df["predicted_label"].astype(str).str.upper()
            pfake = df.get("prob_fake", df.get("probability_fake"))
            preal = df.get("prob_real", df.get("probability_real"))
            merged[f"prob_fake__{name}"] = pfake.to_numpy()
            merged[f"prob_real__{name}"] = preal.to_numpy()
            merged[f"conf__{name}"] = df["confidence"].to_numpy()
            if "token_length" in df and merged["token_length"].isna().all():
                merged["token_length"] = df["token_length"].to_numpy()
        else:
            file_true = df["label"].astype(str).str.upper()
            pred = df["prediction"].astype(str).str.upper()
            merged[f"raw__{name}"] = df["raw_response"].astype(str).to_numpy()
            merged[f"conf__{name}"] = np.nan  # no logprobs available from provider

        assert (
            file_true.values == true_binary.values
        ).all(), f"{name}: gold labels are not row-aligned with test.csv"

        merged[f"pred__{name}"] = pred.to_numpy()

    return merged


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


def overall_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    y_true = merged["true_binary"].to_numpy()
    rows = []
    for name in MODELS:
        y_pred = merged[f"pred__{name}"].to_numpy()
        pm, rm, fm, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASSES, average="macro", zero_division=0
        )
        pw, rw, fw, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0
        )
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "precision_macro": pm,
                "recall_macro": rm,
                "f1_macro": fm,
                "precision_weighted": pw,
                "recall_weighted": rw,
                "f1_weighted": fw,
                "mcc": matthews_corrcoef(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def per_class_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    y_true = merged["true_binary"].to_numpy()
    rows = []
    for name in MODELS:
        y_pred = merged[f"pred__{name}"].to_numpy()
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASSES, zero_division=0
        )
        for i, cls in enumerate(CLASSES):
            rows.append(
                {
                    "model": name,
                    "class": cls,
                    "precision": p[i],
                    "recall": r[i],
                    "f1": f[i],
                    "support": int(s[i]),
                }
            )
    return pd.DataFrame(rows)


def confusion_tables(merged: pd.DataFrame):
    y_true = merged["true_binary"].to_numpy()
    rows, text = [], []
    for name in MODELS:
        y_pred = merged[f"pred__{name}"].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        rows.append(
            {
                "model": name,
                "true_FAKE_pred_FAKE": cm[0, 0],
                "true_FAKE_pred_REAL": cm[0, 1],
                "true_REAL_pred_FAKE": cm[1, 0],
                "true_REAL_pred_REAL": cm[1, 1],
            }
        )
        text.append(
            f"{name}\n"
            f"                 pred_FAKE  pred_REAL\n"
            f"  true_FAKE      {cm[0,0]:>9}  {cm[0,1]:>9}\n"
            f"  true_REAL      {cm[1,0]:>9}  {cm[1,1]:>9}\n"
        )
    return pd.DataFrame(rows), "\n".join(text)


def error_directions(merged: pd.DataFrame) -> pd.DataFrame:
    y_true = merged["true_binary"].to_numpy()
    rows = []
    for name in MODELS:
        y_pred = merged[f"pred__{name}"].to_numpy()
        wrong = y_true != y_pred
        f2r = int(np.sum(wrong & (y_true == "FAKE")))
        r2f = int(np.sum(wrong & (y_true == "REAL")))
        rows.append(
            {
                "model": name,
                "fake_to_real": f2r,
                "real_to_fake": r2f,
                "total_errors": int(wrong.sum()),
                "error_rate": float(wrong.mean()),
            }
        )
    return pd.DataFrame(rows)


def performance_by_original_label(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in MODELS:
        correct = merged[f"pred__{name}"] == merged["true_binary"]
        for lab in ORIGINAL_LABELS:
            m = merged["original_label"] == lab
            n = int(m.sum())
            c = int((m & correct).sum())
            rows.append(
                {
                    "model": name,
                    "original_label": lab,
                    "samples": n,
                    "correct": c,
                    "incorrect": n - c,
                    "accuracy": c / n if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def accuracy_by_token_length(merged: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per BERT-tokenizer length bucket, for every model.

    Token length comes from the BERT baseline export; it is a property of the
    statement, not of the model, so it is a fair axis for all four systems.
    """
    bins = [0, 16, 32, 64, 128, 10_000]
    names = ["1-16", "17-32", "33-64", "65-128", "129+"]
    grp = pd.cut(merged["token_length"], bins=bins, labels=names, right=True)
    rows = []
    for name in MODELS:
        correct = merged[f"pred__{name}"] == merged["true_binary"]
        for bucket in names:
            m = grp == bucket
            n = int(m.sum())
            c = int((m & correct).sum())
            rows.append(
                {
                    "model": name,
                    "length_bucket": bucket,
                    "samples": n,
                    "correct": c,
                    "accuracy": c / n if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def mcnemar_tests(merged: pd.DataFrame) -> pd.DataFrame:
    """Pairwise McNemar test (with continuity correction) on the test-set
    predictions, plus Cohen's kappa on raw label agreement."""
    y_true = merged["true_binary"].to_numpy()
    hits = {m: (merged[f"pred__{m}"].to_numpy() == y_true) for m in MODELS}
    rows = []
    for a, b in combinations(MODELS, 2):
        b01 = int(np.sum(hits[a] & ~hits[b]))  # a right, b wrong
        b10 = int(np.sum(~hits[a] & hits[b]))  # a wrong, b right
        n = b01 + b10
        if n == 0:
            stat, p = 0.0, 1.0
        else:
            stat = (abs(b01 - b10) - 1) ** 2 / n
            p = float(chi2.sf(stat, df=1))
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "a_correct_b_wrong": b01,
                "a_wrong_b_correct": b10,
                "mcnemar_chi2_cc": stat,
                "p_value": p,
                "significant_0.05": p < 0.05,
                "cohen_kappa_predictions": cohen_kappa_score(
                    merged[f"pred__{a}"], merged[f"pred__{b}"]
                ),
            }
        )
    return pd.DataFrame(rows)


def confidence_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in MODELS:
        conf = merged[f"conf__{name}"]
        if conf.isna().all():
            rows.append(
                {
                    "model": name,
                    "confidence_available": False,
                    "correct_mean": np.nan,
                    "correct_median": np.nan,
                    "incorrect_mean": np.nan,
                    "incorrect_median": np.nan,
                    "high_conf_errors_ge_0.90": np.nan,
                    "high_conf_errors_ge_0.95": np.nan,
                }
            )
            continue
        correct = merged[f"pred__{name}"] == merged["true_binary"]
        cc, ci = conf[correct], conf[~correct]
        rows.append(
            {
                "model": name,
                "confidence_available": True,
                "correct_mean": cc.mean(),
                "correct_median": cc.median(),
                "incorrect_mean": ci.mean(),
                "incorrect_median": ci.median(),
                "high_conf_errors_ge_0.90": int((ci >= 0.90).sum()),
                "high_conf_errors_ge_0.95": int((ci >= 0.95).sum()),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

MAJORITY_ACC = 818 / 1267  # always-predict-FAKE baseline on the LIAR test split


def _bar_labels(ax, fmt="{:.3f}", pad=3):
    for c in ax.containers:
        ax.bar_label(c, fmt=fmt, padding=pad, fontsize=8)


def plot_accuracy(overall: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = overall.set_index("model").loc[MODELS]
    ax.bar(
        range(len(MODELS)),
        order["accuracy"],
        color=[MODEL_COLOURS[m] for m in MODELS],
    )
    ax.axhline(MAJORITY_ACC, ls="--", lw=1.2, color="#555")
    ax.text(
        len(MODELS) - 0.5,
        MAJORITY_ACC + 0.008,
        f"majority-class baseline = {MAJORITY_ACC:.3f}",
        ha="right",
        fontsize=8,
        color="#555",
    )
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 0.8)
    ax.set_title("Test accuracy (LIAR binary, 1267 statements)")
    for i, v in enumerate(order["accuracy"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "accuracy_comparison.png", dpi=150)
    plt.close(fig)


def plot_f1_per_class(per_class: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.35
    x = np.arange(len(MODELS))
    for j, cls in enumerate(CLASSES):
        vals = [
            per_class[(per_class.model == m) & (per_class["class"] == cls)]["f1"].iloc[0]
            for m in MODELS
        ]
        ax.bar(x + (j - 0.5) * width, vals, width, label=f"{cls} F1")
        for xi, v in zip(x + (j - 0.5) * width, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=20, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 0.85)
    ax.set_title("Per-class F1 (FAKE vs REAL)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "f1_per_class.png", dpi=150)
    plt.close(fig)


def plot_macro_weighted(overall: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.35
    x = np.arange(len(MODELS))
    order = overall.set_index("model").loc[MODELS]
    ax.bar(x - width / 2, order["f1_macro"], width, label="Macro F1", color="#4C72B0")
    ax.bar(
        x + width / 2,
        order["f1_weighted"],
        width,
        label="Weighted F1",
        color="#DD8452",
    )
    for xi, v in zip(x - width / 2, order["f1_macro"]):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + width / 2, order["f1_weighted"]):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=20, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 0.75)
    ax.set_title("Macro vs weighted F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "macro_weighted_f1.png", dpi=150)
    plt.close(fig)


def plot_confusion_grid(merged: pd.DataFrame):
    y_true = merged["true_binary"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, name in zip(axes.ravel(), MODELS):
        y_pred = merged[f"pred__{name}"].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1], labels=[f"pred {c}" for c in CLASSES])
        ax.set_yticks([0, 1], labels=[f"true {c}" for c in CLASSES])
        thr = cm.max() / 2
        for r in range(2):
            for c in range(2):
                ax.text(
                    c,
                    r,
                    f"{cm[r, c]}",
                    ha="center",
                    va="center",
                    color="white" if cm[r, c] > thr else "black",
                    fontsize=12,
                )
    fig.suptitle("Confusion matrices (rows = gold, columns = prediction)")
    fig.tight_layout()
    fig.savefig(PLOTS / "confusion_matrix_grid.png", dpi=150)
    plt.close(fig)


def plot_accuracy_by_original_label(by_lab: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.2
    x = np.arange(len(ORIGINAL_LABELS))
    for k, name in enumerate(MODELS):
        vals = [
            by_lab[(by_lab.model == name) & (by_lab.original_label == lab)][
                "accuracy"
            ].iloc[0]
            for lab in ORIGINAL_LABELS
        ]
        ax.bar(
            x + (k - 1.5) * width,
            vals,
            width,
            label=name,
            color=MODEL_COLOURS[name],
        )
    ax.axvline(2.5, color="#999", lw=1, ls=":")
    bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    ax.text(1.0, 1.06, "gold label = FAKE", ha="center", fontsize=9, color="#555", bbox=bbox)
    ax.text(4.0, 1.06, "gold label = REAL", ha="center", fontsize=9, color="#555", bbox=bbox)
    ax.set_xticks(x)
    ax.set_xticklabels(ORIGINAL_LABELS)
    ax.set_ylabel("Accuracy on that slice")
    ax.set_ylim(0, 1.12)
    ax.set_title("Accuracy per original LIAR truthfulness level")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "accuracy_by_original_label.png", dpi=150)
    plt.close(fig)


def plot_bert_confidence(merged: pd.DataFrame):
    bert_models = [m for m in MODELS if m.startswith("BERT")]
    fig, axes = plt.subplots(1, len(bert_models), figsize=(11, 4.2), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), bert_models):
        conf = merged[f"conf__{name}"]
        correct = merged[f"pred__{name}"] == merged["true_binary"]
        ax.hist(
            conf[correct],
            bins=20,
            range=(0.5, 1.0),
            alpha=0.7,
            label="correct",
            color="#55A868",
        )
        ax.hist(
            conf[~correct],
            bins=20,
            range=(0.5, 1.0),
            alpha=0.7,
            label="incorrect",
            color="#C44E52",
        )
        ax.set_title(name)
        ax.set_xlabel("confidence (max softmax prob.)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle("BERT confidence — correct vs incorrect predictions")
    fig.tight_layout()
    fig.savefig(PLOTS / "bert_confidence_distributions.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    merged = load_aligned()
    merged.to_csv(OUT / "predictions_merged.csv", index=False, encoding="utf-8")

    overall = overall_metrics(merged)
    per_class = per_class_metrics(merged)
    cm_df, cm_txt = confusion_tables(merged)
    err_dir = error_directions(merged)
    by_lab = performance_by_original_label(merged)
    conf = confidence_summary(merged)
    by_len = accuracy_by_token_length(merged)
    mcnemar = mcnemar_tests(merged)

    overall.to_csv(OUT / "metrics_overall.csv", index=False)
    per_class.to_csv(OUT / "metrics_per_class.csv", index=False)
    cm_df.to_csv(OUT / "confusion_matrices.csv", index=False)
    (OUT / "confusion_matrices.txt").write_text(cm_txt, encoding="utf-8")
    err_dir.to_csv(OUT / "error_directions.csv", index=False)
    by_lab.to_csv(OUT / "performance_by_original_label.csv", index=False)
    conf.to_csv(OUT / "confidence_summary.csv", index=False)
    by_len.to_csv(OUT / "accuracy_by_token_length.csv", index=False)
    mcnemar.to_csv(OUT / "mcnemar_tests.csv", index=False)

    summary = {
        "test_samples": int(len(merged)),
        "class_support": merged["true_binary"].value_counts().to_dict(),
        "majority_class_accuracy": MAJORITY_ACC,
        "models": MODELS,
        "overall": overall.set_index("model").round(4).to_dict(orient="index"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_accuracy(overall)
    plot_f1_per_class(per_class)
    plot_macro_weighted(overall)
    plot_confusion_grid(merged)
    plot_accuracy_by_original_label(by_lab)
    plot_bert_confidence(merged)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print("\n=== OVERALL METRICS ===")
    print(overall.round(4).to_string(index=False))
    print("\n=== PER-CLASS METRICS ===")
    print(per_class.round(4).to_string(index=False))
    print("\n=== CONFUSION MATRICES ===")
    print(cm_txt)
    print("=== ERROR DIRECTIONS ===")
    print(err_dir.round(4).to_string(index=False))
    print("\n=== ACCURACY BY ORIGINAL LIAR LABEL ===")
    print(
        by_lab.pivot(index="original_label", columns="model", values="accuracy")
        .loc[ORIGINAL_LABELS, MODELS]
        .round(4)
        .to_string()
    )
    print("\n=== CONFIDENCE SUMMARY ===")
    print(conf.round(4).to_string(index=False))
    print("\n=== ACCURACY BY BERT-TOKEN LENGTH ===")
    print(
        by_len.pivot(index="length_bucket", columns="model", values="accuracy")
        .reindex(["1-16", "17-32", "33-64", "65-128", "129+"])[MODELS]
        .round(4)
        .to_string()
    )
    print("\n=== PAIRWISE McNEMAR (test-set predictions) ===")
    print(mcnemar.round(4).to_string(index=False))
    print(f"\nArtifacts written to {OUT}")


if __name__ == "__main__":
    main()