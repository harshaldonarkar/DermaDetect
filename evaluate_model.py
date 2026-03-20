"""
Comprehensive evaluation script for DermaDetect — research paper grade.

For each trained model found it computes:
  - Overall accuracy
  - Per-class Precision, Recall (Sensitivity), F1-score, Specificity
  - Macro-averaged AUC (One-vs-Rest)
  - ROC curves per class
  - Confusion matrix (counts + normalised)

Across all models it produces:
  - Comparison summary table (CSV + PNG)
  - Overlaid ROC curves for each class

Outputs go to:  evaluation_results/
    ├── <model>_report.txt
    ├── <model>_confusion_matrix.png
    ├── <model>_roc_curves.png
    ├── comparison_summary.csv
    ├── comparison_accuracy_auc.png
    └── roc_overlay_<class>.png

Usage:
    python evaluate_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
from PIL import Image
import tensorflow as tf

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "evaluation_results"
IMG_SIZE   = 224
BATCH_SIZE = 32

HAM_METADATA_CSV = "HAM10000_metadata.csv"
HAM_IMAGE_DIRS   = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
HAM_LABEL_MAP    = {
    "akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "nv": 4, "vasc": 5, "mel": 6,
}
ISIC_METADATA_CSV = "ISIC_2019_Training_GroundTruth.csv"
ISIC_IMAGE_DIR    = "ISIC_2019_Training_Input"
ISIC_LABEL_MAP    = {
    "MEL": 6, "NV": 4, "BCC": 1, "AK": 0, "BKL": 2, "DF": 3, "VASC": 5, "SCC": 0,
}

CLASS_NAMES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanocytic Nevi",
    "Pyogenic Granuloma",
    "Melanoma",
]
N_CLASSES = len(CLASS_NAMES)

# Models to evaluate — (display name, model file, preprocess_fn)
MODEL_CONFIGS = [
    ("EfficientNetB0",
     "best_model_efficientnet.keras",
     None),   # internal preprocessing
    ("VGG16",
     "best_model_vgg16.keras",
     tf.keras.applications.vgg16.preprocess_input),
    ("ResNet50",
     "best_model_resnet50.keras",
     tf.keras.applications.resnet50.preprocess_input),
    ("MobileNetV2",
     "best_model_mobilenetv2.keras",
     tf.keras.applications.mobilenet_v2.preprocess_input),
    ("InceptionV3",
     "best_model_inceptionv3.keras",
     tf.keras.applications.inception_v3.preprocess_input),
]

PALETTE = ["#175810", "#c0392b", "#2980b9", "#8e44ad", "#e67e22"]
# ─────────────────────────────────────────────────────────────────────────────


def load_ham10000():
    df = pd.read_csv(HAM_METADATA_CSV)
    df["label"] = df["dx"].map(HAM_LABEL_MAP)

    def find_path(image_id):
        for d in HAM_IMAGE_DIRS:
            p = os.path.join(d, f"{image_id}.jpg")
            if os.path.exists(p):
                return p
        return None

    df["path"] = df["image_id"].apply(find_path)
    return df.dropna(subset=["path", "label"])[["path", "label"]]


def load_isic2019():
    df = pd.read_csv(ISIC_METADATA_CSV)
    class_cols = [c for c in ISIC_LABEL_MAP if c in df.columns]

    def get_label(row):
        for col in class_cols:
            if row[col] == 1.0:
                return ISIC_LABEL_MAP[col]
        return None

    df["label"] = df.apply(get_label, axis=1)
    df["path"]  = df["image"].apply(lambda x: os.path.join(ISIC_IMAGE_DIR, f"{x}.jpg"))
    df = df.dropna(subset=["label"])
    df = df[df["path"].apply(os.path.exists)]
    return df[["path", "label"]]


def load_all():
    frames = []
    if os.path.exists(HAM_METADATA_CSV):
        frames.append(load_ham10000())
    if os.path.exists(ISIC_METADATA_CSV):
        frames.append(load_isic2019())
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].astype(int)
    return df


def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32)


def predict_in_batches(model, paths, preprocess_fn=None):
    """Returns (y_pred_classes, y_proba) for all paths."""
    all_proba = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i: i + BATCH_SIZE]
        batch = np.stack([preprocess_image(p) for p in batch_paths])
        if preprocess_fn is not None:
            batch = preprocess_fn(batch)
        proba = model.predict(batch, verbose=0)
        all_proba.append(proba)
        print(f"  Processed {min(i + BATCH_SIZE, len(paths))}/{len(paths)}", end="\r")
    print()
    y_proba = np.concatenate(all_proba, axis=0)
    y_pred  = np.argmax(y_proba, axis=1)
    return y_pred, y_proba


def compute_specificity(y_true, y_pred, n_classes):
    """Per-class specificity = TN / (TN + FP)."""
    specificities = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        tn = len(y_true) - tp - fp - fn
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    return specificities


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        [f"{model_name} — Confusion Matrix (counts)",
         f"{model_name} — Confusion Matrix (normalised)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Greens",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, linewidths=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_roc_curves(y_true_bin, y_proba, model_name):
    """Plot per-class ROC curves for one model."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    auc_scores = []
    for i, (cls_name, ax) in enumerate(zip(CLASS_NAMES, axes)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        ax.plot(fpr, tpr, color="#175810", lw=2,
                label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(cls_name, fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)

    # Hide last empty subplot
    if len(CLASS_NAMES) < len(axes):
        axes[-1].axis("off")

    macro_auc = np.mean(auc_scores)
    fig.suptitle(f"{model_name} — ROC Curves (Macro AUC = {macro_auc:.3f})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{model_name.lower().replace(' ', '_')}_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return auc_scores, macro_auc, path


def plot_roc_overlay(all_results, class_idx):
    """Overlay ROC curves for all models for a single class."""
    cls_name = CLASS_NAMES[class_idx]
    fig, ax  = plt.subplots(figsize=(7, 6))

    for (model_name, _, y_true_bin, y_proba), color in zip(all_results, PALETTE):
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_proba[:, class_idx])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{model_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Comparison — {cls_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    safe_name = cls_name.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"roc_overlay_{safe_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_comparison_chart(summary_df):
    """Bar chart comparing accuracy and AUC across all models."""
    models  = summary_df["Model"].tolist()
    accs    = summary_df["Accuracy (%)"].tolist()
    aucs    = [v * 100 for v in summary_df["Macro AUC"].tolist()]
    x       = np.arange(len(models))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1   = ax.bar(x - width/2, accs, width, label="Accuracy (%)", color="#175810", alpha=0.85)
    bars2   = ax.bar(x + width/2, aucs, width, label="Macro AUC × 100", color="#5da64e", alpha=0.85)

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Comparison — Accuracy vs Macro AUC", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim([0, 105])
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.1f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.1f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "comparison_accuracy_auc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def evaluate_one(model_name, model_path, preprocess_fn, test_df):
    """Load a model and compute full metrics. Returns a results dict."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")

    model = tf.keras.models.load_model(model_path, compile=False)

    y_true          = test_df["label"].values
    y_true_bin      = label_binarize(y_true, classes=list(range(N_CLASSES)))
    y_pred, y_proba = predict_in_batches(model, test_df["path"].values, preprocess_fn)

    accuracy   = (y_true == y_pred).mean() * 100
    macro_auc  = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
    specificity = compute_specificity(y_true, y_pred, N_CLASSES)

    # Per-class AUC
    per_class_auc = []
    for i in range(N_CLASSES):
        per_class_auc.append(roc_auc_score(y_true_bin[:, i], y_proba[:, i]))

    print(f"\n  Accuracy : {accuracy:.2f}%")
    print(f"  Macro AUC: {macro_auc:.4f}")
    print(f"\nPer-class report:\n{report_str}")

    # Specificity table
    print("  Per-class Specificity:")
    for name, spec in zip(CLASS_NAMES, specificity):
        print(f"    {name:<25} {spec:.3f}")

    # Save individual report
    safe_name   = model_name.lower().replace(" ", "_")
    report_path = os.path.join(OUTPUT_DIR, f"{safe_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Macro AUC (OvR):  {macro_auc:.4f}\n\n")
        f.write(report_str)
        f.write("\nPer-class Specificity:\n")
        for name, spec in zip(CLASS_NAMES, specificity):
            f.write(f"  {name:<25} {spec:.3f}\n")
        f.write("\nPer-class AUC:\n")
        for name, a in zip(CLASS_NAMES, per_class_auc):
            f.write(f"  {name:<25} {a:.4f}\n")

    # Confusion matrix
    cm_path = plot_confusion_matrix(y_true, y_pred, model_name)
    print(f"\n  Saved confusion matrix: {cm_path}")

    # ROC curves
    auc_scores, _, roc_path = plot_roc_curves(y_true_bin, y_proba, model_name)
    print(f"  Saved ROC curves: {roc_path}")

    return {
        "model_name": model_name,
        "accuracy":   accuracy,
        "macro_auc":  macro_auc,
        "y_true_bin": y_true_bin,
        "y_proba":    y_proba,
        "per_class_auc":  per_class_auc,
        "specificity":    specificity,
        "report_str":     report_str,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load shared test set ──────────────────────────────────────────────────
    print("Loading dataset...")
    df = load_all()
    print(f"  Total samples: {len(df)}")
    _, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    print(f"  Test samples : {len(test_df)}")

    # ── Evaluate each available model ────────────────────────────────────────
    results     = []
    all_results = []   # for overlay ROC

    for model_name, model_path, preprocess_fn in MODEL_CONFIGS:
        if not os.path.exists(model_path):
            print(f"\n  [SKIP] {model_name} — {model_path} not found.")
            continue
        r = evaluate_one(model_name, model_path, preprocess_fn, test_df)
        results.append(r)
        all_results.append((model_name, model_path, r["y_true_bin"], r["y_proba"]))

    if not results:
        print("\nNo trained models found. Train at least one model first.")
        return

    # ── Comparison summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")

    rows = []
    for r in results:
        report_dict = classification_report(
            np.argmax(r["y_true_bin"], axis=1),
            np.argmax(r["y_proba"], axis=1),
            target_names=CLASS_NAMES, output_dict=True
        )
        row = {
            "Model":        r["model_name"],
            "Accuracy (%)": round(r["accuracy"], 2),
            "Macro AUC":    round(r["macro_auc"], 4),
            "Macro F1":     round(report_dict["macro avg"]["f1-score"], 4),
            "Macro Recall": round(report_dict["macro avg"]["recall"], 4),
            "Macro Prec.":  round(report_dict["macro avg"]["precision"], 4),
            "Melanoma AUC": round(r["per_class_auc"][6], 4),
            "Melanoma Rec.":round(report_dict["Melanoma"]["recall"], 4),
            "Melanoma Spec.":round(r["specificity"][6], 4),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False)
    print(summary_df.to_string(index=False))

    csv_path = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Saved summary: {csv_path}")

    # ── Comparison bar chart ──────────────────────────────────────────────────
    chart_path = plot_comparison_chart(summary_df)
    print(f"  Saved comparison chart: {chart_path}")

    # ── ROC overlay per class (only if 2+ models available) ──────────────────
    if len(all_results) >= 2:
        print("\nGenerating per-class ROC overlay plots...")
        for i in range(N_CLASSES):
            path = plot_roc_overlay(all_results, i)
            print(f"  {CLASS_NAMES[i]}: {path}")

    print(f"\nAll results saved to ./{OUTPUT_DIR}/")
    print("\nKey files for paper:")
    print(f"  - comparison_summary.csv         → Table 1 (comparison table)")
    print(f"  - comparison_accuracy_auc.png    → Figure: bar chart")
    print(f"  - *_confusion_matrix.png         → Figure: confusion matrices")
    print(f"  - *_roc_curves.png               → Figure: ROC per model")
    print(f"  - roc_overlay_melanoma.png       → Figure: Melanoma ROC comparison")


if __name__ == "__main__":
    main()
