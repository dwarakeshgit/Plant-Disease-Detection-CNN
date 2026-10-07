"""
evaluate.py - Evaluate the saved CNN on the EXISTING validation folder.

IMPORTANT:
These are VALIDATION-SET results.
The validation set was used by EarlyStopping and ModelCheckpoint,
so these results should NOT be presented as an independent test result.

Run:
    python evaluate.py
"""

import os
import json
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

VAL_DIR = os.path.join("PlantVillage_15", "val")
MODEL_PATH = os.path.join("model", "plant_cnn.keras")
CLASSES_PATH = os.path.join("model", "class_names.json")
HISTORY_PATH = os.path.join("results", "history.json")

RESULTS_DIR = "results"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

TAG = "Validation-set evaluation"


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    MODEL_PATH,
    CLASSES_PATH,
    HISTORY_PATH,
]

for path in required_files:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
            f"Make sure the training step has been completed."
        )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print(f"Model loaded: {MODEL_PATH}")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print(f"Number of classes: {len(class_names)}")


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Validation dataset loaded.")


# ============================================================
# GET TRUE LABELS
# ============================================================

y_true = np.concatenate(
    [y.numpy() for _, y in val_ds],
    axis=0
)


# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\nRunning predictions...")

# The model already contains the Rescaling(1./255) layer,
# so raw 0-255 images can be passed directly.
image_ds = val_ds.map(lambda x, y: x)

probs = model.predict(
    image_ds,
    verbose=1
)

y_pred = np.argmax(probs, axis=1)

assert len(y_true) == len(y_pred), (
    "Label/prediction count mismatch."
)


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
)

precision_weighted, recall_weighted, f1_weighted, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
)


# Per-class metrics
precision_class, recall_class, f1_class, support_class = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        zero_division=0,
    )
)


# Classification report
report = classification_report(
    y_true,
    y_pred,
    labels=range(len(class_names)),
    target_names=class_names,
    digits=4,
    zero_division=0,
)


# Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names)),
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print(f"{TAG}")
print("=" * 70)

print(f"Number of validation images : {len(y_true)}")
print(f"Accuracy                   : {accuracy * 100:.2f}%")

print(
    f"Precision (macro)          : "
    f"{precision_macro * 100:.2f}%"
)

print(
    f"Recall (macro)             : "
    f"{recall_macro * 100:.2f}%"
)

print(
    f"F1-score (macro)           : "
    f"{f1_macro * 100:.2f}%"
)

print(
    f"Precision (weighted)       : "
    f"{precision_weighted * 100:.2f}%"
)

print(
    f"Recall (weighted)          : "
    f"{recall_weighted * 100:.2f}%"
)

print(
    f"F1-score (weighted)        : "
    f"{f1_weighted * 100:.2f}%"
)

print("\nClassification Report")
print("-" * 70)
print(report)

print("\nConfusion Matrix")
print("-" * 70)
print(cm)


# ============================================================
# SAVE EVALUATION METRICS
# ============================================================

metrics = {
    "evaluation_set": (
        "validation "
        "(also used by EarlyStopping/ModelCheckpoint)"
    ),

    "num_images": int(len(y_true)),

    "accuracy": float(accuracy),

    "macro": {
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1": float(f1_macro),
    },

    "weighted": {
        "precision": float(precision_weighted),
        "recall": float(recall_weighted),
        "f1": float(f1_weighted),
    },

    "per_class": {
        class_name: {
            "precision": float(precision_class[i]),
            "recall": float(recall_class[i]),
            "f1": float(f1_class[i]),
            "support": int(support_class[i]),
        }
        for i, class_name in enumerate(class_names)
    },

    "confusion_matrix": cm.tolist(),

    "class_names": class_names,
}


evaluation_json_path = os.path.join(
    RESULTS_DIR,
    "evaluation_metrics.json"
)

with open(
    evaluation_json_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=2
    )


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"{TAG}\n\n"
    )

    f.write(
        f"Number of validation images: {len(y_true)}\n"
    )

    f.write(
        f"Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    f.write(report)


# ============================================================
# CREATE CONFUSION MATRIX IMAGE
# ============================================================

print("\nCreating confusion matrix image...")

short_names = [
    class_name.replace("___", "\n")
    for class_name in class_names
]

fig, ax = plt.subplots(
    figsize=(12, 10)
)

im = ax.imshow(
    cm,
    cmap="Greys"
)

ax.set_xticks(
    range(len(class_names))
)

ax.set_yticks(
    range(len(class_names))
)

ax.set_xticklabels(
    short_names,
    rotation=45,
    ha="right",
    fontsize=8
)

ax.set_yticklabels(
    short_names,
    fontsize=8
)


threshold = cm.max() / 2.0

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            ),
            fontsize=9,
        )


ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

ax.set_title(
    f"Confusion Matrix - {TAG}"
)

fig.colorbar(
    im,
    ax=ax
)

fig.tight_layout()

confusion_matrix_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

fig.savefig(
    confusion_matrix_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# LOAD TRAINING HISTORY
# ============================================================

print("\nLoading training history...")

with open(
    HISTORY_PATH,
    "r",
    encoding="utf-8"
) as f:

    history = json.load(f)


# ============================================================
# TRAINING ACCURACY GRAPH
# ============================================================

if "accuracy" in history and "val_accuracy" in history:

    epochs = range(
        1,
        len(history["accuracy"]) + 1
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        epochs,
        history["accuracy"],
        label="Training Accuracy"
    )

    ax.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    ax.set_title(
        "Training and Validation Accuracy"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    accuracy_graph_path = os.path.join(
        RESULTS_DIR,
        "training_accuracy.png"
    )

    fig.savefig(
        accuracy_graph_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

else:

    print(
        "Warning: accuracy history not found."
    )


# ============================================================
# TRAINING LOSS GRAPH
# ============================================================

if "loss" in history and "val_loss" in history:

    epochs = range(
        1,
        len(history["loss"]) + 1
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        epochs,
        history["loss"],
        label="Training Loss"
    )

    ax.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.set_title(
        "Training and Validation Loss"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    loss_graph_path = os.path.join(
        RESULTS_DIR,
        "training_loss.png"
    )

    fig.savefig(
        loss_graph_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

else:

    print(
        "Warning: loss history not found."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(
    f"1. {evaluation_json_path}"
)

print(
    f"2. {report_path}"
)

print(
    f"3. {confusion_matrix_path}"
)

if "accuracy" in history and "val_accuracy" in history:
    print(
        f"4. {os.path.join(RESULTS_DIR, 'training_accuracy.png')}"
    )

if "loss" in history and "val_loss" in history:
    print(
        f"5. {os.path.join(RESULTS_DIR, 'training_loss.png')}"
    )

print("\nRemember:")
print(
    "These are validation-set results, not independent test-set results."
)