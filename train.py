"""
train.py - Plant Disease Detection (CNN, TensorFlow/Keras)
Uses the existing PlantVillage_15/train and PlantVillage_15/val folders as-is.

  python train.py            -> inspect only (dataset info + model summary)
  python train.py --train    -> train the CNN and save the best model
"""
import os, sys, json, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------------- Config ----------------
SEED = 42
TRAIN_DIR = os.path.join("PlantVillage_15", "train")
VAL_DIR = os.path.join("PlantVillage_15", "val")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
MAX_EPOCHS = 30
MODEL_DIR = "model"
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(MODEL_DIR, "plant_cnn.keras")

# ---------------- Reproducibility ----------------
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ---------------- Data (existing split, no re-splitting) ----------------
def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, labels="inferred", label_mode="categorical",
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, labels="inferred", label_mode="categorical",
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    class_names = train_ds.class_names
    if class_names != val_ds.class_names:
        raise ValueError("Train and val class folders do not match!")
    AUTOTUNE = tf.data.AUTOTUNE
    return (train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE), class_names)


def compute_class_weights(class_names):
    """Class weights from the real training image counts (handles imbalance)."""
    counts = []
    for c in class_names:
        folder = os.path.join(TRAIN_DIR, c)
        counts.append(len([f for f in os.listdir(folder)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))]))
    total, n = sum(counts), len(counts)
    return {i: total / (n * cnt) for i, cnt in enumerate(counts)}, counts


# ---------------- Model ----------------
def build_model(num_classes):
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ], name="augmentation")  # active only during training

    def block(x, filters, drop):
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        return layers.Dropout(drop)(x)

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)          # normalization
    x = block(x, 32, 0.1)
    x = block(x, 64, 0.2)
    x = block(x, 128, 0.25)
    x = block(x, 256, 0.3)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="plant_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ---------------- Main ----------------
def main():
    do_train = "--train" in sys.argv
    train_ds, val_ds, class_names = load_datasets()
    class_weights, counts = compute_class_weights(class_names)

    print("\nClasses (index order):")
    for i, (c, n) in enumerate(zip(class_names, counts)):
        print(f"  {i}: {c}  (train images: {n}, weight: {class_weights[i]:.2f})")

    model = build_model(len(class_names))
    model.summary()

    if not do_train:
        print("\nInspect-only mode. Run `python train.py --train` to train.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss",
                                  save_best_only=True, verbose=1),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS,
                        class_weight=class_weights, callbacks=cbs)

    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    print(f"\nDone. Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()