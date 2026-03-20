"""
Training script for DermaDetect skin cancer classifier.

Trains EfficientNetB0 (main model) or baseline models for comparison.
EfficientNetB0 uses optimised hyperparameters (oversampling, focal loss,
stronger class weights, deeper fine-tuning) for best accuracy.

Usage:
    python train_model.py                          # EfficientNetB0 (default)
    python train_model.py --model vgg16
    python train_model.py --model resnet50
    python train_model.py --model mobilenetv2
    python train_model.py --model inceptionv3
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from PIL import Image3

# ── Speed: mixed precision (2-3x faster on Apple Silicon / GPU) ───────────────
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 128  # 24GB unified memory can handle this — halves steps vs 64
EPOCHS_FROZEN   = 10   # head-only phase (early stopping handles over-training)
EPOCHS_FINETUNE = 50   # fine-tune top layers (early stopping guards)

# HAM10000
HAM_METADATA_CSV = "HAM10000_metadata.csv"
HAM_IMAGE_DIRS   = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
HAM_LABEL_MAP    = {
    "akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "nv": 4, "vasc": 5, "mel": 6,
}

# ISIC 2019
ISIC_METADATA_CSV = "ISIC_2019_Training_GroundTruth.csv"
ISIC_IMAGE_DIR    = "ISIC_2019_Training_Input"
ISIC_LABEL_MAP    = {
    "MEL": 6, "NV": 4, "BCC": 1, "AK": 0, "BKL": 2, "DF": 3, "VASC": 5, "SCC": 0,
}

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "efficientnetb0": {
        "base_fn":    lambda: tf.keras.applications.EfficientNetB0(
                          weights="imagenet", include_top=False,
                          input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        "preprocess":  None,   # EfficientNetB0 includes preprocessing internally
        "save_path":   "best_model_efficientnet.keras",
        "unfreeze":    60,     # deeper fine-tuning (was 30)
    },
    "vgg16": {
        "base_fn":    lambda: tf.keras.applications.VGG16(
                          weights="imagenet", include_top=False,
                          input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        "preprocess":  tf.keras.applications.vgg16.preprocess_input,
        "save_path":   "best_model_vgg16.keras",
        "unfreeze":    8,
    },
    "resnet50": {
        "base_fn":    lambda: tf.keras.applications.ResNet50(
                          weights="imagenet", include_top=False,
                          input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        "preprocess":  tf.keras.applications.resnet50.preprocess_input,
        "save_path":   "best_model_resnet50.keras",
        "unfreeze":    30,
    },
    "mobilenetv2": {
        "base_fn":    lambda: tf.keras.applications.MobileNetV2(
                          weights="imagenet", include_top=False,
                          input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        "preprocess":  tf.keras.applications.mobilenet_v2.preprocess_input,
        "save_path":   "best_model_mobilenetv2.keras",
        "unfreeze":    30,
    },
    "inceptionv3": {
        "base_fn":    lambda: tf.keras.applications.InceptionV3(
                          weights="imagenet", include_top=False,
                          input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        "preprocess":  tf.keras.applications.inception_v3.preprocess_input,
        "save_path":   "best_model_inceptionv3.keras",
        "unfreeze":    30,
    },
}
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


def load_dataset():
    frames = []
    if os.path.exists(HAM_METADATA_CSV):
        ham = load_ham10000()
        print(f"  HAM10000: {len(ham)} samples")
        frames.append(ham)
    if os.path.exists(ISIC_METADATA_CSV):
        isic = load_isic2019()
        print(f"  ISIC 2019: {len(isic)} samples")
        frames.append(isic)
    if not frames:
        raise FileNotFoundError("No dataset CSVs found.")
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].astype(int)
    return df


def oversample_minorities(df, target_labels, multiplier=2):
    """Duplicate rows for minority classes to reduce class imbalance."""
    minority = df[df["label"].isin(target_labels)]
    extras = pd.concat([minority] * multiplier, ignore_index=True)
    return pd.concat([df, extras], ignore_index=True).sample(frac=1, random_state=42)


def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32)


def make_tf_dataset(paths, labels, preprocess_fn=None, augment=False):
    def load(path, label):
        img = tf.numpy_function(preprocess_image, [path], tf.float32)
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)

    if preprocess_fn is not None:
        ds = ds.map(
            lambda x, y: (preprocess_fn(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    if augment:
        augmenter = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.3),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ])
        ds = ds.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def focal_loss(gamma=2.0, alpha=0.25):
    """Categorical focal loss — better than cross-entropy for class imbalance."""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=7)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -y_true_one_hot * tf.math.log(y_pred)
        pt = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1, keepdims=True)
        focal_weight = alpha * tf.pow(1.0 - pt, gamma)
        loss = focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return loss_fn


def build_model(base_fn, trainable_base=False):
    base = base_fn()
    base.trainable = trainable_base
    x      = GlobalAveragePooling2D()(base.output)
    x      = Dense(256, activation="relu")(x)
    x      = BatchNormalization()(x)
    x      = Dropout(0.4)(x)
    x      = Dense(128, activation="relu")(x)
    x      = Dropout(0.3)(x)
    # Cast to float32 before softmax — required for mixed precision stability
    x      = tf.keras.layers.Activation("linear", dtype="float32")(x)
    output = Dense(7, activation="softmax", dtype="float32")(x)
    return Model(inputs=base.input, outputs=output)


def main():
    parser = argparse.ArgumentParser(description="Train a skin cancer classifier.")
    parser.add_argument(
        "--model", default="efficientnetb0",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to train (default: efficientnetb0)"
    )
    args = parser.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    print(f"\n{'='*60}")
    print(f"  Training: {args.model.upper()}")
    print(f"  Save path: {cfg['save_path']}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading dataset (HAM10000 + ISIC 2019)...")
    df = load_dataset()
    print(f"  Total before oversampling: {len(df)} samples")
    print(f"  Class distribution:\n{df['label'].value_counts().sort_index()}\n")

    # Oversample minority classes: Melanoma(6), AK(0), DF(3), Pyogenic(5)
    df = oversample_minorities(df, target_labels=[0, 3, 5, 6], multiplier=2)
    print(f"  Total after oversampling: {len(df)} samples")
    print(f"  Class distribution after oversampling:\n{df['label'].value_counts().sort_index()}\n")

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"].values,
    )
    cw = dict(enumerate(class_weights))
    cw[6] *= 4.5  # Melanoma — most dangerous miss, highest priority
    cw[0] *= 2.5  # Actinic Keratosis — pre-malignant, was low recall
    cw[3] *= 2.0  # Dermatofibroma — very small support
    cw[5] *= 2.0  # Pyogenic Granuloma — small support
    print(f"Class weights: {cw}\n")

    preprocess_fn = cfg["preprocess"]
    train_ds = make_tf_dataset(
        train_df["path"].values, train_df["label"].values.astype(np.int32),
        preprocess_fn=preprocess_fn, augment=True
    )
    val_ds = make_tf_dataset(
        val_df["path"].values, val_df["label"].values.astype(np.int32),
        preprocess_fn=preprocess_fn
    )

    callbacks = [
        ModelCheckpoint(cfg["save_path"], save_best_only=True,
                        monitor="val_accuracy", mode="max", verbose=1),
        EarlyStopping(patience=8, restore_best_weights=True,
                      monitor="val_accuracy", mode="max"),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_accuracy",
                          mode="max", verbose=1),
    ]

    # ── Phase 1: Train head only ──────────────────────────────────────────────
    print("Phase 1: Training classification head (base frozen)...")
    model = build_model(cfg["base_fn"], trainable_base=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_FROZEN, class_weight=cw, callbacks=callbacks,
    )

    # ── Phase 2: Fine-tune top layers ─────────────────────────────────────────
    print(f"\nPhase 2: Fine-tuning top {cfg['unfreeze']} layers...")
    for layer in model.layers[-cfg["unfreeze"]:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),  # lower LR for stable fine-tuning
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_FINETUNE, class_weight=cw, callbacks=callbacks,
    )

    print(f"\nTraining complete. Best model saved to: {cfg['save_path']}")


if __name__ == "__main__":
    main()
