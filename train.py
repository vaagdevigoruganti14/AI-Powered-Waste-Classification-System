import os
import json
import random
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf


DATASETS_ROOT = os.path.join("data sets")
TRASHNET_DIR = os.path.join(DATASETS_ROOT, "trashnet")
ORGANIC_DIR = os.path.join(DATASETS_ROOT, "organic waste")
E_WASTE_DIR = os.path.join(DATASETS_ROOT, "ewaste")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


def is_image_file(filename: str) -> bool:
    lower = filename.lower()
    return lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def list_images_recursive(root: str) -> List[str]:
    files: List[str] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if is_image_file(fn):
                files.append(os.path.join(dp, fn))
    return files


def build_file_label_lists() -> Tuple[List[str], List[str]]:
    # Map folders → 3 super-classes
    # recyclable: all images inside trashnet/* subfolders
    recyclable_paths: List[str] = list_images_recursive(TRASHNET_DIR) if os.path.isdir(TRASHNET_DIR) else []

    # organic: all images inside organic waste/* subfolders
    organic_paths: List[str] = list_images_recursive(ORGANIC_DIR) if os.path.isdir(ORGANIC_DIR) else []

    # hazardous: all images under ewaste (common Kaggle set)
    hazardous_paths: List[str] = list_images_recursive(E_WASTE_DIR) if os.path.isdir(E_WASTE_DIR) else []

    x_paths: List[str] = []
    y_labels: List[str] = []

    for p in recyclable_paths:
        x_paths.append(p)
        y_labels.append("recyclable")
    for p in organic_paths:
        x_paths.append(p)
        y_labels.append("organic")
    for p in hazardous_paths:
        x_paths.append(p)
        y_labels.append("hazardous")

    if len(x_paths) == 0:
        raise RuntimeError(
            "No images found. Make sure 'data sets/trashnet', 'data sets/organic waste', and 'data sets/ewaste' exist."
        )

    # keep deterministic split
    rng = random.Random(SEED)
    indices = list(range(len(x_paths)))
    rng.shuffle(indices)
    x_paths = [x_paths[i] for i in indices]
    y_labels = [y_labels[i] for i in indices]

    return x_paths, y_labels


def preprocess_image(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img


def make_dataset(filepaths: List[str], labels: List[int], augment: bool) -> tf.data.Dataset:
    paths_ds = tf.data.Dataset.from_tensor_slices(filepaths)
    imgs_ds = paths_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int32))
    ds = tf.data.Dataset.zip((imgs_ds, labels_ds))

    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.shuffle(1024, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    x_paths, y_labels_str = build_file_label_lists()

    class_names = sorted(list({c for c in y_labels_str}))
    class_to_index = {c: i for i, c in enumerate(class_names)}
    y_indices = [class_to_index[c] for c in y_labels_str]

    x_train, x_val, y_train, y_val = train_test_split(
        x_paths, y_indices, test_size=0.2, random_state=SEED, stratify=y_indices
    )

    train_ds = make_dataset(x_train, y_train, augment=True)
    val_ds = make_dataset(x_val, y_val, augment=False)

    model = build_model(num_classes=len(class_names))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="waste_classifier_cnn.h5", monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # Save final model and class indices
    model.save("waste_classifier_cnn.h5")
    with open("class_indices.json", "w", encoding="utf-8") as f:
        json.dump(class_to_index, f, indent=2)

    # quick evaluation
    eval_res = model.evaluate(val_ds, verbose=0)
    print({"val_loss": float(eval_res[0]), "val_accuracy": float(eval_res[1])})


if __name__ == "__main__":
    main()


