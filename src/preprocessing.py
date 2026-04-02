import os
import tempfile
import zipfile
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 128

# PathMNIST: 9 classes — folder names should be "0" … "8" for retrain uploads
PATHMNIST_NUM_CLASSES = 9


def preprocess_single_image(image_array):
    """
    Preprocesses a single image for prediction.
    Expects a numpy array of shape (H, W, 3) with values 0-255.
    """
    img = tf.cast(image_array, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)
    return np.expand_dims(img.numpy(), axis=0)


def _mobilenet_map(img, label):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)
    return img, label


def preprocess_batch_dataset(dataset_generator, batch_size=16):
    """
    Maps preprocessing across a generator yielding (image, label) for training/retraining.
    """

    def _map_fn(img, label):
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_input(img)
        return img, label[0]

    ds = tf.data.Dataset.from_generator(
        lambda: dataset_generator,
        output_signature=(
            tf.TensorSpec(shape=(28, 28, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(1,), dtype=tf.int64),
        ),
    )
    return ds.map(_map_fn).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _has_image_file(folder: str) -> bool:
    image_ext = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    for name in os.listdir(folder):
        if os.path.splitext(name)[1].lower() in image_ext:
            return True
    return False


def find_pathmnist_class_root(extract_root: str) -> Optional[str]:
    """
    Finds a directory that contains subfolders named "0" … "8" (PathMNIST / rubric layout),
    each with at least one image file.
    """
    for root, dirs, _ in os.walk(extract_root):
        if not all(str(i) in dirs for i in range(PATHMNIST_NUM_CLASSES)):
            continue
        ok = True
        for i in range(PATHMNIST_NUM_CLASSES):
            p = os.path.join(root, str(i))
            if not _has_image_file(p):
                ok = False
                break
        if ok:
            return root
    return None


def build_tf_datasets_from_class_folders(
    directory: str,
    batch_size: int = 16,
    validation_split: float = 0.2,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """
    Builds train/val tf.data datasets from a directory with one subfolder per class.
    Subfolder names must be "0" … "8" (PathMNIST alignment).
    """
    count = 0
    for i in range(PATHMNIST_NUM_CLASSES):
        p = os.path.join(directory, str(i))
        if os.path.isdir(p):
            count += len(
                [
                    f
                    for f in os.listdir(p)
                    if os.path.splitext(f)[1].lower()
                    in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
                ]
            )

    class_names = [str(i) for i in range(PATHMNIST_NUM_CLASSES)]

    # Tiny uploads (e.g., 1 image per class) are common in demos and can fail
    # with strict train/validation splitting. Use one dataset for both.
    if count < 20:
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            class_names=class_names,
            seed=seed,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
        )
        ds = ds.map(_mobilenet_map, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE
        )
        return ds, ds, count

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        class_names=class_names,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        class_names=class_names,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
    )
    train_ds = train_ds.map(_mobilenet_map, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(_mobilenet_map, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, count


def extract_zip_to_folder(zip_bytes: bytes, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp.write(zip_bytes)
        zip_path = tmp.name
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
    finally:
        try:
            os.unlink(zip_path)
        except OSError:
            pass
    return dest_dir


def build_medmnist_retrain_datasets(
    n_samples: int = 512,
    batch_size: int = 16,
    validation_split: float = 0.2,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """
    Fallback: PathMNIST subset from MedMNIST for demonstration when user zip has no valid layout.
    """
    try:
        from medmnist import PathMNIST as MedPathMNIST
    except ImportError as e:
        raise RuntimeError(
            "medmnist is not installed in this runtime. "
            "Provide a zip with PathMNIST-style folders 0-8, or install full requirements."
        ) from e

    train_set = MedPathMNIST(split="train", download=True)
    n = min(n_samples, len(train_set))
    images = []
    labels = []
    for i in range(n):
        img, label = train_set[i]
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        images.append(arr)
        lab = np.asarray(label).reshape(-1)[0]
        labels.append(int(lab))
    x = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int64)
    split = int(len(y) * (1.0 - validation_split))
    ds_train = tf.data.Dataset.from_tensor_slices((x[:split], y[:split]))
    ds_val = tf.data.Dataset.from_tensor_slices((x[split:], y[split:]))

    def prep(img, lab):
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_input(img)
        return img, lab

    train_ds = (
        ds_train.map(prep, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        ds_val.map(prep, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, n
