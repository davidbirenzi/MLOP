import os

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

MODEL_PATH = os.path.join("models", "mobilenet_pathmnist.h5")


def _patch_dense_h5_compat():
    """
    Newer Keras saves Dense layers with `quantization_config` in the H5 config.
    Keras 3's Layer.from_config(config) rejects unknown keys — strip before delegating.
    """
    try:
        from keras.layers import Dense
    except ImportError:
        from tensorflow.keras.layers import Dense

    if getattr(Dense, "_mlop_h5_compat_patched", False):
        return

    _orig = Dense.from_config

    def from_config(config):
        cfg = dict(config)
        cfg.pop("quantization_config", None)
        return _orig(cfg)

    Dense.from_config = from_config
    Dense._mlop_h5_compat_patched = True


def load_production_model():
    """
    Loads the saved fine-tuned MobileNetV2 model for inference.
    """
    if not os.path.isfile(MODEL_PATH):
        print(f"No model file at {MODEL_PATH}. Train the model or add the .h5 file.")
        return None
    try:
        _patch_dense_h5_compat()
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(f"Successfully loaded production model from {MODEL_PATH}")
        return model
    except (OSError, ValueError, TypeError) as e:
        print(f"Failed to load model from {MODEL_PATH}: {e}")
        return None


def retrain_pipeline(new_train_ds, new_val_ds, epochs=5):
    """
    Fine-tunes the existing production model on new data (custom pre-trained workflow).
    Expects tf.data.Dataset batches of (images, sparse integer labels 0..8 for PathMNIST).
    """
    model = load_production_model()

    if model is None:
        raise RuntimeError("Cannot retrain: base production model was not found.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    print("Starting fine-tuning on uploaded / prepared data...")
    history = model.fit(
        new_train_ds,
        validation_data=new_val_ds,
        epochs=epochs,
        callbacks=[early_stop],
    )

    model.save(MODEL_PATH)
    print(f"Retraining complete. Weights saved to {MODEL_PATH}")
    return history
