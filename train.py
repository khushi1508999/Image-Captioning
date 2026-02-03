"""Training pipeline for image captioning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from model import IMAGE_SIZE, build_caption_model

START_TOKEN = "startseq"
END_TOKEN = "endseq"


def load_captions(csv_path: Path) -> Tuple[List[str], List[str]]:
    """Load image paths and captions from a CSV with columns: image, caption."""
    df = pd.read_csv(csv_path)
    if "image" not in df.columns or "caption" not in df.columns:
        raise ValueError("CSV must contain 'image' and 'caption' columns.")

    images = df["image"].astype(str).tolist()
    captions = df["caption"].astype(str).tolist()
    captions = [f"{START_TOKEN} {cap.strip()} {END_TOKEN}" for cap in captions]
    return images, captions


def build_tokenizer(captions: List[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
    tokenizer.fit_on_texts(captions)
    return tokenizer


def sequences_from_captions(
    tokenizer: Tokenizer, captions: List[str]
) -> Tuple[np.ndarray, np.ndarray, int]:
    sequences = tokenizer.texts_to_sequences(captions)
    max_length = max(len(seq) for seq in sequences)

    input_sequences = [seq[:-1] for seq in sequences]
    target_sequences = [seq[1:] for seq in sequences]

    input_sequences = pad_sequences(input_sequences, maxlen=max_length - 1, padding="post")
    target_sequences = pad_sequences(target_sequences, maxlen=max_length - 1, padding="post")
    return input_sequences, target_sequences, max_length


def preprocess_image(image_path: tf.Tensor) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image


def make_dataset(
    image_paths: List[str],
    input_sequences: np.ndarray,
    target_sequences: np.ndarray,
    batch_size: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, input_sequences, target_sequences)
    )

    def _map_fn(img_path: tf.Tensor, inp: tf.Tensor, target: tf.Tensor):
        image = preprocess_image(img_path)
        return (image, inp), target

    dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def masked_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def save_tokenizer(tokenizer: Tokenizer, path: Path) -> None:
    data = tokenizer.to_json()
    path.write_text(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an image captioning model.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to captions CSV.")
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    image_paths, captions = load_captions(args.csv)
    image_paths = [str(args.image_root / path) for path in image_paths]

    tokenizer = build_tokenizer(captions, args.vocab_size)
    input_sequences, target_sequences, max_length = sequences_from_captions(
        tokenizer, captions
    )

    model = build_caption_model(
        vocab_size=min(args.vocab_size, len(tokenizer.word_index) + 1),
        max_length=max_length,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=masked_loss)

    dataset = make_dataset(
        image_paths, input_sequences, target_sequences, args.batch_size
    )

    model.fit(dataset, epochs=args.epochs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "caption_model"
    model.save(model_path)
    save_tokenizer(tokenizer, args.output_dir / "tokenizer.json")

    metadata = {
        "max_length": max_length,
        "vocab_size": min(args.vocab_size, len(tokenizer.word_index) + 1),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()