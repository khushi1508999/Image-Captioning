"""Inference utilities for image captioning."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from model import IMAGE_SIZE

START_TOKEN = "startseq"
END_TOKEN = "endseq"


def load_artifacts(artifacts_dir: Path) -> Tuple[tf.keras.Model, dict, tf.keras.preprocessing.text.Tokenizer]:
    model = tf.keras.models.load_model(artifacts_dir / "caption_model")
    metadata = json.loads((artifacts_dir / "metadata.json").read_text())
    tokenizer = tokenizer_from_json((artifacts_dir / "tokenizer.json").read_text())
    return model, metadata, tokenizer


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = tf.image.decode_image(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image.numpy()


def generate_caption(
    model: tf.keras.Model,
    tokenizer: tf.keras.preprocessing.text.Tokenizer,
    image: np.ndarray,
    max_length: int,
) -> str:
    word_index = tokenizer.word_index
    index_word = {idx: word for word, idx in word_index.items()}

    caption = [word_index.get(START_TOKEN)]
    for _ in range(max_length - 1):
        input_seq = pad_sequences([caption], maxlen=max_length - 1, padding="post")
        preds = model.predict([image[None, ...], input_seq], verbose=0)
        next_id = int(np.argmax(preds[0, len(caption) - 1, :]))
        next_word = index_word.get(next_id, "")
        if next_word == END_TOKEN or next_word == "":
            break
        caption.append(next_id)

    words = [index_word.get(idx, "") for idx in caption[1:]]
    return " ".join(word for word in words if word)