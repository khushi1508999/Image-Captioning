"""Streamlit app for image captioning."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from inference import generate_caption, load_artifacts, preprocess_image


ARTIFACTS_DIR = Path("artifacts")


def main() -> None:
    st.set_page_config(page_title="Image Captioning", page_icon="🖼️")
    st.title("🖼️ Image Captioning")
    st.write("Upload an image and let the model generate a caption.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload an image to get started.")
        return

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if not (ARTIFACTS_DIR / "caption_model").exists():
        st.warning("Model artifacts not found. Train the model first.")
        return

    model, metadata, tokenizer = load_artifacts(ARTIFACTS_DIR)
    image_array = preprocess_image(uploaded_file.read())
    caption = generate_caption(
        model,
        tokenizer,
        image_array,
        max_length=metadata["max_length"],
    )

    st.subheader("Generated Caption")
    st.write(caption or "No caption generated.")


if __name__ == "__main__":
    main()