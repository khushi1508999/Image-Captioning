# Image-Captioning
End-to-end image captioning example using TensorFlow, a ResNet50 encoder, and an LSTM decoder. The project includes data preprocessing, training, inference, and a Streamlit UI.

## Project Structure
- `model.py` - CNN encoder + LSTM decoder architecture.
- `train.py` - Data preprocessing and training pipeline.
- `inference.py` - Model loading and caption generation.
- `app.py` - Streamlit web application.

## Dataset Format
Provide a CSV file with two columns:
- `image`: relative path to the image file.
- `caption`: the caption text.

Example `captions.csv`:
```
image,caption
images/dog.jpg,a dog running in the park
images/cat.jpg,a cat sitting on a couch
```

## Training
```
python train.py --csv captions.csv --image-root . --epochs 5
```
Artifacts are saved to `artifacts/`:
- `caption_model/` (SavedModel)
- `tokenizer.json`
- `metadata.json`

## Streamlit App
```
streamlit run app.py
```
Upload an image and the model will generate a caption.
