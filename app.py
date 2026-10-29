
import json
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "best_plant_disease_model.keras"
LABELS_PATH = "class_names.json"
IMG_SIZE = 160

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as file:
    class_names = json.load(file)


def predict_disease(image):
    if image is None:
        return {"Please upload a leaf image": 1.0}

    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[-5:][::-1]

    results = {}
    for index in top_indices:
        label = class_names[index].replace("___", " - ").replace("_", " ")
        results[label] = float(predictions[index])

    return results


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌿 Plant Disease Detection AI")
    gr.Markdown(
        "Upload a picture of a plant leaf to predict its disease category. "
        "The model recognizes 38 PlantVillage categories. "
        "Predictions are estimates, not confirmed diagnoses."
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload a plant leaf")
        output = gr.Label(num_top_classes=5, label="Disease predictions")

    predict_button = gr.Button("Detect Plant Disease", variant="primary")
    predict_button.click(
        fn=predict_disease,
        inputs=image_input,
        outputs=output
    )

    gr.Markdown(
        "Built with TensorFlow, MobileNetV2, and the PlantVillage dataset."
    )


demo.launch()
