
import os
import json
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

# Configuration
IMG_SIZE = 160
MODEL_PATH = "best_plant_disease_model.keras"
LABELS_PATH = "class_names.json"

# Load the saved model and disease labels
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)


def predict_disease(image):
    if image is None:
        return {"Please upload a leaf image": 1.0}

    # Prepare image in the same format used during training
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict disease probabilities
    predictions = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[-5:][::-1]

    results = {}
    for index in top_indices:
        disease_name = (
            class_names[index]
            .replace("___", " - ")
            .replace("_", " ")
        )
        results[disease_name] = float(predictions[index])

    return results


# Build the web interface
with gr.Blocks(theme=gr.themes.Soft(), title="Plant Disease Detection AI") as demo:
    gr.Markdown("# 🌿 Plant Disease Detection AI")
    gr.Markdown(
        "Upload a plant leaf image to predict its disease category. "
        "This model recognizes 38 PlantVillage categories."
    )

    with gr.Row():
        image_input = gr.Image(
            type="pil",
            label="Upload a plant leaf image"
        )
        prediction_output = gr.Label(
            num_top_classes=5,
            label="Top disease predictions"
        )

    predict_button = gr.Button(
        "Detect Plant Disease",
        variant="primary"
    )

    predict_button.click(
        fn=predict_disease,
        inputs=image_input,
        outputs=prediction_output
    )

    gr.Markdown(
        "**Note:** Predictions are estimates, not confirmed diagnoses. "
        "Results may be less reliable for images outside the training dataset."
    )
    gr.Markdown(
        "Built using TensorFlow, MobileNetV2 and the PlantVillage dataset."
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000"))
    )
