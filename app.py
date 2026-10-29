
import os
import json
import logging
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

# -------------------------------
# 1. Project configuration
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "best_plant_disease_model.keras"
)

LABELS_PATH = os.path.join(
    BASE_DIR, "class_names.json"
)

IMG_SIZE = 160

# -------------------------------
# 2. Load the trained AI model
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Upload best_plant_disease_model.keras "
        "to the root of your GitHub repository."
    )

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(
        f"Class labels file not found: {LABELS_PATH}. "
        "Upload class_names.json to the repository."
    )

logger.info("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

if not isinstance(class_names, list) or len(class_names) != 38:
    raise ValueError(
        "class_names.json must contain exactly 38 class names."
    )

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        "The number of model outputs does not match "
        "the number of class names."
    )

logger.info("Model and 38 class names loaded successfully.")

# -------------------------------
# 3. Predict plant disease
# -------------------------------
def predict_disease(image):
    if image is None:
        return {"Please upload a leaf image": 1.0}

    try:
        # Convert image to RGB and resize it
        image = image.convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        # Convert image into a numerical array
        image_array = np.array(image, dtype=np.float32)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Predict using the trained model
        # The model already contains its Rescaling layer.
        predictions = model.predict(
            image_array, verbose=0
        )[0]

        # Select the five highest-scoring categories
        top_indices = np.argsort(predictions)[-5:][::-1]

        results = {}

        for index in top_indices:
            label = class_names[int(index)]
            confidence = float(predictions[index])
            results[label.replace("___", " - ")] = confidence

        return results

    except Exception:
        logger.exception("Prediction failed")
        raise gr.Error(
            "Could not process this image. "
            "Please upload a clear plant leaf photo."
        )

# -------------------------------
# 4. Create the web interface
# -------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🌿 Plant Disease Detection AI

        Upload a plant leaf image to predict its likely
        disease category using a trained deep-learning model.

        **Dataset:** PlantVillage  
        **Model:** MobileNetV2 transfer learning  
        **Categories:** 38
        """
    )

    gr.Markdown(
        """
        **Note:** Predictions are model estimates, not
        confirmed diagnoses. Accuracy may vary with lighting,
        image quality, and plant varieties.
        """
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

    with gr.Row():
        predict_button = gr.Button(
            "Detect Plant Disease",
            variant="primary"
        )
        clear_button = gr.ClearButton(
            [image_input, prediction_output],
            value="Clear"
        )

    predict_button.click(
        fn=predict_disease,
        inputs=image_input,
        outputs=prediction_output
    )

    gr.Markdown(
        """
        ---
        Developed as an AI/ML academic project using
        Python, TensorFlow, MobileNetV2, and Gradio.
        """
    )

# -------------------------------
# 5. Start the app on Render
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
