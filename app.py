
import os

# ==========================================
# 1. TENSORFLOW CPU CONFIGURATION
# Must be set before importing TensorFlow
# ==========================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import json
import csv
import logging
import tempfile
import time
from datetime import datetime

import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

# ==========================================
# 2. PROJECT CONFIGURATION
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "best_plant_disease_model.keras"
)

LABELS_PATH = os.path.join(
    BASE_DIR, "class_names.json"
)

EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")

IMG_SIZE = 160
NUM_CLASSES = 38

# Disable automatic XLA/JIT compilation.
tf.config.optimizer.set_jit(False)

# Limit CPU threads for the Render free instance.
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    logger.warning("TensorFlow threading was already initialized.")

# ==========================================
# 3. LOAD TRAINED MODEL AND CLASS LABELS
# ==========================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(
        f"Class labels file not found: {LABELS_PATH}"
    )

logger.info("Loading Plant Disease Detection model...")

load_start = time.perf_counter()

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

if not isinstance(class_names, list) or len(class_names) != NUM_CLASSES:
    raise ValueError(
        "class_names.json must contain exactly 38 class names."
    )

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        "Model output count does not match class labels."
    )

logger.info(
    "Model loaded successfully in %.2f seconds.",
    time.perf_counter() - load_start
)

# ==========================================
# 4. PREDICT PLANT DISEASE
# ==========================================

def predict_disease(image):

    if image is None:
        return (
            {},
            None,
            "Please upload a plant leaf image first."
        )

    report_path = None

    try:
        logger.info("Prediction request received.")

        prediction_start = time.perf_counter()

        # Convert and resize the uploaded image.
        image = image.convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        logger.info("Image preprocessing completed.")

        # Direct eager inference instead of model.predict().
        # This avoids the extra Keras predict data pipeline.
        inference_start = time.perf_counter()

        prediction_tensor = model(
            image_array,
            training=False
        )

        predictions = prediction_tensor.numpy()[0]

        logger.info(
            "Inference completed in %.2f seconds.",
            time.perf_counter() - inference_start
        )

        # Select the five highest-scoring categories.
        top_indices = np.argsort(
            predictions
        )[-5:][::-1]

        results = {}

        for index in top_indices:
            label = class_names[int(index)]
            confidence = float(predictions[index])

            results[
                label.replace("___", " - ")
            ] = confidence

        # ==================================
        # GENERATE DOWNLOADABLE CSV REPORT
        # ==================================

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix="plant_disease_report_",
            delete=False,
            newline="",
            encoding="utf-8"
        ) as report_file:

            report_path = report_file.name

            writer = csv.writer(report_file)

            writer.writerow([
                "Plant Disease Detection Report"
            ])

            writer.writerow([
                "Analysis Date",
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ])

            writer.writerow([
                "Model",
                "MobileNetV2 transfer learning"
            ])

            writer.writerow([
                "Image Input Size",
                f"{IMG_SIZE} x {IMG_SIZE}"
            ])

            writer.writerow([])

            writer.writerow([
                "Rank",
                "Predicted Class",
                "Model Score (%)"
            ])

            for rank, index in enumerate(
                top_indices,
                start=1
            ):

                label = class_names[
                    int(index)
                ].replace("___", " - ")

                score = float(
                    predictions[index]
                ) * 100

                writer.writerow([
                    rank,
                    label,
                    f"{score:.2f}"
                ])

        # Main prediction.
        top_label = class_names[
            int(top_indices[0])
        ].replace("___", " - ")

        top_score = float(
            predictions[top_indices[0]]
        ) * 100

        total_time = (
            time.perf_counter() - prediction_start
        )

        status = (
            "### Analysis complete\n"
            f"**Top prediction:** {top_label}\n\n"
            f"**Model score:** {top_score:.2f}%\n\n"
            f"**Processing time:** {total_time:.2f} seconds\n\n"
            "Download the CSV report for all five predictions."
        )

        logger.info(
            "Prediction completed successfully in %.2f seconds.",
            total_time
        )

        return results, report_path, status

    except Exception as error:

        logger.exception("Prediction failed.")

        # Remove an incomplete report if one was created.
        if report_path and os.path.exists(report_path):
            try:
                os.remove(report_path)
            except OSError:
                pass

        raise gr.Error(
            f"Image analysis failed: {str(error)}"
        )


# ==========================================
# 5. RESET FUNCTION
# ==========================================

def reset_app():

    return (
        None,
        {},
        None,
        "Upload a leaf image to begin a new analysis."
    )


# ==========================================
# 6. FOREST-GREEN WEBSITE DESIGN
# ==========================================

custom_css = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}

#main-header {
    text-align: center;
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #123c2b, #26734d);
    color: white;
    margin-bottom: 18px;
}

#main-header h1 {
    color: white !important;
    font-size: 34px !important;
}

.info-card {
    border: 1px solid #39845a;
    border-radius: 14px;
    padding: 15px;
    text-align: center;
}

#disclaimer {
    font-size: 13px;
    opacity: 0.85;
}
"""

theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="lime",
    neutral_hue="slate"
)

# ==========================================
# 7. CREATE GRADIO INTERFACE
# ==========================================

with gr.Blocks(
    title="PlantGuard AI - Plant Disease Detection"
) as demo:

    # Header
    gr.HTML("""
        <div id="main-header">
            <h1>🌿 PlantGuard AI</h1>
            <p>AI-powered Plant Disease Detection System</p>
            <p>Upload a leaf image and discover its likely disease category.</p>
        </div>
    """)

    # Project information
    gr.Markdown("## About This Project")

    with gr.Row():

        with gr.Column():
            gr.Markdown("""
            <div class="info-card">
                <h3>🌱 Dataset</h3>
                <h2>PlantVillage</h2>
                <p>54,303 images in the original dataset</p>
            </div>
            """)

        with gr.Column():
            gr.Markdown("""
            <div class="info-card">
                <h3>🧠 AI Model</h3>
                <h2>MobileNetV2</h2>
                <p>Transfer learning with TensorFlow/Keras</p>
            </div>
            """)

        with gr.Column():
            gr.Markdown("""
            <div class="info-card">
                <h3>📊 Categories</h3>
                <h2>38 classes</h2>
                <p>Plant species and disease categories</p>
            </div>
            """)

    gr.Markdown("""
    ### How to use

    1. Upload a clear image of a plant leaf.
    2. Click **Analyze Leaf**.
    3. Review the five highest-scoring predictions.
    4. Download the CSV report if needed.
    """)

    # Main application
    with gr.Row():

        with gr.Column(scale=1):

            image_input = gr.Image(
                type="pil",
                label="📷 Upload Plant Leaf",
                height=340
            )

            with gr.Row():

                predict_button = gr.Button(
                    "🔍 Analyze Leaf",
                    variant="primary"
                )

                clear_button = gr.Button(
                    "🔄 Reset",
                    variant="secondary"
                )

        with gr.Column(scale=1):

            prediction_output = gr.Label(
                num_top_classes=5,
                label="🏆 Top 5 Predictions"
            )

            status_output = gr.Markdown(
                "Upload a leaf image to begin a new analysis."
            )

            report_output = gr.File(
                label="📄 Download Analysis Report"
            )

    # ==================================
    # EXAMPLE LEAF IMAGES
    # ==================================

    gr.Markdown("---")
    gr.Markdown("## 🖼️ Example Leaf Images")

    gr.Markdown(
        "Optional: add JPG, JPEG, PNG, or WEBP leaf images "
        "to the repository's `examples` folder to display them here."
    )

    if os.path.isdir(EXAMPLES_DIR):

        example_files = [
            os.path.join(EXAMPLES_DIR, filename)
            for filename in sorted(
                os.listdir(EXAMPLES_DIR)
            )
            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            )
        ][:6]

        if example_files:

            gr.Examples(
                examples=example_files,
                inputs=image_input,
                label="Click an example to load it"
            )

        else:

            gr.Markdown(
                "The examples folder is empty. "
                "Upload some leaf images to display examples."
            )

    else:

        gr.Markdown(
            "Example gallery will appear after you add images "
            "to the `examples` folder in GitHub."
        )

    # ==================================
    # TECHNICAL DETAILS
    # ==================================

    gr.Markdown("---")
    gr.Markdown("## ⚙️ Technical Details")

    with gr.Accordion(
        "Model and dataset information",
        open=False
    ):

        gr.Markdown("""
        - **Dataset:** PlantVillage
        - **Original dataset size:** 54,303 images
        - **Number of classes:** 38
        - **Architecture:** MobileNetV2
        - **Framework:** TensorFlow and Keras
        - **Image input size:** 160 × 160 RGB
        - **Interface:** Gradio
        - **Deployment:** Render

        The application uses the previously trained model.
        Uploading a new image performs inference; it does not
        retrain the model.
        """)

    gr.Markdown("""
    <div id="disclaimer">
    <b>Important:</b> Predictions are estimates from a machine-learning
    model, not confirmed plant disease diagnoses. Results can vary
    with lighting, image quality, plant variety, and real-world conditions.
    </div>

    <br>

    <center>
    🌿 PlantGuard AI | Academic AI/ML Project
    </center>
    """)

    # ==================================
    # BUTTON ACTIONS
    # ==================================

    predict_button.click(
        fn=predict_disease,
        inputs=[image_input],
        outputs=[
            prediction_output,
            report_output,
            status_output
        ]
    )

    clear_button.click(
        fn=reset_app,
        inputs=[],
        outputs=[
            image_input,
            prediction_output,
            report_output,
            status_output
        ]
    )


# ==========================================
# 8. START APPLICATION ON RENDER
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", "10000")
    )

    demo.launch(
        theme=theme,
        css=custom_css,
        server_name="0.0.0.0",
        server_port=port
    )
