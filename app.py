
import streamlit as st
import streamlit.components.v1 as components
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import os

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="AI Emergency Traffic Control",
    page_icon="🚑",
    layout="wide"
)

st.markdown("""
<style>
/* Futuristic emergency-control background */
.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(14, 116, 190, 0.22),
            transparent 35%
        ),
        radial-gradient(
            circle at 95% 30%,
            rgba(30, 64, 120, 0.20),
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #06111f 0%,
            #0b192b 50%,
            #071426 100%
        );
    color: #f1f5f9;
}

/* Remove the solid header background */
[data-testid="stHeader"] {
    background: transparent;
}

/* Improve spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Make headings brighter */
h1, h2, h3 {
    color: #f0f9ff !important;
    letter-spacing: 0.3px;
}

/* Give content panels a clean glass-like look */
div[data-testid="stFileUploader"],
div[data-testid="stAlert"],
div[data-testid="stExpander"] {
    border-radius: 12px;
}

/* Style buttons */
.stButton > button {
    border-radius: 9px;
    border: 1px solid #2586b8;
    background: linear-gradient(135deg, #075985, #0369a1);
    color: white;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #7dd3fc;
    color: white;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.25);
}

/* Make the footer less distracting */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.title("🚑 AI Emergency Traffic Control")
st.write(
    "An AI prototype that detects ambulances and activates "
    "a simulated roadside warning beacon."
)

# ---------------- CONFIGURATION ----------------

WORKFLOW_ID = "general-segmentation-api"
CONFIDENCE_THRESHOLD = 0.60

# Read the API key from Streamlit Secrets or an environment variable.
try:
    api_key = st.secrets.get("ROBOFLOW_API_KEY", "")
except Exception:
    api_key = ""

if not api_key:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")

if not api_key:
    st.error(
        "Roboflow API key not found. Add ROBOFLOW_API_KEY "
        "in Streamlit Settings → Secrets."
    )
    st.stop()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# ---------------- IMAGE UPLOAD ----------------

st.subheader("📷 Ambulance Detection")

uploaded_file = st.file_uploader(
    "Upload a road image containing an ambulance",
    type=["jpg", "jpeg", "png"]
)

ambulance_detected = False
best_confidence = 0.0
annotated_image = None
error_message = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    annotated_image = image.copy()

    st.image(image, caption="Uploaded road image", use_container_width=True)

    with st.spinner("AI is checking the image..."):
        try:
            # Ask the Roboflow workflow to look for ambulances.
            results = client.run_workflow(
                workspace_name="yong-han-lee",
                workflow_id=WORKFLOW_ID,
                images={"image": image},
                parameters={"classes": "Ambulance"}
            )

            # Extract predictions from the workflow response.
            predictions = []

            if isinstance(results, list) and len(results) > 0:
                first_result = results[0]

                if isinstance(first_result, dict):
                    prediction_data = first_result.get("predictions", {})

                    if isinstance(prediction_data, dict):
                        predictions = prediction_data.get("predictions", [])
                    elif isinstance(prediction_data, list):
                        predictions = prediction_data

            # Only accept the class Ambulance.
            draw = ImageDraw.Draw(annotated_image)

            for prediction in predictions:
                if not isinstance(prediction, dict):
                    continue

                class_name = str(
                    prediction.get("class", prediction.get("class_name", ""))
                ).strip().lower()

                confidence = float(prediction.get("confidence", 0))

                # Ignore cars, stopped vehicles and all other classes.
                if class_name != "ambulance":
                    continue

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                ambulance_detected = True
                best_confidence = max(best_confidence, confidence)

                # Draw a box if the model supplies bounding-box coordinates.
                if all(
                    key in prediction
                    for key in ["x", "y", "width", "height"]
                ):
                    x = float(prediction["x"])
                    y = float(prediction["y"])
                    w = float(prediction["width"])
                    h = float(prediction["height"])

                    left = x - w / 2
                    top = y - h / 2
                    right = x + w / 2
                    bottom = y + h / 2

                    draw.rectangle(
                        [left, top, right, bottom],
                        outline="lime",
                        width=4
                    )
                    draw.text(
                        (left, max(0, top - 22)),
                        f"AMBULANCE {confidence:.0%}",
                        fill="lime",
                        stroke_width=2,
                        stroke_fill="black"
                    )

        except Exception as exc:
            error_message = (
                f"{type(exc).__name__}: {exc}"
            )

    if error_message:
        st.error("The AI could not process this image.")
        st.caption(error_message)

    elif ambulance_detected:
        st.success(
            f"🚑 Ambulance detected! Confidence: {best_confidence:.1%}"
        )
        st.image(
            annotated_image,
            caption="AI detection result",
            use_container_width=True
        )

    else:
        st.info(
            "No ambulance detected above the 60% confidence threshold. "
            "The warning light remains OFF."
        )

# ---------------- ROADSIDE WARNING LIGHT ----------------

st.divider()
st.subheader("🚨 Physical Roadside Warning-Light System")

st.write(
    "When the AI detects an ambulance, a roadside amber beacon "
    "is simulated as a warning to approaching drivers."
)

if ambulance_detected:
    light_colour = "#ffb000"
    animation = "blink 0.7s infinite"
    status_text = "AMBULANCE DETECTED — WARNING LIGHT ACTIVE"
    status_colour = "#ffcf40"
else:
    light_colour = "#555b65"
    animation = "none"
    status_text = "STANDBY — WARNING LIGHT OFF"
    status_colour = "#d1d5db"

components.html(
    f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: transparent;
        }}

        .scene {{
            background: #252d38;
            color: white;
            border-radius: 14px;
            padding: 24px;
            text-align: center;
        }}

        .heading {{
            font-size: 18px;
            font-weight: bold;
        }}

        .beacon {{
            width: 48px;
            height: 48px;
            margin: 22px auto 0;
            border-radius: 50%;
            background: {light_colour};
            border: 3px solid #fff0bd;
            box-shadow: 0 0 24px {light_colour};
            animation: {animation};
        }}

        .post {{
            width: 12px;
            height: 65px;
            margin: auto;
            background: #cbd5e1;
        }}

        .base {{
            width: 72px;
            height: 12px;
            margin: auto;
            border-radius: 3px;
            background: #94a3b8;
        }}

        .road {{
            margin-top: 24px;
            padding: 15px;
            border-top: 3px dashed white;
            font-size: 28px;
        }}

        .status {{
            margin-top: 16px;
            color: {status_colour};
            font-size: 16px;
            font-weight: bold;
        }}

        .note {{
            margin-top: 12px;
            color: #cbd5e1;
            font-size: 12px;
        }}

        @keyframes blink {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.2; }}
        }}
    </style>
    </head>

    <body>
        <div class="scene">
            <div class="heading">ROADSIDE AMBER WARNING UNIT</div>
            <div class="beacon"></div>
            <div class="post"></div>
            <div class="base"></div>
            <div class="road">🚗　🚙　🚕</div>
            <div class="status">{status_text}</div>
            <div class="note">SIMULATED DEVICE — NOT A REAL TRAFFIC SIGNAL</div>
        </div>
    </body>
    </html>
    """,
    height=390,
    scrolling=False
)

st.caption(
    "This is a visual simulation only. A real implementation would "
    "require a roadside sensor or camera, a controller, and a physical "
    "amber beacon."
)

st.divider()
st.caption(
    "AI Emergency Traffic Control · Hackathon prototype · "
    "Image-based inference only"
)
