import os
import streamlit as st
from PIL import Image
from inference_sdk import InferenceHTTPClient

st.set_page_config(
    page_title="AI Emergency Traffic Control",
    page_icon="🚑",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background: #f3f6fb; }
.hero {
    background: linear-gradient(120deg, #102a43, #176b87);
    color: white; padding: 1.6rem 2rem; border-radius: 18px;
    margin-bottom: 1rem;
}
.hero h1 { margin: 0; font-size: 2rem; }
.hero p { margin: .4rem 0 0 0; color: #e1f4ff; }
.panel {
    background: white; padding: 1.2rem; border-radius: 16px;
    border: 1px solid #dce5ef; min-height: 150px;
}
.small { color: #526579; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🚑 AI Emergency Traffic Control</h1>
  <p>Smart-city concept prototype · Upload an image to test ambulance detection</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About this prototype")
    st.write("This demo analyzes an uploaded image using a Roboflow workflow.")
    st.info("Prototype only: it does not analyze live traffic or control real traffic signals.")
    st.caption("Keep your Roboflow API key in Streamlit Secrets or an environment variable. Never paste it into public code.")

api_key = st.secrets.get("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", ""))
uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "webp"])

col_image, col_status = st.columns([1.05, 1])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    with col_image:
        st.subheader("Uploaded road image")
        st.image(image, use_container_width=True)

    with col_status:
        st.subheader("Detection results")
        if not api_key:
            st.error("API key not configured.")
            st.markdown("""
            To run the real AI model, add `ROBOFLOW_API_KEY` to your app's
            **Streamlit Secrets**. See `README.md` for setup instructions.
            """)
        elif st.button("Run ambulance detection", type="primary", use_container_width=True):
            with st.spinner("Analyzing image with the AI model..."):
                try:
                    client = InferenceHTTPClient(
                            api_url="https://serverless.roboflow.com",
                            api_key=api_key
                            )

                    # Save uploaded bytes to a temporary file for the SDK.
                    import tempfile
                    suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.getvalue())
                        image_path = tmp.name

                    try:
                        result = client.run_workflow(
                            workspace_name="yong-han-lee",
                            workflow_id="general-segmentation-api",
                            images={"image": image_path},
                            parameters={"classes": "Ambulance"},
                            use_cache=True,
                        )
                    finally:
                        try:
                            os.remove(image_path)
                        except OSError:
                            pass

                    predictions = result[0]["predictions"]["predictions"]
                    ambulance_predictions = [
                        item for item in predictions
                        if item.get("class") == "Ambulance"
                    ]

                    if ambulance_predictions:
                        best = max(ambulance_predictions, key=lambda item: item.get("confidence", 0))
                        confidence = float(best.get("confidence", 0)) * 100
                        detected = confidence >= 60
                    else:
                        confidence = 0.0
                        detected = False

                    st.session_state["detection_result"] = {
                        "detected": detected,
                        "confidence": confidence,
                    }
                except Exception as exc:
                        st.error("Detection failed.")
                        st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
                
        outcome = st.session_state.get("detection_result")
        if outcome:
            detected = outcome["detected"]
            confidence = outcome["confidence"]
            if detected:
                st.success("🚑 AMBULANCE DETECTED")
                st.metric("Detection confidence", f"{confidence:.1f}%")
                signal_color = "#20bf6b"
                signal_text = "GREEN — emergency priority (simulated)"
                warning_text = "GIVE WAY TO EMERGENCY VEHICLE"
            else:
                st.warning("No ambulance detected at the confidence threshold.")
                st.metric("Detection confidence", f"{confidence:.1f}%")
                signal_color = "#ef4444"
                signal_text = "RED — normal operation (simulated)"
                warning_text = "NO EMERGENCY DETECTED"

            st.markdown(f"""
            <div class="panel">
              <h3>🚦 Simulated traffic signal</h3>
              <div style="display:flex;align-items:center;gap:16px;">
                <div style="background:#202833;padding:10px 14px;border-radius:24px;display:flex;gap:9px;">
                  <span style="width:20px;height:20px;border-radius:50%;background:{'#ef4444' if not detected else '#454545'};display:inline-block"></span>
                  <span style="width:20px;height:20px;border-radius:50%;background:#454545;display:inline-block"></span>
                  <span style="width:20px;height:20px;border-radius:50%;background:{signal_color};display:inline-block"></span>
                </div>
                <div><b>{signal_text}</b><br><span class="small">{warning_text}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="panel">
              <h3>🚦 Traffic signal preview</h3>
              <p class="small">Run detection to update the simulated signal and warning.</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("Upload an image to begin. For the best demo, use a clear road image with or without an ambulance.")

st.divider()
st.caption("AI Emergency Traffic Control · Hackathon prototype · Image-based inference only")
