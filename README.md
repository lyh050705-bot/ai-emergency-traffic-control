# AI Emergency Traffic Control — Website Prototype

This is a Streamlit website version of the Google Colab ambulance-detection demo.

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set your Roboflow API key as an environment variable (do not put it in `app.py`):

   **Windows PowerShell**
   ```powershell
   $env:ROBOFLOW_API_KEY="YOUR_KEY_HERE"
   streamlit run app.py
   ```

   **macOS/Linux**
   ```bash
   export ROBOFLOW_API_KEY="YOUR_KEY_HERE"
   streamlit run app.py
   ```

## Deploy with Streamlit Community Cloud

1. Upload `app.py` and `requirements.txt` to a private or public GitHub repository, as appropriate.
2. Create a Streamlit Community Cloud app connected to that repository.
3. In the app's **Settings → Secrets**, add:

   ```toml
   ROBOFLOW_API_KEY = "YOUR_KEY_HERE"
   ```

4. Deploy and test with an uploaded image.

## Important notes

- Never commit your API key to GitHub or paste it into the Python source code.
- The configured Roboflow workflow is `yong-han-lee/general-segmentation-api`, with the parameter `classes = "Ambulance"`. Confirm that the API key has access to this workflow.
- The dashboard is a simulation. This prototype processes uploaded images only; it does not use live camera feeds or control real traffic signals.
- The code assumes the workflow returns predictions at `result[0]["predictions"]["predictions"]`, matching the existing Colab prototype. If the workflow output format changes, update the parsing code.
