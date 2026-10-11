import streamlit as st
import requests
from PIL import Image
import pandas as pd

# Configure the Streamlit page styling
st.set_page_config(
    page_title="Industrial Fault Detection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint (Points to your FastAPI backend)
API_URL = "http://localhost:8000/predict"

def main():
    # Header Section
    st.title("🏭 Industrial Fault Detection Dashboard")
    st.markdown("""
    Upload an image of an industrial component from the factory floor to detect physical defects. 
    This system is powered by a custom ResNet-50 computer vision pipeline trained on the MVTec AD dataset.
    """)

    # Sidebar Information
    st.sidebar.header("System Architecture")
    st.sidebar.info(
        "**Frontend:** Streamlit\n\n"
        "**Backend:** FastAPI\n\n"
        "**Model:** Custom ResNet-50 (TensorFlow/Keras)\n\n"
        "**Tracking:** MLflow"
    )

    st.sidebar.header("Instructions")
    st.sidebar.markdown(
        "1. Ensure the FastAPI server is running (`python api/main.py`).\n"
        "2. Upload a `.png` or `.jpg` image of a component.\n"
        "3. Click **Run Inspection**."
    )

    # Main File Uploader
    uploaded_file = st.file_uploader("Upload Component Image for Inspection", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Create a two-column layout for visual balance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Live Camera Feed / Upload")
            image = Image.open(uploaded_file)
            st.image(image, caption="Component ready for analysis", use_column_width=True)

        with col2:
            st.subheader("AI Inspection Results")
            
            if st.button("Run Inspection", type="primary", use_container_width=True):
                with st.spinner("Analyzing structural and textural features..."):
                    try:
                        # Prepare the file for a multipart POST request
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        
                        # Send the image to the FastAPI backend
                        response = requests.post(API_URL, files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            prediction = result.get("prediction", "Unknown")
                            confidence = result.get("confidence", 0.0)
                            raw_probs = result.get("raw_probabilities", [])

                            # Display the final verdict with color coding
                            st.markdown("### Verdict:")
                            if prediction.lower() == "good":
                                st.success(f"✅ **{prediction.upper()}** (No anomalies detected)")
                            else:
                                st.error(f"⚠️ **{prediction.upper()}** (Structural or textural defect detected)")
                            
                            # Display Confidence Score
                            st.metric(label="Model Confidence", value=f"{confidence * 100:.2f}%")

                            # Display the raw probabilities as a bar chart
                            if raw_probs:
                                st.markdown("#### Softmax Probabilities:")
                                # Mapping to a dataframe for Streamlit charting
                                df = pd.DataFrame({
                                    "Probability": raw_probs
                                }, index=["Class 0 (Anomaly)", "Class 1 (Good)"])
                                
                                st.bar_chart(df, color="#2e86c1")

                        else:
                            st.error(f"API Error [{response.status_code}]: {response.text}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("🚨 Connection Refused: Ensure your FastAPI backend is running on http://localhost:8000.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()