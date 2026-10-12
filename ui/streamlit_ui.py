
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Industrial Fault Detection",
    page_icon="🏭",
    layout="wide",
)

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000/predict",
)


def main():
    st.title("Industrial Fault Detection")
    st.caption("ResNet-50 transfer learning · FastAPI · MLflow")

    uploaded = st.file_uploader(
        "Upload an industrial component image",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded is None:
        st.info("Upload an image to begin inspection.")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("Unable to read this image.")
        return

    left, right = st.columns(2)

    with left:
        st.subheader("Input image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Inspection result")

        if st.button("Run inspection", type="primary"):
            try:
                uploaded.seek(0)
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type or "image/jpeg",
                        )
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()

                prediction = result["prediction"]
                confidence = result["confidence"]

                st.markdown(f"### Prediction: {prediction}")
                st.metric(
                    "Model confidence",
                    f"{confidence * 100:.2f}%",
                )

                if prediction.lower() in {"good", "normal"}:
                    st.success("Predicted as a non-defective item.")
                else:
                    st.warning("Predicted as a potentially defective item.")

                probabilities = result.get("probabilities", {})
                if probabilities:
                    df = pd.DataFrame(
                        {
                            "Class": list(probabilities.keys()),
                            "Probability": list(probabilities.values()),
                        }
                    ).set_index("Class")
                    st.bar_chart(df)

            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
            except (KeyError, ValueError):
                st.error("The API returned an unexpected response.")


if __name__ == "__main__":
    main()
