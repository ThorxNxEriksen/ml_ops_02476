import os
import requests
import streamlit as st
from google.cloud import run_v2

def get_backend_url():
    """Get the URL of the backend service."""
    # Set the parent project and location for the Cloud Run service
    parent = "projects/quickdrawproject-448508/locations/europe-west1"

    # Initialize the Cloud Run client
    client = run_v2.ServicesClient()

    # List all services in the specified location
    services = client.list_services(parent=parent)

    # Look for the service named "api"
    for service in services:
        if service.name.split("/")[-1] == "api":
            return service.uri  # Return the URL of the backend service

    # If the service is not found, fall back to an environment variable
    return os.environ.get("BACKEND", None)

def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/predict"
    response = requests.post(predict_url, files={"file": image}, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None

def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)
    st.write(f"Currently working on {backend}")
    st.title("Welcome to the image classification of images in the Quick, Draw! dataset")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)

        if result is not None:
            prediction = result["category"]
            confidence = result["confidence"]

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", prediction)
            st.write("Confidence:", confidence)
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()