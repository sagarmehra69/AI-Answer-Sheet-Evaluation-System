import streamlit as st
import requests

st.title("📄 Batch Upload")

uploaded_file = st.file_uploader(
    "Upload Answer Sheet",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    st.image(uploaded_file)

    if st.button("Evaluate"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )
        }

        response = requests.post(
            "http://127.0.0.1:8000/evaluate-image",
            files=files,
        )

        if response.status_code == 200:

            result = response.json()

            st.success("Evaluation Completed")

            st.subheader("OCR Output")

            st.text(result["ocr_text"])

            st.subheader("Detected Questions")

            st.json(result["questions"])

            st.subheader("Evaluation Results")

            st.json(result["results"])

        else:

            st.error("Evaluation Failed")