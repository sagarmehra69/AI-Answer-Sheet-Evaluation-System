import streamlit as st

st.set_page_config(
    page_title="AI Answer Sheet Evaluation System", page_icon="📄", layout="wide"
)

st.title("📄 AI-Powered Answer Sheet Evaluation System")

st.markdown("""
Automatically evaluate student answer sheets using:

- OCR (Optical Character Recognition)
- Semantic Similarity
- Keyword Matching
- Rubric-Based Evaluation
- Automated Report Generation
""")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("OCR Engine", "PaddleOCR")

with col2:
    st.metric("Evaluation", "AI Enabled")

with col3:
    st.metric("Reports", "PDF + Excel")

st.success("System Ready")
