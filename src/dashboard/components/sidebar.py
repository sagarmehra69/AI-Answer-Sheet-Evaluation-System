import streamlit as st


def render_sidebar():

    st.sidebar.title("Navigation")

    st.sidebar.markdown("---")

    st.sidebar.write("Admin Dashboard")
    st.sidebar.write("Batch Upload")
    st.sidebar.write("Teacher Review")
    st.sidebar.write("Examiner Review")
    st.sidebar.write("Student Results")
    st.sidebar.write("Analytics")
