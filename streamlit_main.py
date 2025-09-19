import os
import streamlit as st
import pandas as pd

from utils.streamlit_utils import *
from utils.path import Get_path

# -----------------------------------------------------------------------------
# Constants & paths
# -----------------------------------------------------------------------------
MODEL_PATH = os.path.join(Get_path.model_path, "trained_model.pkl")
ENCODER_PATH = Get_path.encoder_path

# -----------------------------------------------------------------------------
# Streamlit UI – file upload
# -----------------------------------------------------------------------------
st.markdown("## 프로젝트명: 소셜리스닝 데이터 분석을 위한 AI 기반 프로파일 예측 모델 개발")
st.title("User Profile Prediction")
st.write(
    "Upload a file containing free-text responses to predict various profile attributes."
)

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file", type=["csv", "xlsx"]
)

# -----------------------------------------------------------------------------
# Data loading & validation
# -----------------------------------------------------------------------------
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, index_col=0)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    if "text" not in df.columns:
        st.error("Uploaded file must contain a 'text' column.")
        st.stop()

    st.success("File successfully uploaded.")
    st.write(df.head())

    # -------------------------------------------------------------------------
    # Model & encoder loading
    # -------------------------------------------------------------------------
    model, encoder_dict = get_model_encoder(MODEL_PATH, ENCODER_PATH)

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------
    df = predict_user_profile(
        model=model,
        encoder_dict=encoder_dict,
        df=df,
        model_ckpt="bert-base-uncased",
        chunk_size=32,
        batch_size=4,
    )

    # -------------------------------------------------------------------------
    # Display results
    # -------------------------------------------------------------------------
    st.subheader("Prediction Results")
    pred_cols = [
        col
        for col in df.columns
        if col not in ["text", "EMBEDDING"] and col.endswith("_pred")
    ]
    st.write(df[pred_cols].head())

    # -------------------------------------------------------------------------
    # Distribution plots
    # -------------------------------------------------------------------------
    order = [
        col
        for col in df.columns
        if col not in ["text", "EMBEDDING"] and col.endswith("_pred")
    ]

    st.subheader("Prediction Distributions")
    for col in order:
        plot_pie(df[f"{col}"], f"{col.capitalize()} Distribution")

    # -------------------------------------------------------------------------
    # Cross‑distribution analysis
    # -------------------------------------------------------------------------
    st.subheader("Compare Distribution Between Two Variables")
    categorical_columns = [f"{k}_pred" for k in encoder_dict.keys()]
    col1 = st.selectbox("Select Variable 1", pred_cols, index=0)
    col2 = st.selectbox("Select Variable 2", pred_cols, index=1)
    show_cross_distribution(df, col1, col2)

    # -------------------------------------------------------------------------
    # Download results
    # -------------------------------------------------------------------------
    st.subheader("Download Prediction Results")
    output_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV", output_csv, "prediction_results.csv", "text/csv"
    )
