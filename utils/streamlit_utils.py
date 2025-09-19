import os
import pandas as pd, numpy as np

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import streamlit as st

from utils.utils import get_pickle, batch_extractor



@st.cache_resource
def get_model_encoder(mpath: str, epath: str) -> tuple[tf.keras.Model, dict[str, LabelEncoder]]:
    '''
    Load trained model and dictionary of label encoders : {column : LabelEncoder()}
    '''
    model = get_pickle(mpath)
    
    encoder_dict = {}
    for fname in os.listdir(epath):
        if fname.endswith(".pkl"):
            col_name = os.path.splitext(fname)[0]
            encoder_path = os.path.join(epath, fname)
            encoder = get_pickle(encoder_path)
            encoder_dict[col_name] = encoder

    return model, encoder_dict


def predict_user_profile(
    model : tf.keras.Model,
    encoder_dict: dict[str, LabelEncoder],
    df: pd.DataFrame,
    model_ckpt: str,
    chunk_size: int =32,
    batch_size: int =4
)-> pd.DataFrame:
    
    '''
    Extract embeddings, perform prediction and decoding,  
    then return the DataFrame including predicted labels.
    '''

    # Embedding Extraction
    with st.spinner("Extraction BERT embeddings in progress..."):
        df_with_emb = batch_extractor(df, "text", chunk_size, model_ckpt, batch_size)
        st.success("Embeddings extracted successfully.")

    # Prediction
    with st.spinner("Predicting User Profile..."):
        X = np.array(df_with_emb["EMBEDDING"].tolist())
        predictions = model.predict(X)
        st.success("Prediction Completed successfully.")

    # Decoding
    print("Start decoding")
    pred_order =  model.output_names
    print("pred_order", pred_order)
    
    for i, key in enumerate(pred_order):
        print("i:",i,"\nkey:",key)
        pred_labels = np.argmax(predictions[i], axis=1)
        decoded = encoder_dict[key].inverse_transform(pred_labels)
        df_with_emb[f"{key}_pred"] = decoded

    print(df_with_emb)

    return df_with_emb



def plot_pie(data: pd.Series, title: str):
    ''' 
    Display a pie chart of a categorical variable on the website
    '''
    fig, ax = plt.subplots()
    counts = data.value_counts()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    ax.axis("equal")
    st.pyplot(fig)


def show_cross_distribution(df: pd.DataFrame, col1: str, col2: str):
    '''
    Display a bar chart comparing the distribution of two categorical variables.
    '''
    if col1 != col2:
        cross_tab = pd.crosstab(df[col1], df[col2], normalize="index")
        st.write("Normalized Crosstab (by row):")
        st.dataframe(cross_tab)

        fig, ax = plt.subplots(figsize=(6, 4))
        cross_tab.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"Distribution of {col2} within {col1}")
        ax.set_ylabel("Proportion")
        ax.legend(title=col2, bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig)
    else:
        st.info("Please select two **different** variables to compare.")
