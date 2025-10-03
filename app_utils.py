# app_utils.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import os

# Define the path to the processed data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

@st.cache_data
def load_data(filename):
    """
    Loads a parquet or geoparquet file from the processed data directory.
    Uses Streamlit's caching to enhance performance.
    """
    path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(path):
        st.error(f"Data file not found: {filename}. Please run preprocess_data.py first.")
        return None
        
    if filename.endswith(".geoparquet"):
        return gpd.read_parquet(path)
    else:
        return pd.read_parquet(path)

def page_config(title):
    """
    Sets the page configuration for a standard Streamlit page.
    """
    st.set_page_config(
        page_title=title,
        layout="wide",
        initial_sidebar_state="expanded"
    )