# Home.py

import streamlit as st
from app_utils import page_config

# Set the page configuration
page_config("Harehills Interactive Analysis")

# --- Page Content ---

st.title("An Interactive Analysis of Harehills, Leeds")

st.markdown("""
Welcome to an interactive exploration of the Harehills area in Leeds. 
This project uses publicly available data to delve into three key areas: **deprivation**, **crime**, and **police stop & search operations**.

### The Motivation
The goal of this analysis is to provide a data-driven perspective on a specific neighbourhood. By comparing local statistics to city-wide trends, we can better understand the unique challenges and characteristics of the area.

### How to Use This App
Use the sidebar on the left to navigate between the different analytical sections:
- **Deprivation (IMD):** Explore how Harehills ranks in the Index of Multiple Deprivation (IMD), both nationally and within Leeds.
- **Crime:** Investigate reported crime rates, comparing Harehills to other Leeds neighbourhoods across various crime types.
- **Stop and Search:** Analyse data on police stop and search incidents, with breakdowns by ethnicity, age, and outcome.

All visualisations are interactive. Use the buttons and dropdown menus to filter the data and tailor the analysis to your interests.
""")

st.info("All data is sourced from Police Data UK, the Office for National Statistics (Census 2021), and the Ministry of Housing, Communities & Local Government (IMD 2019).")