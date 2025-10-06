import streamlit as st

# Set the page configuration
# This replaces the call to your custom page_config function
st.set_page_config(
    page_title="Harehills Interactive Analysis",
    layout="wide"
)

# --- Page Content ---

st.title("Harehills: Interactive Data Analysis")

st.markdown("""
Welcome to an interactive exploration of the Harehills area in Leeds. This project uses publicly available data to examine three key themes: **deprivation**, **crime**, and **police stop & search** incidents. The aim is to offer a data-driven examination of the neighbourhood by situating local statistics within the broader context of city-wide trends, providing a clear view of how the area is reflected in administrative data.

**How to Use This App**

Use the sidebar on the left to navigate between the different analytical sections:

*   **Deprivation (IMD):** Explore how Harehills ranks in the Index of Multiple Deprivation (IMD), both nationally and within Leeds.
*   **Crime:** Investigate reported crime rates, comparing Harehills to other Leeds neighbourhoods across various crime types.
*   **Stop and Search:** Analyse data on police stop and search incidents, with breakdowns by ethnicity, age, and outcome.

All visualisations are interactive. Use the buttons and dropdown menus to filter the data and tailor the analysis to your interests.
""")

# Data source information
st.info("All data is sourced from Police Data UK, the Office for National Statistics (Census 2021), and the Ministry of Housing, Communities & Local Government (IMD 2019).")