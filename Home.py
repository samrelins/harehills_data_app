import streamlit as st

# Set the page configuration
# This replaces the call to your custom page_config function
st.set_page_config(
    page_title="Harehills Interactive Analysis",
    layout="wide"
)

# --- Page Content ---

st.title("Harehills: The Data Behind the Narrative")

st.markdown("""
Welcome to an interactive exploration of the Harehills area in Leeds. This project uses publicly available data to examine three key themes: deprivation, crime, and police stop & search operations. The aim of this analysis is to offer a data-driven examination of a neighbourhood often defined by headlines and reputation [1, 2]. By situating local statistics within the broader context of city-wide trends, we can build a more nuanced understanding of how the area is reflected in administrative data.

This external view, shaped by recurring crises and amplified by media, often paints a one-dimensional picture of a "problem" area [1]. It's a narrative rooted in the tangible challenges of high crime rates and deep socio-economic deprivation, with the Gipton & Harehills ward ranking among the most deprived nationally [2, 3].

However, this picture is profoundly incomplete. To its residents, Harehills is a diverse and resilient community, a place celebrated for its vibrant cultural mix and a powerful "grassroots spirit that has been lost in other more invested in parts of the city" [1]. This app provides the raw data to explore the space between these two competing narratives.
""")

# Data source information
st.info("All data is sourced from Police Data UK, the Office for National Statistics (Census 2021), and the Ministry of Housing, Communities & Local Government (IMD 2019).")

# Citations for the introductory text
st.markdown("""
---
### Citations
1.  Welcome to Leeds,(https://welcometoleeds.co.uk/suburb-guide/harehills/).
2.  The Guardian, [*How unrest in Leeds escalated – and was defused*](https://www.theguardian.com/uk-news/article/2024/jul/19/how-unrest-in-leeds-escalated-and-was-defused-harehills), 2024.
3.  Yorkshire Voice, [*Living in Poverty and deprived areas are seriously bad for your health*](https://yorkshirevoice.com/living-in-poverty-and-deprived-areas-are-seriously-bad-for-your-health), 2024.
""")