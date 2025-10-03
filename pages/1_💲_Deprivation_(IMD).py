# pages/1_Deprivation_(IMD).py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
import pandas as pd
from app_utils import page_config, load_data

# --- Page Configuration and Data Loading ---
page_config("Deprivation Analysis (IMD)")
leeds_imd_gdf = load_data("leeds_imd_gdf.geoparquet")

# --- Page Content ---
st.title("Deprivation Analysis: The Index of Multiple Deprivation (IMD)")

st.markdown("""
The Index of Multiple Deprivation (IMD) is the official measure of relative deprivation for small areas in England. It ranks every neighbourhood from most deprived to least deprived. A rank of 1 indicates the most deprived area.

Here, we explore where the Harehills MSOAs (Middle-layer Super Output Areas) stand, both in the national context and within Leeds itself.
""")

# --- Section 1: Interactive Choropleth Map ---
st.header("IMD Deciles Map")

st.markdown("""
This map shows deprivation deciles across Leeds. **Decile 1 represents the 10% most deprived areas**, while decile 10 represents the 10% least deprived. You can switch between national deciles and deciles calculated only for Leeds.

- **National Deciles:** Both Harehills North and South fall into the most deprived decile nationally. However, many surrounding regions share this characteristic.
- **Leeds-Specific Deciles:** When we re-rank based only on areas within Leeds, Harehills South remains in the most deprived decile (1), while Harehills North moves to the second-most deprived (2).
""")

# Interactive widget for map
decile_type = st.radio(
    "Select Decile Type:",
    options=[('National Deciles', 'msoadecile'), ('Leeds-Specific Deciles', 'leeds_decile')],
    format_func=lambda x: x[0],
    horizontal=True
)
decile_column = decile_type[1]

# Plotting function for map
def plot_choropleth(gdf, column):
    legend_title = 'IMD Decile\n(1 = Most Deprived)'
    if column == 'leeds_decile':
        gdf['leeds_decile'] = pd.qcut(-gdf['imd19 score'], q=10, labels=False, duplicates='drop') + 1
        legend_title = 'Leeds IMD Decile\n(1 = Most Deprived)'

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    gdf_web_mercator = gdf.to_crs(epsg=3857)

    gdf_web_mercator.plot(
        column=column, categorical=True, cmap='Reds_r', linewidth=0.7, ax=ax,
        edgecolor='#333333', legend=True, alpha=0.5,
        legend_kwds={
            'loc': 'lower left', 'title': legend_title, 'frameon': True, 
            'facecolor': 'white', 'framealpha': 0.9
        }
    )
    # Highlight Harehills
    gdf_web_mercator[gdf_web_mercator['area'] == 'Harehills'].plot(
        ax=ax, edgecolor='yellow', facecolor='none', linewidth=2
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_axis_off()
    return fig

if leeds_imd_gdf is not None:
    st.pyplot(plot_choropleth(leeds_imd_gdf.copy(), decile_column))


# --- Section 2: Histogram of IMD Scores ---
st.header("Distribution of IMD Scores Across Leeds")
st.markdown("This histogram shows the spread of IMD scores across all Leeds MSOAs. While both Harehills MSOAs have high scores (indicating higher deprivation), they are not the absolute highest in the city.")

def plot_histogram(gdf):
    harehills_scores = gdf[gdf['area'] == 'Harehills']
    score_north = harehills_scores[harehills_scores['msoa_hocl_name'] == 'Harehills North']['imd19 score'].iloc[0]
    score_south = harehills_scores[harehills_scores['msoa_hocl_name'] == 'Harehills South']['imd19 score'].iloc[0]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(data=gdf, x='imd19 score', kde=True, bins=20, ax=ax)
    ax.axvline(x=score_north, color='#d9534f', linestyle='--', linewidth=2, label=f'Harehills North (Score: {score_north:.2f})')
    ax.axvline(x=score_south, color='#f0ad4e', linestyle='--', linewidth=2, label=f'Harehills South (Score: {score_south:.2f})')
    ax.set_xlabel('IMD Score (Higher = More Deprived)')
    ax.set_ylabel('Number of MSOAs')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    return fig

if leeds_imd_gdf is not None:
    st.pyplot(plot_histogram(leeds_imd_gdf))


# --- Section 3: Deprivation Ranking Table ---
st.header("Deprivation Rankings in Leeds")
st.markdown("The table below shows the definitive ranks for all MSOAs in Leeds. You can scroll through the list to see the full context. **Harehills South is the 4th most deprived MSOA in Leeds**, and **Harehills North is the 16th**.")

def display_ranked_table(gdf):
    """
    Creates a scrollable, ranked table of all MSOAs, with custom styling
    for Harehills North and South.
    """
    # Calculate rank and sort the entire dataframe
    gdf['leeds_rank'] = gdf['imd19 score'].rank(method='dense', ascending=False).astype(int)
    sorted_gdf = gdf.sort_values('leeds_rank').reset_index(drop=True)

    # Format the columns for display
    display_df = sorted_gdf[['msoa_hocl_name', 'leeds_rank', 'imd19 score', 'msoadecile']].rename(columns={
        'msoa_hocl_name': 'MSOA Name', 'leeds_rank': 'Rank in Leeds',
        'imd19 score': 'IMD Score', 'msoadecile': 'National Decile'
    })

    # Define the new styling function
    def style_harehills_rows(row):
        style = ''
        if row['MSOA Name'] == 'Harehills South':
            # Orange and bold text for Harehills South
            style = 'color: #f0ad4e; font-weight: bold;'
        elif row['MSOA Name'] == 'Harehills North':
            # Red and bold text for Harehills North
            style = 'color: #d9534f; font-weight: bold;'
        
        # Apply the style to the entire row if a match is found
        return [style] * len(row) if style else [''] * len(row)

    # Apply the styling to the dataframe
    styled_df = display_df.style.apply(style_harehills_rows, axis=1).format({'IMD Score': "{:.2f}"})
    
    return styled_df

if leeds_imd_gdf is not None:
    st.dataframe(display_ranked_table(leeds_imd_gdf), use_container_width=True, height=500)