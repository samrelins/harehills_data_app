# pages/3_Stop_and_Search.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from app_utils import page_config, load_data

# --- Page Configuration and Data Loading ---
page_config("Stop & Search Analysis")

# Load all necessary datasets
data_files = {
    "eth_df": "sns_analysis_eth_df.parquet",
    "age_eth_df": "sns_analysis_age_eth_df.parquet",
    "object_df": "sns_analysis_object_of_search_df.parquet",
    "outcome_df": "sns_analysis_outcome_df.parquet",
    "total_df": "sns_analysis_total_df.parquet",
    "ts_sns_df": "ts_sns_df.parquet",
    "ts_sns_total_df": "ts_sns_total_df.parquet",
    "percentiles_ts_df": "leeds_percentiles_ts_df.parquet",
    "percentiles_ts_total_df": "leeds_percentiles_ts_total_df.parquet"
}
dfs = {name: load_data(file) for name, file in data_files.items()}

# Convert month columns to datetime
for name in ['ts_sns_df', 'ts_sns_total_df', 'percentiles_ts_df', 'percentiles_ts_total_df']:
    if dfs[name] is not None:
        dfs[name]['month'] = pd.to_datetime(dfs[name]['month'])

# --- Page Content ---
st.title("Stop and Search Analysis")
st.markdown("""
This final section analyses police stop and search (S&S) data, comparing Harehills to the rest of Leeds. The rates are calculated per 1,000 people, using either census (residential) or workday population figures.
""")

# --- Section 1: S&S Rate Distributions (Box Plot) ---
st.header("How do Harehills' S&S Rates Compare?")
st.markdown("""
The plots below show the distribution of S&S rates across Leeds for different categories. The dots for Harehills North and South are consistently far above the median for the rest of the city, indicating significantly higher rates of stop and search activity.
""")

col1, col2 = st.columns(2)
with col1:
    category_box = st.selectbox(
        "Select Category to Analyse:",
        options=[('Ethnicity', 'ethnicity_simple'), ('Object of Search', 'object_of_search'), ('Outcome', 'outcome'), ('Age Group', 'age_group')],
        format_func=lambda x: x[0]
    )
with col2:
    pop_rate_col_box = st.radio(
        "Select Population Base:",
        options=[('Census Population', 'rate_per_1k_census'), ('Workday Population', 'rate_per_1k_workday')],
        format_func=lambda x: x[0],
        key='radio_sns_box',
        horizontal=True
    )

def plot_sns_rate_distributions(dfs, category_info, rate_col):
    category_name, category_col = category_info
    
    # Select correct dataframe
    if category_col == 'ethnicity_simple': base_df = dfs['eth_df']
    elif category_col == 'age_group': base_df = dfs['age_eth_df']
    elif category_col == 'object_of_search': base_df = dfs['object_df']
    else: base_df = dfs['outcome_df'] # Outcome
        
    if 'workday' in rate_col and category_col == 'age_group':
        st.warning("Workday population data is not available with an age breakdown. Please select Census Population.")
        return None

    plot_df = base_df[base_df[rate_col] > 0]
    rest_of_leeds = plot_df[plot_df['area'] == 'Rest of Leeds']
    harehills = plot_df[plot_df['area'] == 'Harehills']

    fig, ax = plt.subplots(figsize=(18, 10))
    order = rest_of_leeds.groupby(category_col)[rate_col].median().sort_values(ascending=False).index
    
    sns.boxplot(data=rest_of_leeds, x=category_col, y=rate_col, ax=ax, order=order, color='lightgray', showfliers=False)
    sns.stripplot(data=harehills, x=category_col, y=rate_col, hue='msoa_hocl_name', ax=ax, order=order,
                  size=12, linewidth=1.5, edgecolor='black', jitter=False, palette={'Harehills North': '#d9534f', 'Harehills South': '#f0ad4e'})

    rate_name = "Census" if "census" in rate_col else "Workday"
    ax.set_ylabel('S&S Incidents per 1,000 People (Log Scale)', fontsize=14)
    ax.set_xlabel('')
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=14, length=0)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.legend(title='Harehills MSOAs')
    plt.tight_layout()
    return fig

if all(df is not None for df in dfs.values()):
    fig = plot_sns_rate_distributions(dfs, category_box, pop_rate_col_box[1])
    if fig:
        st.pyplot(fig)

# --- Section 2: Top 25 MSOAs by Ethnicity (Bar Chart) ---
st.header("Top 25 MSOAs by S&S Rate and Ethnicity")
st.markdown("""
This chart shows the top 25 MSOAs with the highest S&S rates, which you can filter by ethnicity. When using the workday population, Harehills South shows the highest S&S rates in Leeds for every reported ethnicity except 'Black' and 'Asian'.
""")

# Prepare ethnicity options
if dfs['eth_df'] is not None:
    eth_options = ['All'] + sorted([e for e in dfs['eth_df']['ethnicity_simple'].unique() if e.lower() != 'unknown'])
else:
    eth_options = ['All']

col3, col4 = st.columns(2)
with col3:
    ethnicity_bar = st.selectbox("Select Ethnicity:", options=eth_options)
with col4:
    pop_rate_col_bar = st.radio(
        "Select Population Base:",
        options=[('Census Population', 'rate_per_1k_census'), ('Workday Population', 'rate_per_1k_workday')],
        format_func=lambda x: x[0], key='radio_sns_bar', horizontal=True
    )

def plot_msoa_sns_rates(dfs, ethnicity, rate_col):
    source_df = dfs['total_df'] if ethnicity == 'All' else dfs['eth_df'][dfs['eth_df']['ethnicity_simple'] == ethnicity]
    sorted_df = source_df.sort_values(by=rate_col, ascending=False).head(25)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.barplot(data=sorted_df, x='msoa_hocl_name', y=rate_col, hue='area', 
                palette={'Harehills': '#d9534f', 'Rest of Leeds': '#5bc0de'}, dodge=False, ax=ax)

    rate_name = "Census" if "census" in rate_col else "Workday"
    ax.set_xlabel('')
    ax.set_ylabel('S&S Incidents per 1,000 People')
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.7)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=14, length=0)
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Area')
    plt.tight_layout()
    return fig

if all(df is not None for df in dfs.values()):
    st.pyplot(plot_msoa_sns_rates(dfs, ethnicity_bar, pop_rate_col_bar[1]))

# --- Section 3: S&S Rates Over Time ---
st.header("S&S Rate Trends Over Time")
st.markdown("""
This time series shows monthly S&S rates. Rates for most ethnicities appear to have fallen from a high point at the beginning of the dataset. 

Notably, the S&S rate for the 'White' population in Harehills is exceptionally high compared to the rest of Leeds and appears to be the primary driver of the overall high S&S rates in the area, particularly in the latter half of the dataset.
""")

col5, col6 = st.columns(2)
with col5:
    ethnicity_ts = st.selectbox("Select Ethnicity:", options=eth_options, key='ts_eth_select')
with col6:
    pop_rate_col_ts = st.radio(
        "Select Population Base:",
        options=[('Census Population', 'rate_per_1k_census'), ('Workday Population', 'rate_per_1k_workday')],
        format_func=lambda x: x[0], key='radio_sns_ts', horizontal=True
    )

def plot_sns_time_series(dfs, ethnicity, rate_col):
    if ethnicity == 'All':
        harehills_data = dfs['ts_sns_total_df'][dfs['ts_sns_total_df']['area'] == 'Harehills']
        leeds_percentiles = dfs['percentiles_ts_total_df']
    else:
        harehills_data = dfs['ts_sns_df'][(dfs['ts_sns_df']['ethnicity_simple'] == ethnicity) & (dfs['ts_sns_df']['area'] == 'Harehills')]
        leeds_percentiles = dfs['percentiles_ts_df'][dfs['percentiles_ts_df']['ethnicity_simple'] == ethnicity]

    p10, median, p90, rate_name = ('census_p10', 'census_median', 'census_p90', 'Census') if "census" in rate_col else ('workday_p10', 'workday_median', 'workday_p90', 'Workday')
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.fill_between(leeds_percentiles['month'], leeds_percentiles[p10], leeds_percentiles[p90],
                    alpha=0.2, color='gray', label='Leeds 10th-90th Percentile')
    sns.lineplot(data=leeds_percentiles, x='month', y=median, ax=ax, linestyle='--', color='black', label='Leeds Median Rate')
    sns.lineplot(data=harehills_data, x='month', y=rate_col, hue='msoa_hocl_name', ax=ax, marker='o', linewidth=2,
                 palette={'Harehills North': '#d9534f', 'Harehills South': '#f0ad4e'})

    ax.set_xlabel('')
    ax.set_ylabel('S&S Incidents per 1,000 People')
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.7)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=14, length=0)
    ax.legend(title='Area Comparison')
    plt.tight_layout()
    return fig

if all(df is not None for df in dfs.values()):
    st.pyplot(plot_sns_time_series(dfs, ethnicity_ts, pop_rate_col_ts[1]))