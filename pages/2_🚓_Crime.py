# pages/2_Crime.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from app_utils import page_config, load_data

# --- Page Configuration and Data Loading ---
page_config("Crime Analysis")
crime_df = load_data("crime_analysis_df.parquet")
ts_df = load_data("ts_analysis_df.parquet")
percentiles_df = load_data("leeds_percentiles_df.parquet")

# Ensure month is datetime for plotting
if ts_df is not None:
    ts_df['month'] = pd.to_datetime(ts_df['month'])
if percentiles_df is not None:
    percentiles_df['month'] = pd.to_datetime(percentiles_df['month'])

# --- Page Content ---
st.title("Crime Analysis")
st.markdown("""
This section explores reported crime rates in Harehills compared to the rest of Leeds. 
""")

with st.expander("**Note on Population Data**"):
    st.info("""
    Crime rates are calculated per 1,000 people. This analysis uses two different population bases, and the choice of base affects the interpretation of the resulting rate.

    #### Census (Residential) Population
    This is the official number of people who live in an area, based on census data.

    * **Interpretation:** This rate measures crime risk relative to the **resident community**.
    * **Considerations:** In areas with a small residential population but high daytime or evening footfall (e.g., city centres, retail parks), this denominator can produce an **artificially inflated crime rate**. It does not account for the large transient population of commuters, shoppers, and visitors who are also exposed to the risk of crime.

    #### Workday Population
    This is an estimate of the number of people present in an area during a typical workday, including residents and commuters.

    * **Interpretation:** This rate measures crime risk relative to the **total ambient population** during daytime hours. It is a proxy for an area's footfall.
    * **Considerations:** This measure provides a more representative rate of risk in busy commercial areas by using a larger, more appropriate denominator. However, it may be less relevant for crimes that disproportionately affect residents in their homes or outside of standard work hours, such as residential burglary.
    """)

# --- Section 1: Crime Rate Distributions (Box Plot) ---
st.header("How Do Harehills' Crime Rates Compare?")
st.markdown("""
The box plots below show the distribution of crime rates for all MSOAs in the 'Rest of Leeds'. The orange and red dots represent the rates for Harehills South and North, respectively. 

A dot above the box indicates a higher-than-average rate for that crime type. As you can see, both Harehills MSOAs consistently rank high across most categories, often falling in the top 25% (above the third quartile of the box).
""")

# Widget for population base
pop_rate_col_box = st.radio(
    "Select Population Base for Rate Calculation:",
    options=[('Census Population', 'rate_per_1k_census'), ('Workday Population', 'rate_per_1k_workday')],
    format_func=lambda x: x[0],
    key='radio_boxplot',
    horizontal=True
)

def plot_rate_distributions(df, rate_col):
    """
    Plots crime rate distributions, adding a conditional grid line for the max value
    to avoid overlaps and using standard number formatting.
    """
    rest_of_leeds = df[df['area'] == 'Rest of Leeds']
    harehills = df[df['area'] == 'Harehills']
    
    fig, ax = plt.subplots(figsize=(18, 10))
    order = rest_of_leeds.groupby('crime_type')[rate_col].median().sort_values(ascending=False).index
    
    sns.boxplot(data=rest_of_leeds, x='crime_type', y=rate_col, ax=ax, order=order, color='lightgray', showfliers=False)
    sns.stripplot(
        data=harehills, x='crime_type', y=rate_col, hue='msoa_hocl_name', ax=ax, order=order,
        size=10, linewidth=1.5, edgecolor='black', jitter=False,
        palette={'Harehills North': '#d9534f', 'Harehills South': '#f0ad4e'}
    )
    
    rate_name = "Census" if "census" in rate_col else "Workday"
    ax.set_ylabel('Crime Incidents per 1,000 People (Log Scale)', fontsize=14)
    ax.set_xlabel('')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    ax.set_yscale('log')
    
    # Ensure standard number formatting on the y-axis
    ax.yaxis.set_major_formatter(ScalarFormatter())
    
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=14, length=0)

    # --- ADDED: Conditionally add the max value line to prevent overlap ---
    if not harehills.empty:
        max_val = harehills[rate_col].max()
        current_ticks = ax.get_yticks()
        
        # Check if the max value is too close (e.g., within 8%) to any existing tick
        is_too_close = any(abs(max_val - tick) / tick < 0.08 for tick in current_ticks if tick > 0)
        
        # Only add the line and label if it's not too close to another line
        if not is_too_close:
            ax.axhline(y=max_val, color='dimgray', linestyle='--', linewidth=0.9, zorder=0)
            ax.text(0, max_val, f'{max_val:.0f} ', 
                    transform=ax.get_yaxis_transform(),
                    ha='right', va='center', color='black', fontsize=12)

    ax.legend(title='Harehills MSOAs')
    plt.tight_layout()

    return fig

if crime_df is not None:
    st.pyplot(plot_rate_distributions(crime_df, pop_rate_col_box[1]))


# --- Section 2: Top 20 MSOAs by Crime Type (Bar Chart) ---
st.header("Top 20 MSOAs by Crime Rate")
st.markdown("""
This chart ranks the top 20 MSOAs in Leeds for a selected crime type. Using the workday population often places Harehills South at or near the top for several categories, including "Possession of weapons", "Public order", and "Violence and sexual offences".
""")

col1, col2 = st.columns(2)
with col1:
    crime_type_bar = st.selectbox(
        "Select Crime Type:",
        options=sorted(crime_df['crime_type'].unique()) if crime_df is not None else [],
        index=0
    )
with col2:
    pop_rate_col_bar = st.radio(
        "Select Population Base:",
        options=[('Census Population', 'rate_per_1k_census'), ('Workday Population', 'rate_per_1k_workday')],
        format_func=lambda x: x[0],
        key='radio_barchart',
        horizontal=True
    )

def plot_msoa_crime_rates(df, crime_type, rate_col):
    filtered_df = df[df['crime_type'] == crime_type]
    sorted_df = filtered_df.sort_values(by=rate_col, ascending=False).head(20)
    
    palette = {'Harehills': '#d9534f', 'Rest of Leeds': '#5bc0de'}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=sorted_df, y='msoa_hocl_name', x=rate_col, hue='area', palette=palette, dodge=False, orient='h', ax=ax)
    
    rate_name = "Census" if "census" in rate_col else "Workday"
    ax.set_xlabel('Crime Incidents per 1,000 People')
    ax.set_ylabel('MSOA Name')
    ax.legend().remove()
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.7)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=8, length=0)
    plt.tight_layout()
    return fig

if crime_df is not None:
    st.pyplot(plot_msoa_crime_rates(crime_df, crime_type_bar, pop_rate_col_bar[1]))


# --- Section 3: Crime Rates Over Time (Time Series) ---
st.header("Crime Rate Trends Over Time")
st.markdown("""
This time series shows how crime rates have fluctuated monthly. The shaded gray area represents the 10th-90th percentile range for all other Leeds MSOAs, with the black dashed line showing the median. This provides context for the Harehills rates.

For "Public order" offences, you can see a moderate spike around June 2024, but rates were consistently high before and after, suggesting the issue is persistent rather than a one-off event.
""")

col3, col4 = st.columns(2)
with col3:
    crime_type_ts = st.selectbox(
        "Select Crime Type:",
        options=sorted(ts_df['crime_type'].unique()) if ts_df is not None else [],
        index=8 # Default to Public Order
    )
with col4:
    pop_rate_col_ts = st.radio(
        "Select Population Base:",
        options=[('Census Population', 'rate_per_1k_census'), ('Workday Population', 'rate_per_1k_workday')],
        format_func=lambda x: x[0],
        key='radio_timeseries',
        horizontal=True
    )

def plot_time_series(ts_data, percentiles_data, crime_type, rate_col):
    harehills_data = ts_data[(ts_data['crime_type'] == crime_type) & (ts_data['area'] == 'Harehills')]
    leeds_percentiles = percentiles_data[percentiles_data['crime_type'] == crime_type]
    
    if "census" in rate_col:
        p10, median, p90 = 'census_p10', 'census_median', 'census_p90'
        rate_name = "Census"
    else:
        p10, median, p90 = 'workday_p10', 'workday_median', 'workday_p90'
        rate_name = "Workday"
        
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.fill_between(leeds_percentiles['month'], leeds_percentiles[p10], leeds_percentiles[p90],
                    alpha=0.2, color='gray', label='Leeds 10th-90th Percentile')
    sns.lineplot(data=leeds_percentiles, x='month', y=median, ax=ax, linestyle='--', color='black', label='Leeds Median Rate')
    sns.lineplot(data=harehills_data, x='month', y=rate_col, hue='msoa_hocl_name', ax=ax, marker='o', linewidth=2.5,
                 palette={'Harehills North': '#d9534f', 'Harehills South': '#f0ad4e'})
    
    ax.set_xlabel('')
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.7)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=10, length=0)
    ax.set_ylabel('Crime Incidents per 1,000 People')
    ax.legend(title='Area Comparison')
    plt.tight_layout()
    return fig

if ts_df is not None and percentiles_df is not None:
    st.pyplot(plot_time_series(ts_df, percentiles_df, crime_type_ts, pop_rate_col_ts[1]))