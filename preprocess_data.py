#
# preprocess_data.py
#
# Description:
# This script loads and processes all raw data for the Harehills analysis app.
# It uses the original, unaltered helper functions from utils.py and follows
# the exact logic from the harehills_analysis.ipynb notebook to generate
# the final data files required for the interactive visualizations.
#

import os
import pandas as pd
import geopandas as gpd
import numpy as np

# Import the original, unaltered functions
from utils import *

# --- Configuration (Corrected) ---
# Get the absolute path of the current script's directory
try:
    # This works when running as a script
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive environments like Jupyter
    PROJECT_ROOT = os.getcwd()

# Define data directories relative to the project root
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# --- Main Data Pipeline ---

def main():
    """
    Orchestrates the entire data loading and processing pipeline.
    """
    print("--- Starting Data Preprocessing ---")

    # --- 1. Load and Clean Raw Data (from Notebook Cell 3) ---
    print("\n[Phase 1/5] Loading and cleaning raw data...")
    pduk_base_dir = os.path.join(RAW_DATA_DIR, "all_data_police_uk")
    crime_df, _, sns_df = load_police_data_uk_data(pduk_base_dir)

    sns_df = sns_df.pipe(clean_sns_data)
    sns_df = sns_df[sns_df.reporting_force_name == "West Yorkshire"]

    crime_df = clean_column_names(crime_df)
    crime_df['reporting_force_name'] = crime_df.source_file.apply(lambda x: x[8:-11])
    crime_df = apply_reporting_force_mappings(crime_df)
    crime_df.dropna(subset='lsoa_name', inplace=True)
    crime_df = crime_df[crime_df.lsoa_name.str.contains("Leeds")]

    imd_path = os.path.join(RAW_DATA_DIR, "imd_data/imd2019_msoa_level_data.csv")
    imd_df = pd.read_csv(imd_path)

    msoa_path = os.path.join(RAW_DATA_DIR, "boundaries/msoa_boundaries")
    msoa_gdf = gpd.read_file(msoa_path)
    msoa_gdf = msoa_gdf.pipe(standardize_geography_columns)
    msoa_names_path = os.path.join(RAW_DATA_DIR, "join_data/msoa_names_21.csv")
    msoa_gdf = add_hocl_msoa_names(msoa_gdf, msoa_names_path)

    sns_df = sns_df.pipe(assign_points_to_geography, msoa_gdf, geo_id_col="msoa_code", geo_name_col="msoa_name")
    sns_df.dropna(subset="msoa_name", inplace=True)
    sns_df = sns_df[sns_df.msoa_name.str.contains("Leeds")]
    msoa_gdf.dropna(subset="msoa_name", inplace=True)
    msoa_gdf = msoa_gdf[msoa_gdf.msoa_name.str.contains("Leeds")]
    sns_df = add_hocl_msoa_names(sns_df, msoa_names_path)

    crime_df = crime_df.pipe(assign_points_to_geography, msoa_gdf, geo_id_col="msoa_code", geo_name_col="msoa_name")
    crime_df = add_hocl_msoa_names(crime_df, msoa_names_path)

    census_path = os.path.join(RAW_DATA_DIR, "raw_census/census_21_msoa_stats.csv")
    census_df = clean_census_data_full(census_path)
    census_df.dropna(subset="msoa_name", inplace=True)
    census_df = census_df[census_df.msoa_name.str.contains("Leeds")]
    census_df = add_hocl_msoa_names(census_df, msoa_names_path)

    workday_path = os.path.join(RAW_DATA_DIR, "raw_census/census_21_workday_msoa_ethnicity_stats.csv")
    workday_df = clean_census_data_full(workday_path)
    workday_df.dropna(subset="msoa_name", inplace=True)
    workday_df = workday_df[workday_df.msoa_name.str.contains("Leeds")]
    workday_df = add_hocl_msoa_names(workday_df, msoa_names_path)

    # --- 2. Process Data for Plots (from Subsequent Notebook Cells) ---
    print("\n[Phase 2/5] Processing IMD analysis data...")
    imd_df.columns = imd_df.columns.str.lower()
    imd_df.rename(columns={'msoac': 'msoa_code'}, inplace=True)
    leeds_imd_gdf = msoa_gdf.merge(imd_df, on='msoa_code', how='left')
    harehills_names = ["Harehills North", "Harehills South"]
    leeds_imd_gdf['area'] = leeds_imd_gdf['msoa_hocl_name'].apply(lambda x: 'Harehills' if x in harehills_names else 'Rest of Leeds')
    
    print("\n[Phase 3/5] Processing Crime analysis data...")
    census_pop = census_df.groupby('msoa_hocl_name')['value'].sum().reset_index().rename(columns={'value': 'census_population'})
    workday_pop = workday_df.groupby('msoa_hocl_name')['value'].sum().reset_index().rename(columns={'value': 'workday_population'})
    population_df = pd.merge(census_pop, workday_pop, on='msoa_hocl_name', how='outer')
    crime_counts_df = crime_df.groupby(['msoa_hocl_name', 'crime_type']).size().reset_index(name='crime_count')
    crime_analysis_df = pd.merge(crime_counts_df, population_df, on='msoa_hocl_name', how='left')
    crime_analysis_df['census_pop_adj'] = crime_analysis_df['census_population'].replace(0, np.nan)
    crime_analysis_df['workday_pop_adj'] = crime_analysis_df['workday_population'].replace(0, np.nan)
    crime_analysis_df['rate_per_1k_census'] = (crime_analysis_df['crime_count'] / crime_analysis_df['census_pop_adj']) * 1000
    crime_analysis_df['rate_per_1k_workday'] = (crime_analysis_df['crime_count'] / crime_analysis_df['workday_pop_adj']) * 1000
    crime_analysis_df.fillna(0, inplace=True)
    crime_analysis_df['area'] = crime_analysis_df['msoa_hocl_name'].apply(lambda x: 'Harehills' if x in harehills_names else 'Rest of Leeds')

    crime_df_ts = crime_df.copy()
    crime_df_ts['month'] = pd.to_datetime(crime_df_ts['month'], format='%Y-%m')
    monthly_crime_counts = crime_df_ts.groupby(['month', 'msoa_hocl_name', 'crime_type']).size().reset_index(name='crime_count')
    ts_analysis_df = pd.merge(monthly_crime_counts, population_df, on='msoa_hocl_name', how='left')
    ts_analysis_df['census_pop_adj'] = ts_analysis_df['census_population'].replace(0, np.nan)
    ts_analysis_df['workday_pop_adj'] = ts_analysis_df['workday_population'].replace(0, np.nan)
    ts_analysis_df['rate_per_1k_census'] = (ts_analysis_df['crime_count'] / ts_analysis_df['census_pop_adj']) * 1000
    ts_analysis_df['rate_per_1k_workday'] = (ts_analysis_df['crime_count'] / ts_analysis_df['workday_pop_adj']) * 1000
    ts_analysis_df.fillna(0, inplace=True)
    ts_analysis_df['area'] = ts_analysis_df['msoa_hocl_name'].apply(lambda x: 'Harehills' if x in harehills_names else 'Rest of Leeds')
    leeds_msoas_df = ts_analysis_df[ts_analysis_df['area'] == 'Rest of Leeds']
    def p10(x): return x.quantile(0.1)
    def p90(x): return x.quantile(0.9)
    leeds_percentiles_df = leeds_msoas_df.groupby(['month', 'crime_type']).agg(
        census_p10=('rate_per_1k_census', p10), census_median=('rate_per_1k_census', 'median'), census_p90=('rate_per_1k_census', p90),
        workday_p10=('rate_per_1k_workday', p10), workday_median=('rate_per_1k_workday', 'median'), workday_p90=('rate_per_1k_workday', p90)
    ).reset_index()

    print("\n[Phase 4/5] Processing Stop & Search analysis data...")
    census_age_eth_pop = census_df.groupby(['msoa_hocl_name', 'ethnicity_simple', 'age_group'])['value'].sum().reset_index().rename(columns={'value': 'census_population'})
    census_eth_pop = census_df.groupby(['msoa_hocl_name', 'ethnicity_simple'])['value'].sum().reset_index().rename(columns={'value': 'census_population'})
    workday_eth_pop = workday_df.groupby(['msoa_hocl_name', 'ethnicity_simple'])['value'].sum().reset_index().rename(columns={'value': 'workday_population'})
    sns_age_eth_counts = sns_df.groupby(['msoa_hocl_name', 'ethnicity_simple', 'age_group']).size().reset_index(name='sns_count')
    sns_object_counts = sns_df.groupby(['msoa_hocl_name', 'object_of_search']).size().reset_index(name='sns_count')
    sns_outcome_counts = sns_df.groupby(['msoa_hocl_name', 'outcome']).size().reset_index(name='sns_count')
    
    sns_analysis_age_eth_df = pd.merge(sns_age_eth_counts, census_age_eth_pop, on=['msoa_hocl_name', 'ethnicity_simple', 'age_group'], how='left')
    sns_analysis_age_eth_df['rate_per_1k_census'] = (sns_analysis_age_eth_df['sns_count'] / sns_analysis_age_eth_df['census_population']) * 1000
    
    eth_pop_df = pd.merge(census_eth_pop, workday_eth_pop, on=['msoa_hocl_name', 'ethnicity_simple'], how='outer')
    sns_eth_counts = sns_df.groupby(['msoa_hocl_name', 'ethnicity_simple']).size().reset_index(name='sns_count')
    sns_analysis_eth_df = pd.merge(sns_eth_counts, eth_pop_df, on=['msoa_hocl_name', 'ethnicity_simple'], how='left')
    sns_analysis_eth_df['rate_per_1k_census'] = (sns_analysis_eth_df['sns_count'] / sns_analysis_eth_df['census_population']) * 1000
    sns_analysis_eth_df['rate_per_1k_workday'] = (sns_analysis_eth_df['sns_count'] / sns_analysis_eth_df['workday_population']) * 1000

    sns_analysis_object_of_search_df = pd.merge(sns_object_counts, population_df, on='msoa_hocl_name', how='left')
    sns_analysis_object_of_search_df['rate_per_1k_census'] = (sns_analysis_object_of_search_df['sns_count'] / sns_analysis_object_of_search_df['census_population']) * 1000
    sns_analysis_object_of_search_df['rate_per_1k_workday'] = (sns_analysis_object_of_search_df['sns_count'] / sns_analysis_object_of_search_df['workday_population']) * 1000

    sns_analysis_outcome_df = pd.merge(sns_outcome_counts, population_df, on='msoa_hocl_name', how='left')
    sns_analysis_outcome_df['rate_per_1k_census'] = (sns_analysis_outcome_df['sns_count'] / sns_analysis_outcome_df['census_population']) * 1000
    sns_analysis_outcome_df['rate_per_1k_workday'] = (sns_analysis_outcome_df['sns_count'] / sns_analysis_outcome_df['workday_population']) * 1000
    
    def final_prep_sns(df):
        df.fillna(0, inplace=True)
        df['area'] = df['msoa_hocl_name'].apply(lambda x: 'Harehills' if x in harehills_names else 'Rest of Leeds')
        return df
    
    sns_analysis_age_eth_df = final_prep_sns(sns_analysis_age_eth_df)
    sns_analysis_eth_df = final_prep_sns(sns_analysis_eth_df)
    sns_analysis_object_of_search_df = final_prep_sns(sns_analysis_object_of_search_df)
    sns_analysis_outcome_df = final_prep_sns(sns_analysis_outcome_df)

    sns_total_counts = sns_df.groupby('msoa_hocl_name').size().reset_index(name='sns_count')
    sns_analysis_total_df = pd.merge(sns_total_counts, population_df, on='msoa_hocl_name', how='left')
    sns_analysis_total_df['rate_per_1k_census'] = (sns_analysis_total_df['sns_count'] / sns_analysis_total_df['census_population']) * 1000
    sns_analysis_total_df['rate_per_1k_workday'] = (sns_analysis_total_df['sns_count'] / sns_analysis_total_df['workday_population']) * 1000
    sns_analysis_total_df = final_prep_sns(sns_analysis_total_df)

    sns_df_ts = sns_df.copy()
    sns_df_ts['month'] = pd.to_datetime(sns_df_ts['date']).dt.to_period('M').dt.to_timestamp()
    monthly_sns_counts_eth = sns_df_ts.groupby(['month', 'msoa_hocl_name', 'ethnicity_simple']).size().reset_index(name='sns_count')
    ts_sns_df = pd.merge(monthly_sns_counts_eth, eth_pop_df, on=['msoa_hocl_name', 'ethnicity_simple'], how='left')
    ts_sns_df['rate_per_1k_census'] = (ts_sns_df['sns_count'] / ts_sns_df['census_population']) * 1000
    ts_sns_df['rate_per_1k_workday'] = (ts_sns_df['sns_count'] / ts_sns_df['workday_population']) * 1000
    ts_sns_df = final_prep_sns(ts_sns_df)
    
    monthly_sns_total_counts = sns_df_ts.groupby(['month', 'msoa_hocl_name']).size().reset_index(name='sns_count')
    ts_sns_total_df = pd.merge(monthly_sns_total_counts, population_df, on='msoa_hocl_name', how='left')
    ts_sns_total_df['rate_per_1k_census'] = (ts_sns_total_df['sns_count'] / ts_sns_total_df['census_population']) * 1000
    ts_sns_total_df['rate_per_1k_workday'] = (ts_sns_total_df['sns_count'] / ts_sns_total_df['workday_population']) * 1000
    ts_sns_total_df = final_prep_sns(ts_sns_total_df)
    
    leeds_msoas_ts_df = ts_sns_df[ts_sns_df['area'] == 'Rest of Leeds']
    leeds_percentiles_ts_df = leeds_msoas_ts_df.groupby(['month', 'ethnicity_simple']).agg(
        census_p10=('rate_per_1k_census', p10), census_median=('rate_per_1k_census', 'median'), census_p90=('rate_per_1k_census', p90),
        workday_p10=('rate_per_1k_workday', p10), workday_median=('rate_per_1k_workday', 'median'), workday_p90=('rate_per_1k_workday', p90)
    ).reset_index()

    leeds_msoas_ts_total_df = ts_sns_total_df[ts_sns_total_df['area'] == 'Rest of Leeds']
    leeds_percentiles_ts_total_df = leeds_msoas_ts_total_df.groupby(['month']).agg(
        census_p10=('rate_per_1k_census', p10), census_median=('rate_per_1k_census', 'median'), census_p90=('rate_per_1k_census', p90),
        workday_p10=('rate_per_1k_workday', p10), workday_median=('rate_per_1k_workday', 'median'), workday_p90=('rate_per_1k_workday', p90)
    ).reset_index()

    # --- 5. Collect and Save All DataFrames ---
    print("\n[Phase 5/5] Saving processed data files...")
    
    data_to_save = {
        'leeds_imd_gdf': leeds_imd_gdf, 'crime_analysis_df': crime_analysis_df,
        'ts_analysis_df': ts_analysis_df, 'leeds_percentiles_df': leeds_percentiles_df,
        'sns_analysis_age_eth_df': sns_analysis_age_eth_df, 'sns_analysis_eth_df': sns_analysis_eth_df,
        'sns_analysis_object_of_search_df': sns_analysis_object_of_search_df, 'sns_analysis_outcome_df': sns_analysis_outcome_df,
        'sns_analysis_total_df': sns_analysis_total_df, 'ts_sns_df': ts_sns_df,
        'ts_sns_total_df': ts_sns_total_df, 'leeds_percentiles_ts_df': leeds_percentiles_ts_df,
        'leeds_percentiles_ts_total_df': leeds_percentiles_ts_total_df
    }

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    for filename, df in data_to_save.items():
        is_gdf = isinstance(df, gpd.GeoDataFrame)
        extension = "geoparquet" if is_gdf else "parquet"
        output_path = os.path.join(PROCESSED_DATA_DIR, f"{filename}.{extension}")
        
        try:
            df.to_parquet(output_path, engine='pyarrow')
            print(f"  Successfully saved {output_path}")
        except Exception as e:
            print(f"  ERROR saving {filename}: {e}")

    print("\n--- Preprocessing complete. ---")

if __name__ == "__main__":
    try:
        import pyarrow
    except ImportError:
        print("This script requires the 'pyarrow' library. Please install it using: pip install pyarrow")
        exit()
    main()