#
# utils.py
#
# Description:
# A direct copy of the required, unaltered functions from data_prep.py.
# The internal logic of these functions has not been modified.
#

import geopandas as gpd
import glob
import numpy as np
import os
import pandas as pd
import re
from shapely.geometry import Point
import warnings

# ============================================
# COMMON FUNCTIONS
# ============================================

def clean_column_names(df):
    clean_columns = []
    for col in df.columns:
        col_lower = col.lower()
        clean_col = re.sub(r'[^a-z0-9]', '_', col_lower)
        clean_col = re.sub(r'_+', '_', clean_col)
        clean_col = clean_col.strip('_')
        clean_columns.append(clean_col)
    df.columns = clean_columns
    return df

def validate_mapping(df, column_name, mapping_dict):
    if column_name not in df.columns:
        return
    unique_values = df[column_name].dropna().unique()
    unmapped_values = [val for val in unique_values if val not in mapping_dict]
    if unmapped_values:
        raise ValueError(f"Unmapped values found in '{column_name}': {unmapped_values}")

# ============================================
# STOP AND SEARCH DATA FUNCTIONS
# ============================================

def load_police_data_uk_data(base_dir):
    data_containers = {'street': {'dfs': [], 'cols': None}, 'outcomes': {'dfs': [], 'cols': None}, 'stop-and-search': {'dfs': [], 'cols': None}}
    if not os.path.isdir(base_dir):
        print(f"Directory not found: {base_dir}")
        return None, None, None
    for year_month_dir in sorted(os.listdir(base_dir)):
        year_month_path = os.path.join(base_dir, year_month_dir)
        if not os.path.isdir(year_month_path):
            continue
        csv_files = glob.glob(os.path.join(year_month_path, '*.csv'))
        for csv_file in sorted(csv_files):
            for data_type, container in data_containers.items():
                if data_type in csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                    except Exception as e:
                        print(f"Error reading {csv_file}: {e}")
                        continue
                    if container['cols'] is None:
                        container['cols'] = list(df.columns)
                    elif list(df.columns) != container['cols']:
                        print(f"Warning: Column mismatch in {csv_file}. Skipping.")
                        continue
                    df['source_file'] = os.path.basename(csv_file)
                    container['dfs'].append(df)
                    break 
    results = []
    for data_type in ['street', 'outcomes', 'stop-and-search']:
        container = data_containers[data_type]
        if container['dfs']:
            df_final = pd.concat(container['dfs'], ignore_index=True)
            print(f"'{data_type}' master dataframe created with shape: {df_final.shape}")
            results.append(df_final)
        else:
            print(f"No '{data_type}' CSV files found.")
            results.append(None)
    return tuple(results)

def filter_and_basic_transforms(df):
    df = df[df.type != "Vehicle search"].copy()
    if 'policing_operation' in df.columns:
        df.drop(columns="policing_operation", inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.strftime('%Y-%m')
    df['year_month'] = pd.to_datetime(df['year_month'], format='%Y-%m')
    df["reporting_force_name"] = df.source_file.apply(lambda x: x[8:-20])
    return df

def get_reporting_force_mappings():
    pfa_names = {'avon-and-somerset': ('Avon and Somerset', 'E23000036'), 'bedfordshire': ('Bedfordshire', 'E23000026'), 'btp': ('British Transport Police', None), 'cambridgeshire': ('Cambridgeshire', 'E23000023'), 'cheshire': ('Cheshire', 'E23000006'), 'city-of-london': ('London, City of', 'E23000034'), 'cleveland': ('Cleveland', 'E23000013'), 'cumbria': ('Cumbria', 'E23000002'), 'derbyshire': ('Derbyshire', 'E23000018'), 'devon-and-cornwall': ('Devon & Cornwall', 'E23000035'), 'dorset': ('Dorset', 'E23000039'), 'durham': ('Durham', 'E23000008'), 'dyfed-powys': ('Dyfed-Powys', 'W15000004'), 'essex': ('Essex', 'E23000028'), 'gloucestershire': ('Gloucestershire', 'E23000037'), 'greater-manchester': ('Greater Manchester', 'E23000005'), 'gwent': ('Gwent', 'W15000002'), 'hampshire': ('Hampshire', 'E23000030'), 'hertfordshire': ('Hertfordshire', 'E23000027'), 'humberside': ('Humberside', 'E23000012'), 'kent': ('Kent', 'E23000032'), 'lancashire': ('Lancashire', 'E23000003'), 'leicestershire': ('Leicestershire', 'E23000021'), 'lincolnshire': ('Lincolnshire', 'E23000020'), 'merseyside': ('Merseyside', 'E23000004'), 'metropolitan': ('Metropolitan Police', 'E23000001'), 'norfolk': ('Norfolk', 'E23000024'), 'north-wales': ('North Wales', 'W15000001'), 'north-yorkshire': ('North Yorkshire', 'E23000009'), 'northamptonshire': ('Northamptonshire', 'E23000022'), 'northern-ireland': ('Northern Ireland', None), 'northumbria': ('Northumbria', 'E23000007'), 'nottinghamshire': ('Nottinghamshire', 'E23000019'), 'south-wales': ('South Wales', 'W15000003'), 'south-yorkshire': ('South Yorkshire', 'E23000011'), 'staffordshire': ('Staffordshire', 'E23000015'), 'suffolk': ('Suffolk', 'E23000025'), 'surrey': ('Surrey', 'E23000031'), 'sussex': ('Sussex', 'E23000033'), 'thames-valley': ('Thames Valley', 'E23000029'), 'warwickshire': ('Warwickshire', 'E23000017'), 'west-mercia': ('West Mercia', 'E23000016'), 'west-midlands': ('West Midlands', 'E23000014'), 'west-yorkshire': ('West Yorkshire', 'E23000010'), 'wiltshire': ('Wiltshire', 'E23000038')}
    return pfa_names

def apply_reporting_force_mappings(df):
    pfa_names_dict = get_reporting_force_mappings()
    pfa_names = {key: value[0] for key, value in pfa_names_dict.items()}
    pfa_codes = {key: value[1] for key, value in pfa_names_dict.items()}
    validate_mapping(df, 'reporting_force_name', pfa_names)
    df["reporting_force_code"] = df.reporting_force_name.map(pfa_codes)
    df["reporting_force_name"] = df.reporting_force_name.map(pfa_names)
    return df

def get_ethnicity_mappings():
    simple_mapping = {'White - English/Welsh/Scottish/Northern Irish/British': "White", 'Other ethnic group - Not stated': "Unknown", 'White - Any other White background': "White", 'Black/African/Caribbean/Black British - Any other Black/African/Caribbean background': "Black", 'Black/African/Caribbean/Black British - African': "Black", 'Asian/Asian British - Any other Asian background': "Asian", 'Asian/Asian British - Pakistani': "Asian", 'Black/African/Caribbean/Black British - Caribbean': "Black", 'Other ethnic group - Any other ethnic group': "Other", 'Mixed/Multiple ethnic groups - Any other Mixed/Multiple ethnic background': "Mixed", 'Asian/Asian British - Bangladeshi': "Asian", 'Asian/Asian British - Indian': "Asian", 'Mixed/Multiple ethnic groups - White and Black Caribbean': "Mixed", 'White - Irish': "White", 'Mixed/Multiple ethnic groups - White and Asian': "Mixed", 'White - Gypsy or Irish Traveller': "White", 'Mixed/Multiple ethnic groups - White and Black African': "Mixed", 'Other ethnic group - Arab': "Other", 'Asian/Asian British - Chinese': "Asian"}
    full_mapping = {'White - English/Welsh/Scottish/Northern Irish/British': "White British", 'Other ethnic group - Not stated': "Unknown", 'White - Any other White background': "White Other", 'Black/African/Caribbean/Black British - Any other Black/African/Caribbean background': "Black Other", 'Black/African/Caribbean/Black British - African': "Black African", 'Asian/Asian British - Any other Asian background': "Asian Other", 'Asian/Asian British - Pakistani': "Pakistani", 'Black/African/Caribbean/Black British - Caribbean': "Black Caribbean", 'Other ethnic group - Any other ethnic group': "Any Other Ethnic Background", 'Mixed/Multiple ethnic groups - Any other Mixed/Multiple ethnic background': "Mixed Other", 'Asian/Asian British - Bangladeshi': "Bangladeshi", 'Asian/Asian British - Indian': "Indian", 'Mixed/Multiple ethnic groups - White and Black Caribbean': "Mixed White and Black Caribbean", 'White - Irish': "White Irish", 'Mixed/Multiple ethnic groups - White and Asian': "Mixed White and Asian", 'White - Gypsy or Irish Traveller': "Gypsy or Irish Traveller", 'Mixed/Multiple ethnic groups - White and Black African': "Mixed White and Black African", 'Other ethnic group - Arab': "Arab", 'Asian/Asian British - Chinese': "Chinese"}
    return simple_mapping, full_mapping

def apply_ethnicity_mappings(df):
    simple_mapping, full_mapping = get_ethnicity_mappings()
    validate_mapping(df, 'self_defined_ethnicity', simple_mapping)
    df["ethnicity_simple"] = df.self_defined_ethnicity.map(simple_mapping)
    na_eth = df.ethnicity_simple.isna()
    not_na_or_eth = df.officer_defined_ethnicity.notna()
    fill_eths = na_eth & not_na_or_eth
    df.loc[fill_eths, "ethnicity_simple"] = df[fill_eths].officer_defined_ethnicity
    validate_mapping(df, 'self_defined_ethnicity', full_mapping)
    df["ethnicity_full"] = df.self_defined_ethnicity.map(full_mapping)
    return df

def get_categorical_mappings():
    mappings = {'age_groups': {'18-24': "Under 25", 'over 34': "Over 34", '25-34': "25-34", '10-17': "Under 25", 'under 10': "Under 25"}, 'outcomes': {'A no further action disposal': 'No Action', 'Arrest': 'Arrest', 'Community resolution': 'Community Resolution', 'Summons / charged by post': 'Summons', 'Penalty Notice for Disorder': 'PND', 'Khat or Cannabis warning': 'Drug Warning', 'Caution (simple or conditional)': 'Caution'}}
    return mappings

def apply_categorical_mappings(df):
    mappings = get_categorical_mappings()
    if 'gender' in df.columns: df.rename(columns={"gender": "sex"}, inplace=True)
    if 'age_range' in df.columns:
        validate_mapping(df, 'age_range', mappings['age_groups'])
        df["age_group"] = df.age_range.map(mappings['age_groups'])
    if 'outcome' in df.columns:
        validate_mapping(df, 'outcome', mappings['outcomes'])
        df['outcome'] = df.outcome.map(mappings['outcomes'])
    return df

def remove_lancashire_coordinate_corruption(df):
    lancashire_mask = df['reporting_force_name'] == 'Lancashire'
    date_mask = df['year_month'] < pd.to_datetime('2022-05-02')
    coords_mask = df['latitude'].notna() & df['longitude'].notna()
    problematic_mask = lancashire_mask & date_mask & coords_mask
    if problematic_mask.sum() > 0:
        print(f"Removing {problematic_mask.sum():,} Lancashire coordinate records from before 2022-05-02 due to data corruption")
        df.loc[problematic_mask, ['latitude', 'longitude']] = None
    return df

def clean_sns_data(df):
    print("Cleaning column names...")
    df = clean_column_names(df)
    print("Applying basic filtering and transforms...")
    df = filter_and_basic_transforms(df)
    print("Applying police force mappings...")
    df = apply_reporting_force_mappings(df)
    print("Applying ethnicity mappings...")
    df = apply_ethnicity_mappings(df)
    print("Applying categorical mappings...")
    df = apply_categorical_mappings(df)
    print("Removing known corrupted data...")
    df = remove_lancashire_coordinate_corruption(df)
    print(f"Data cleaning complete. Final shape: {df.shape}")
    return df

# ============================================
# CENSUS DATA FUNCTIONS
# ============================================

def load_census_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Census data file not found at {filepath}")
    census_data = pd.read_csv(filepath)
    print(f"Census data loaded with shape: {census_data.shape}")
    return census_data

def find_and_standardise_census_columns(df):
    GEO_LEVEL_PATTERNS = {'msoa': {'code': 'middle_layer_super_output_areas_code', 'name': 'middle_layer_super_output_areas'}}
    DEMOGRAPHIC_PATTERNS = {'ethnicity': ['ethnic'], 'age': ['age'], 'sex': ['sex']}
    rename_map, found_columns, geo_level = {}, {}, None
    for level, patterns in GEO_LEVEL_PATTERNS.items():
        code_col = next((c for c in df.columns if patterns['code'] in c), None)
        name_col = next((c for c in df.columns if patterns['name'] in c and 'code' not in c), None)
        if not name_col: name_col = next((c for c in df.columns if patterns['name'] in c), None)
        if code_col and name_col:
            geo_level = level
            print(f"  ✓ Detected geography level: '{geo_level}'")
            print(f"    - Code column: '{code_col}'")
            print(f"    - Name column: '{name_col}'")
            rename_map[code_col] = f"{geo_level}_code"
            rename_map[name_col] = f"{geo_level}_name"
            found_columns['geography_code'] = code_col
            found_columns['geography_name'] = name_col
            break
    if not geo_level: raise ValueError("Could not detect a matching pair of geography code and name columns.")
    for standard_name, patterns in DEMOGRAPHIC_PATTERNS.items():
        best_match_col = next((c for c in df.columns if c not in rename_map and any(p in c for p in patterns) and 'code' not in c), None)
        if not best_match_col: best_match_col = next((c for c in df.columns if c not in rename_map and any(p in c for p in patterns)), None)
        if best_match_col:
            print(f"  ✓ Found '{standard_name}' column: '{best_match_col}'")
            rename_map[best_match_col] = standard_name
            found_columns[standard_name] = best_match_col
    identified_cols = set(rename_map.keys())
    remaining_cols = [c for c in df.columns if c not in identified_cols]
    numeric_cols = [c for c in remaining_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        value_col = numeric_cols[0]
    elif len(numeric_cols) > 1:
        preferred_cols = [c for c in numeric_cols if 'code' not in c]
        if len(preferred_cols) == 1:
            value_col = preferred_cols[0]
        else:
            raise ValueError(f"Found multiple unidentified numeric columns: {numeric_cols}.")
    else:
        raise ValueError("Could not dynamically find a numeric value/observation column.")
    print(f"  ✓ Found 'value' column: '{value_col}'")
    rename_map[value_col] = 'value'
    found_columns['value'] = value_col
    df = df.rename(columns=rename_map)
    for standard_name in ['ethnicity', 'age', 'sex']:
        if standard_name in found_columns and found_columns[standard_name] in df.columns:
             df = df.drop(columns=[found_columns[standard_name]])
    return df, found_columns, geo_level

def get_census_ethnicity_mappings():
    simple_mapping = {'Does not apply': "Not Applicable", 'Asian, Asian British or Asian Welsh: Bangladeshi': "Asian", 'Other ethnic group: Arab': "Other", 'White: Other White': "White", 'White: Roma': "White", 'White: Gypsy or Irish Traveller': "White", 'White: Irish': "White", 'White: English, Welsh, Scottish, Northern Irish or British': "White", 'Mixed or Multiple ethnic groups: Other Mixed or Multiple ethnic groups': "Mixed", 'Mixed or Multiple ethnic groups: White and Black Caribbean': "Mixed", 'Mixed or Multiple ethnic groups: White and Black African': "Mixed", 'Mixed or Multiple ethnic groups: White and Asian': "Mixed", 'Black, Black British, Black Welsh, Caribbean or African: Other Black': "Black", 'Black, Black British, Black Welsh, Caribbean or African: Caribbean': "Black", 'Black, Black British, Black Welsh, Caribbean or African: African': "Black", 'Asian, Asian British or Asian Welsh: Other Asian': "Asian", 'Asian, Asian British or Asian Welsh: Pakistani': "Asian", 'Asian, Asian British or Asian Welsh: Indian': "Asian", 'Asian, Asian British or Asian Welsh: Chinese': "Asian", 'Other ethnic group: Any other ethnic group': "Other"}
    full_mapping = {'Does not apply': "Not Applicable", 'Asian, Asian British or Asian Welsh: Bangladeshi': "Bangladeshi", 'Other ethnic group: Arab': "Arab", 'White: Other White': "White Other", 'White: Roma': "Roma", 'White: Gypsy or Irish Traveller': "Gypsy or Irish Traveller", 'White: Irish': "White Irish", 'White: English, Welsh, Scottish, Northern Irish or British': "White British", 'Mixed or Multiple ethnic groups: Other Mixed or Multiple ethnic groups': "Mixed Other", 'Mixed or Multiple ethnic groups: White and Black Caribbean': "Mixed White and Black Caribbean", 'Mixed or Multiple ethnic groups: White and Black African': "Mixed White and Black African", 'Mixed or Multiple ethnic groups: White and Asian': "Mixed White and Asian", 'Black, Black British, Black Welsh, Caribbean or African: Other Black': "Black Other", 'Black, Black British, Black Welsh, Caribbean or African: Caribbean': "Black Caribbean", 'Black, Black British, Black Welsh, Caribbean or African: African': "Black African", 'Asian, Asian British or Asian Welsh: Other Asian': "Asian Other", 'Asian, Asian British or Asian Welsh: Pakistani': "Pakistani", 'Asian, Asian British or Asian Welsh: Indian': "Indian", 'Asian, Asian British or Asian Welsh: Chinese': "Chinese", 'Other ethnic group: Any other ethnic group': "Any Other Ethnic Background"}
    return simple_mapping, full_mapping

def get_census_age_mappings():
    age_mappings = {'Aged 15 years and under': 'Under 25', 'Aged 16 to 24 years': 'Under 25', 'Aged 25 to 34 years': "25-34", 'Aged 35 to 49 years': "Over 34", 'Aged 50 to 64 years': "Over 34", 'Aged 65 years and over': "Over 34", 'Aged 24 years and under': 'Under 25', 'Aged under 25 years': 'Under 25', 'Aged 0 to 24 years': 'Under 25'}
    return age_mappings

def clean_census_data_full(filepath):
    print("--- Starting Census Data Cleaning ---")
    df = load_census_data(filepath)
    print("\nStep 1: Standardizing all column names to snake_case...")
    df = clean_column_names(df)
    print("\nStep 2: Finding and standardizing column roles...")
    df, found_columns, geo_level = find_and_standardise_census_columns(df)
    if 'ethnicity' in found_columns:
        print("\nStep 3: Applying ethnicity mappings...")
        simple_map, full_map = get_census_ethnicity_mappings()
        original_ethnicity_col = found_columns['ethnicity']
        df.rename(columns={'ethnicity': original_ethnicity_col}, inplace=True)
        validate_mapping(df, original_ethnicity_col, simple_map)
        df['ethnicity_simple'] = df[original_ethnicity_col].map(simple_map)
        df['ethnicity_full'] = df[original_ethnicity_col].map(full_map)
        df = df.drop(columns=[original_ethnicity_col])
        df = df[df['ethnicity_simple'] != 'Not Applicable'].copy()
    if 'age' in found_columns:
        print("\nStep 4: Applying age mappings...")
        age_map = get_census_age_mappings()
        original_age_col = found_columns['age']
        df.rename(columns={'age': original_age_col}, inplace=True)
        validate_mapping(df, original_age_col, age_map)
        df['age_group'] = df[original_age_col].map(age_map)
        df = df.drop(columns=[original_age_col])
        print("Aggregating data by new age groups...")
        group_cols = [f"{geo_level}_code", f"{geo_level}_name"]
        if 'sex' in df.columns: group_cols.append('sex')
        if 'ethnicity_simple' in df.columns: group_cols.append('ethnicity_simple')
        if 'ethnicity_full' in df.columns: group_cols.append('ethnicity_full')
        group_cols.append('age_group')
        df = df.groupby(group_cols).agg(value=('value', 'sum')).reset_index()
    print("\nStep 5: Finalizing columns...")
    keep_cols = [f"{geo_level}_code", f"{geo_level}_name", 'value']
    for col in ['sex', 'age_group', 'ethnicity_simple', 'ethnicity_full']:
        if col in df.columns: keep_cols.append(col)
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols]
    print("\n--- Census data cleaning complete ---")
    print(f"Final shape: {df.shape}")
    print(f"Final columns: {', '.join(df.columns)}")
    return df

# ============================================
# Geolocation Helper Functions
# ============================================

def assign_points_to_geography(points_df, geography_gdf, geo_id_col, geo_name_col):
    if geography_gdf.crs.to_epsg() != 4326:
        print(f"Converting geography from {geography_gdf.crs} to EPSG:4326")
        geography_gdf = geography_gdf.to_crs('EPSG:4326')
    data = points_df.copy()
    has_coords = data['latitude'].notna() & data['longitude'].notna()
    valid_points = data[has_coords].copy()
    print(f"Processing {len(valid_points):,} points with coordinates...")
    valid_points['geometry'] = valid_points.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    points_gdf = gpd.GeoDataFrame(valid_points, geometry='geometry', crs='EPSG:4326')
    matched = gpd.sjoin(points_gdf, geography_gdf[[geo_id_col, geo_name_col, 'geometry']], how='left', predicate='within')
    matched = matched[~matched.index.duplicated(keep='first')]
    n_matched = matched[geo_id_col].notna().sum()
    n_unmatched = matched[geo_id_col].isna().sum()
    print(f"  Matched within boundaries: {n_matched:,}")
    print(f"  Unmatched: {n_unmatched:,}")
    if n_unmatched > 0:
        print(f"  Finding nearest geography for unmatched points...")
        unmatched_mask = matched[geo_id_col].isna()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS', UserWarning)
            for idx in matched[unmatched_mask].index:
                point_geom = points_gdf.loc[idx, 'geometry']
                distances = geography_gdf.geometry.distance(point_geom)
                nearest_idx = distances.idxmin()
                if distances.loc[nearest_idx] <= 0.008:
                    matched.loc[idx, geo_id_col] = geography_gdf.loc[nearest_idx, geo_id_col]
                    matched.loc[idx, geo_name_col] = geography_gdf.loc[nearest_idx, geo_name_col]
        n_assigned = (matched[unmatched_mask][geo_id_col].notna()).sum()
        print(f"  Assigned {n_assigned:,} points to nearest geography")
    matched = matched.drop(columns=['geometry', 'index_right'], errors='ignore')
    data[geo_id_col], data[geo_name_col] = None, None
    data.loc[has_coords, geo_id_col] = matched[geo_id_col].values
    data.loc[has_coords, geo_name_col] = matched[geo_name_col].values
    total_rows = len(data)
    rows_without_coords = (~has_coords).sum()
    rows_with_coords = has_coords.sum()
    total_assigned = data[geo_id_col].notna().sum()
    rows_with_coords_unassigned = rows_with_coords - total_assigned
    print(f"\nSummary:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Rows without coordinates: {rows_without_coords:,} ({rows_without_coords*100/total_rows:.1f}%)")
    print(f"  Rows with coordinates: {rows_with_coords:,} ({rows_with_coords*100/total_rows:.1f}%)")
    print(f"  Successfully assigned to geography: {total_assigned:,} ({total_assigned*100/rows_with_coords:.1f}% of those with coordinates)")
    print(f"  With coordinates but unassigned: {rows_with_coords_unassigned:,} ({rows_with_coords_unassigned*100/rows_with_coords:.1f}% of those with coordinates)")
    return data

def standardize_geography_columns(df):
    rename_dict = {}
    for col in df.columns:
        if col.startswith('MSOA') and col.endswith('CD'): rename_dict[col] = 'msoa_code'
        elif col.startswith('MSOA') and col.endswith('NM'): rename_dict[col] = 'msoa_name'
        elif col.startswith('LSOA') and col.endswith('CD'): rename_dict[col] = 'lsoa_code'
        elif col.startswith('LSOA') and col.endswith('NM'): rename_dict[col] = 'lsoa_name'
    return df.rename(columns=rename_dict)

def add_hocl_msoa_names(df, msoa_names_path):
    msoa_names_df = pd.read_csv(msoa_names_path)
    msoa_names_df.rename(columns={"msoa21cd": "msoa_code", "msoa21hclnm": "msoa_hocl_name"}, inplace=True)
    df = df.merge(msoa_names_df[["msoa_code", "msoa_hocl_name"]], on="msoa_code", how="left")
    return df