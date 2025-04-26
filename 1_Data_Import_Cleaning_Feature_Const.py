import pandas as pd
import os
import numpy as np

# Filepath to data
file_path = 'R:\\GraduateStudents\\WatsonWilliamP\\Deductible_Project\\Deductible_Project\\Data\\final.csv'

## Output----
output_path = os.path.join(os.path.dirname(file_path), 'final_10_pct_sample.csv')

# Chunk parameteres
chunk_size = 100000
sample_frac = 0.1
np.random.seed(42)

print(f"Starting to read data from: {file_path}")
print(f"Using chunk size: {chunk_size} rows")
print(f"Taking a {sample_frac * 100}% random sample")

# Initialize container for samples
sample_chunks = []

# Read the file in chunks and randomly sample
try:
    # First read the header to see column names
    header_df = pd.read_csv(file_path, nrows=0)
    print(f"Found {len(header_df.columns)} columns in the dataset")

    # Process file in chunks
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)):
        # Sample the chunk
        sample = chunk.sample(frac=sample_frac, random_state=42)
        sample_chunks.append(sample)

        # Print progress info
        rows_processed = (i + 1) * chunk_size
        print(f"Processed chunk {i + 1}: approximately {rows_processed:,} rows")

    # Combine all sample chunks
    df_sample = pd.concat(sample_chunks, ignore_index=True)

    # Basic information about the sample
    print(f"\nSample creation complete!")
    print(f"Sample size: {len(df_sample):,} rows")
    print(f"Memory usage: {df_sample.memory_usage().sum() / 1e6:.2f} MB")

    # Save the sample
    df_sample.to_csv(output_path, index=False)
    print(f"Saved sample to: {output_path}")

    # Display first few rows to verify
    print("\nFirst 5 rows of the sample:")
    print(df_sample.head())

except Exception as e:
    print(f"Error during processing: {str(e)}")

# Path to the sampled data
sample_path = 'R:\\GraduateStudents\\WatsonWilliamP\\Deductible_Project\\Deductible_Project\\Data\\final_10_pct_sample.csv'
clean_path = os.path.join(os.path.dirname(sample_path), 'final_10_pct_clean.csv')

# Read the sampled data
print(f"Loading data from: {sample_path}")
df = pd.read_csv(sample_path)
print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Select only the needed columns
columns_to_keep = [
    'MVDID',
    'SUBSCRIBERID',
    'COMPANY_KEY',
    'PATIENTGENDER',
    'age_at_plan_year_start',
    'CLAIMNUMBER',
    'BILLEDAMOUNT',
    'ALLOWEDAMOUNT',
    'PAIDAMOUNT',
    'COINSURANCEAMOUNT',
    'COPAYAMOUNT',
    'DEDUCTIBLEAMOUNT',
    'CODEVALUE',
    'PROCEDURECODE',
    'SERVICEFROMDATE',
    'Plan_year',
    'family_size',
    'DEDUCTIBLE_CATEGORY',
    'unique_deductible_types',
    'deductible_types'
]

# Create a cleaned dataframe with only the necessary columns
df_clean = df[columns_to_keep].copy()

# Convert ID columns to string
id_cols = ['MVDID', 'SUBSCRIBERID', 'COMPANY_KEY', 'CLAIMNUMBER']
df_clean[id_cols] = df_clean[id_cols].astype(str)

# Convert categorical columns
cat_cols = ['PATIENTGENDER', 'DEDUCTIBLE_CATEGORY', 'deductible_types']
df_clean[cat_cols] = df_clean[cat_cols].astype('category')

# Convert numeric columns
num_cols = ['BILLEDAMOUNT', 'ALLOWEDAMOUNT', 'PAIDAMOUNT', 'COINSURANCEAMOUNT',
            'COPAYAMOUNT', 'DEDUCTIBLEAMOUNT', 'age_at_plan_year_start',
            'family_size', 'unique_deductible_types']
df_clean[num_cols] = df_clean[num_cols].apply(pd.to_numeric, errors='coerce')

# Convert date column
df_clean['SERVICEFROMDATE'] = pd.to_datetime(df_clean['SERVICEFROMDATE'], errors='coerce')

# Convert code columns
code_cols = ['CODEVALUE', 'PROCEDURECODE']
df_clean[code_cols] = df_clean[code_cols].astype(str)

# Count missing values per column
missing_count = df_clean.isnull().sum()

# Calculate percentage of missing values
missing_percent = (missing_count / len(df_clean) * 100).round(2)

# Combine count and percentage in a DataFrame
missing_df = pd.DataFrame({
    'Count': missing_count,
    'Percent': missing_percent
})

# Only show columns with missing values
missing_df = missing_df[missing_df['Count'] > 0].sort_values('Percent', ascending=False)

#results
print(missing_df)


# Get original row count
original_count = len(df_clean)

# Remove rows with missing values in deductible-related columns
df_clean = df_clean.dropna(subset=['DEDUCTIBLE_CATEGORY', 'unique_deductible_types', 'deductible_types'])

# Calculate and display the number and percentage of rows removed
rows_removed = original_count - len(df_clean)
percent_removed = round((rows_removed / original_count * 100), 2)  # Use round() function

print(f"Rows removed: {rows_removed:,} ({percent_removed}%)")
print(f"Remaining rows: {len(df_clean):,}")

df_clean.info()

# Check for duplicates based on all columns
duplicate_rows = df_clean.duplicated().sum()
print(f"Complete duplicates: {duplicate_rows:,} rows")

# Check for duplicates based on key identifiers (adjust columns as needed)
key_columns = ['MVDID', 'CLAIMNUMBER', 'CODEVALUE', 'PROCEDURECODE', 'SERVICEFROMDATE']
duplicate_claim_lines = df_clean.duplicated(subset=key_columns).sum()
print(f"Duplicate claim lines: {duplicate_claim_lines:,} rows")

# Remove duplicate claim lines keeping 1st
df_clean = df_clean.drop_duplicates(subset=key_columns)

###################Begin Constructing Variables##########################################


######## PART 1 - Claim level, family size, company-specific ######################

# Define aggregation functions for each column
agg_dict = {
    'MVDID': 'first',
    'SUBSCRIBERID': 'first',
    'COMPANY_KEY': 'first',
    'PATIENTGENDER': 'first',
    'age_at_plan_year_start': 'first',
    'Plan_year': 'first',
    'family_size': 'first',
    'DEDUCTIBLE_CATEGORY': 'first',
    'deductible_types': 'first',
    'unique_deductible_types': 'first',
    'BILLEDAMOUNT': ['sum', 'mean', 'max', 'count'],
    'ALLOWEDAMOUNT': ['sum', 'mean', 'max'],
    'PAIDAMOUNT': ['sum', 'mean', 'max'],
    'COINSURANCEAMOUNT': ['sum', 'mean', 'max'],
    'COPAYAMOUNT': ['sum', 'mean', 'max'],
    'DEDUCTIBLEAMOUNT': ['sum', 'mean', 'max'],
    'SERVICEFROMDATE': ['min', 'max']
}

# Perform aggregation
claim_df = df_clean.groupby('CLAIMNUMBER').agg(agg_dict)

# Flatten columns
claim_df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in claim_df.columns]
claim_df = claim_df.reset_index()

# Calculate claim-specific features
claim_df['total_claim_lines'] = claim_df['BILLEDAMOUNT_count']
claim_df['claim_duration_days'] = (claim_df['SERVICEFROMDATE_max'] - claim_df['SERVICEFROMDATE_min']).dt.days
claim_df['claim_duration_days'] = claim_df['claim_duration_days'].fillna(0).clip(lower=0)

# Create family identifier and features
claim_df['family_id'] = claim_df['SUBSCRIBERID_first']

# Get family member counts (efficiently using claim-level data)
family_member_counts = df_clean.groupby('SUBSCRIBERID')['MVDID'].nunique()
claim_df['unique_family_members'] = claim_df['family_id'].map(family_member_counts)
claim_df['family_size_matches_members'] = (claim_df['family_size_first'] == claim_df['unique_family_members']).astype(int)

# Company-specific features
company_deductible_types = claim_df.groupby(['COMPANY_KEY_first', 'Plan_year_first'])['deductible_types_first'].nunique().reset_index()
company_deductible_types.columns = ['COMPANY_KEY_first', 'Plan_year_first', 'company_deductible_types_count']

# Verify merge
row_count_before = len(claim_df)
claim_df = claim_df.merge(company_deductible_types, on=['COMPANY_KEY_first', 'Plan_year_first'], how='left')
assert row_count_before == len(claim_df), f"Merge changed row count: {row_count_before} to {len(claim_df)}"

# Flag companies with multiple deductible types
claim_df['company_has_multiple_deductible_types'] = (claim_df['company_deductible_types_count'] > 1).astype(int)

######## Part 2. Financial Ratios & Patterns at Claim Level ############################

# Calculate financial ratios at claim level
claim_df['deductible_to_allowed_ratio'] = claim_df['DEDUCTIBLEAMOUNT_sum'] / claim_df['ALLOWEDAMOUNT_sum'].replace(0, np.nan)
claim_df['patient_responsibility'] = claim_df['COINSURANCEAMOUNT_sum'] + claim_df['COPAYAMOUNT_sum'] + claim_df['DEDUCTIBLEAMOUNT_sum']
claim_df['deductible_to_responsibility_ratio'] = claim_df['DEDUCTIBLEAMOUNT_sum'] / claim_df['patient_responsibility'].replace(0, np.nan)
claim_df['deductible_to_billed_ratio'] = claim_df['DEDUCTIBLEAMOUNT_sum'] / claim_df['BILLEDAMOUNT_sum'].replace(0, np.nan)
claim_df['patient_share'] = claim_df['patient_responsibility'] / claim_df['ALLOWEDAMOUNT_sum'].replace(0, np.nan)
claim_df['insurance_covered_ratio'] = claim_df['PAIDAMOUNT_sum'] / claim_df['ALLOWEDAMOUNT_sum'].replace(0, np.nan)

# Flag claims with specific cost sharing components
claim_df['has_coinsurance'] = (claim_df['COINSURANCEAMOUNT_sum'] > 0).astype(int)
claim_df['has_copay'] = (claim_df['COPAYAMOUNT_sum'] > 0).astype(int)
claim_df['has_deductible'] = (claim_df['DEDUCTIBLEAMOUNT_sum'] > 0).astype(int)

# Fill NaN values with 0 for better downstream handling
ratio_cols = [
    'deductible_to_allowed_ratio', 'deductible_to_responsibility_ratio',
    'deductible_to_billed_ratio', 'patient_share', 'insurance_covered_ratio'
]
claim_df[ratio_cols] = claim_df[ratio_cols].fillna(0)

######## Part 3. Time-based Feature Construction at Claim Level #########################

# Extract time-based features from the first service date of each claim
claim_df['service_year'] = claim_df['SERVICEFROMDATE_min'].dt.year
claim_df['service_month'] = claim_df['SERVICEFROMDATE_min'].dt.month
claim_df['service_quarter'] = claim_df['SERVICEFROMDATE_min'].dt.quarter
claim_df['service_day_of_week'] = claim_df['SERVICEFROMDATE_min'].dt.dayofweek
claim_df['service_day_of_year'] = claim_df['SERVICEFROMDATE_min'].dt.dayofyear
claim_df['is_beginning_of_year'] = (claim_df['service_month'] <= 3).astype(int)
claim_df['is_end_of_year'] = (claim_df['service_month'] >= 10).astype(int)

# Sort claims chronologically
claim_df_sorted = claim_df.sort_values(['MVDID_first', 'Plan_year_first', 'SERVICEFROMDATE_min'])

# Add claim sequence numbers at the claim level
claim_df_sorted['patient_claim_sequence'] = claim_df_sorted.groupby('MVDID_first').cumcount() + 1
claim_df_sorted['patient_plan_year_claim_sequence'] = claim_df_sorted.groupby(['MVDID_first', 'Plan_year_first']).cumcount() + 1

# Update the main dataframe
claim_df = claim_df_sorted.copy()

# Calculate coinsurance patterns at the claim level
claim_df['cumulative_coinsurance'] = claim_df.groupby(['MVDID_first', 'Plan_year_first'])['has_coinsurance'].cumsum()
claim_df['first_coinsurance'] = (claim_df['cumulative_coinsurance'] == 1) & (claim_df['has_coinsurance'] == 1)

# Flag if member ever has coinsurance in the plan year (calculated from claim-level data)
member_has_coins = claim_df.groupby(['MVDID_first', 'Plan_year_first'])['has_coinsurance'].max().reset_index()
member_has_coins.columns = ['MVDID_first', 'Plan_year_first', 'ever_has_coinsurance']

# Add merge check
row_count_before = len(claim_df)
claim_df = claim_df.merge(member_has_coins, on=['MVDID_first', 'Plan_year_first'], how='left')
assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

# For members who have coinsurance, identify claims before the first coinsurance claim
# Add a marker to identify claims that occur before the first coinsurance
claim_df['before_first_coinsurance'] = (
    (claim_df['ever_has_coinsurance'] == 1) &
    (claim_df['cumulative_coinsurance'] == 0)
).astype(int)

# Create a feature for deductible amount on claims before first coinsurance
claim_df['deductible_on_precoins_claim'] = np.where(
    claim_df['before_first_coinsurance'] == 1,
    claim_df['DEDUCTIBLEAMOUNT_sum'],
    0
)

# Create flag for members who never have coinsurance
claim_df['never_has_coinsurance'] = (claim_df['ever_has_coinsurance'] == 0).astype(int)

######## Part 4: Family Contribution Analysis at Claim Level #########################

# Create family identifier using subscriber ID
claim_df['family_id'] = claim_df['SUBSCRIBERID_first']

# Calculate total deductible paid by each claim
claim_df['claim_deductible_amount'] = claim_df['DEDUCTIBLEAMOUNT_sum']

# Calculate total deductible paid by each member within family for each plan year
# This uses claim-level data to calculate member and family totals
member_fam_deduct = claim_df.groupby(['family_id', 'MVDID_first', 'Plan_year_first'])['claim_deductible_amount'].sum().reset_index()
member_fam_deduct.columns = ['family_id', 'MVDID_first', 'Plan_year_first', 'member_fam_deduct_amount']

# Calculate family total deductible
family_deduct = member_fam_deduct.groupby(['family_id', 'Plan_year_first'])['member_fam_deduct_amount'].sum().reset_index()
family_deduct.columns = ['family_id', 'Plan_year_first', 'fam_total_deduct_amount']

# Merge family and member deductible information back to claims
row_count_before = len(claim_df)
claim_df = claim_df.merge(member_fam_deduct, on=['family_id', 'MVDID_first', 'Plan_year_first'], how='left')
assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

row_count_before = len(claim_df)
claim_df = claim_df.merge(family_deduct, on=['family_id', 'Plan_year_first'], how='left')
assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

# Calculate claim contribution to family deductible
claim_df['claim_to_member_deduct_ratio'] = claim_df['claim_deductible_amount'] / claim_df['member_fam_deduct_amount'].replace(0, np.nan)
claim_df['claim_to_family_deduct_ratio'] = claim_df['claim_deductible_amount'] / claim_df['fam_total_deduct_amount'].replace(0, np.nan)

# Calculate member's contribution percentage to family deductible
claim_df['member_deduct_contribution_pct'] = claim_df['member_fam_deduct_amount'] / claim_df['fam_total_deduct_amount'].replace(0, np.nan)

# Calculate the max individual deductible within family
family_max = member_fam_deduct.groupby(['family_id', 'Plan_year_first'])['member_fam_deduct_amount'].max().reset_index()
family_max.columns = ['family_id', 'Plan_year_first', 'family_max_member_deduct']

row_count_before = len(claim_df)
claim_df = claim_df.merge(family_max, on=['family_id', 'Plan_year_first'], how='left')
assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

# Calculate family structure metrics relevant to claims
claim_df['max_to_family_deduct_ratio'] = claim_df['family_max_member_deduct'] / claim_df['fam_total_deduct_amount'].replace(0, np.nan)
claim_df['deduct_per_family_member'] = claim_df['fam_total_deduct_amount'] / claim_df['family_size_first']
claim_df['member_to_avg_deduct_ratio'] = claim_df['member_fam_deduct_amount'] / claim_df['deduct_per_family_member'].replace(0, np.nan)

# Calculate family concentration measures
def calc_concentration(group):
    deduct_amounts = group['member_fam_deduct_amount'].values
    total_deduct = sum(deduct_amounts)

    if total_deduct == 0:
        return pd.Series({
            'deduct_concentration_index': 0,
            'top_member_deduct_share': 0 if len(deduct_amounts) > 0 else 1
        })

    # Calculate share of each member
    shares = deduct_amounts / total_deduct

    # Calculate Herfindahl index (sum of squared shares)
    herfindahl = np.sum(shares ** 2)

    # Calculate top member's share
    top_share = max(shares) if len(shares) > 0 else 1

    return pd.Series({
        'deduct_concentration_index': herfindahl,
        'top_member_deduct_share': top_share
    })

# Apply to each family-year
family_concentration = member_fam_deduct.groupby(['family_id', 'Plan_year_first']).apply(calc_concentration).reset_index()

# Merge concentration metrics
row_count_before = len(claim_df)
claim_df = claim_df.merge(family_concentration, on=['family_id', 'Plan_year_first'], how='left')
assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

# Calculate claim-specific timing features
# Identify claims with deductible
deduct_claims = claim_df[claim_df['claim_deductible_amount'] > 0].copy()

if not deduct_claims.empty:
    # Find first deductible claim for each member and family
    member_first_deduct = deduct_claims.groupby(['family_id', 'MVDID_first', 'Plan_year_first'])['SERVICEFROMDATE_min'].min().reset_index()
    member_first_deduct.columns = ['family_id', 'MVDID_first', 'Plan_year_first', 'first_deduct_date']

    family_first_deduct = member_first_deduct.groupby(['family_id', 'Plan_year_first'])['first_deduct_date'].min().reset_index()
    family_first_deduct.columns = ['family_id', 'Plan_year_first', 'family_first_deduct_date']

    # Merge both to claims
    row_count_before = len(claim_df)
    claim_df = claim_df.merge(member_first_deduct, on=['family_id', 'MVDID_first', 'Plan_year_first'], how='left')
    assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

    row_count_before = len(claim_df)
    claim_df = claim_df.merge(family_first_deduct, on=['family_id', 'Plan_year_first'], how='left')
    assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

    # Calculate claim timing relative to family's first deductible
    claim_df['claim_days_since_family_first_deduct'] = (claim_df['SERVICEFROMDATE_min'] - claim_df['family_first_deduct_date']).dt.days

    # Flag if this claim is the member's first deductible claim
    claim_df['is_member_first_deduct_claim'] = (claim_df['SERVICEFROMDATE_min'] == claim_df['first_deduct_date']).astype(int)

    # Flag if this claim is the family's first deductible claim
    claim_df['is_family_first_deduct_claim'] = (claim_df['SERVICEFROMDATE_min'] == claim_df['family_first_deduct_date']).astype(int)
else:
    # If no claims have deductibles, create placeholder columns
    claim_df['first_deduct_date'] = pd.NaT
    claim_df['family_first_deduct_date'] = pd.NaT
    claim_df['claim_days_since_family_first_deduct'] = np.nan
    claim_df['is_member_first_deduct_claim'] = 0
    claim_df['is_family_first_deduct_claim'] = 0

# Count members with deductible in each family
members_with_deduct = member_fam_deduct[member_fam_deduct['member_fam_deduct_amount'] > 0].groupby(['family_id', 'Plan_year_first']).size().reset_index()
members_with_deduct.columns = ['family_id', 'Plan_year_first', 'members_with_deduct']

row_count_before = len(claim_df)
claim_df = claim_df.merge(members_with_deduct, on=['family_id', 'Plan_year_first'], how='left')
assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

# Calculate percentage of family with deductible
claim_df['pct_family_with_deduct'] = claim_df['members_with_deduct'] / claim_df['family_size_first']

# Fill NAs with appropriate values
deduct_cols = [col for col in claim_df.columns if 'deduct' in col.lower() and claim_df[col].dtype in [np.float64, np.float32]]
for col in deduct_cols:
    if col.endswith('_ratio') or col.endswith('_pct') or col.endswith('_share'):
        claim_df[col] = claim_df[col].fillna(0)

######## Part 5. Multi-Year Analysis Features at Claim Level ######

# For plan changes and continuity features, we need to track member history
# First identify which years each claim member appears in
member_years = claim_df.groupby('MVDID_first')['Plan_year_first'].nunique().reset_index()
member_years.columns = ['MVDID_first', 'member_years_in_data']

# Merge check
row_count_before = len(claim_df)
claim_df = claim_df.merge(member_years, on='MVDID_first', how='left')
assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

# Track if deductible_types changed for a member across years
# Get one record per member-year with their deductible type
deduct_type_by_year = claim_df.groupby(['MVDID_first', 'Plan_year_first'])['deductible_types_first'].first().reset_index()
deduct_type_by_year.columns = ['MVDID_first', 'Plan_year_first', 'deductible_type_by_year']

# Calculate number of unique deductible types a member had over all years
member_deduct_types = deduct_type_by_year.groupby('MVDID_first')['deductible_type_by_year'].nunique().reset_index()
member_deduct_types.columns = ['MVDID_first', 'deductible_type_changes']

# Merge check
row_count_before = len(claim_df)
claim_df = claim_df.merge(member_deduct_types, on='MVDID_first', how='left')
assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

# Flag if this year's deductible type is different from previous year for this member
deduct_type_by_year = deduct_type_by_year.sort_values(['MVDID_first', 'Plan_year_first'])
deduct_type_by_year['prev_deductible_type'] = deduct_type_by_year.groupby('MVDID_first')['deductible_type_by_year'].shift(1)
deduct_type_by_year['plan_changed'] = (~(deduct_type_by_year['deductible_type_by_year'] == deduct_type_by_year['prev_deductible_type'])).astype(int)
deduct_type_by_year = deduct_type_by_year[['MVDID_first', 'Plan_year_first', 'plan_changed']]

# Merge check
row_count_before = len(claim_df)
claim_df = claim_df.merge(deduct_type_by_year, on=['MVDID_first', 'Plan_year_first'], how='left')
assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

# Fill missing values
claim_df['plan_changed'] = claim_df['plan_changed'].fillna(0).astype(int)

# Simplified inflation adjustment (use actual healthcare inflation data if available)
inflation_factors = {
    2016: 1.00, 2017: 1.03, 2018: 1.06, 2019: 1.10, 2020: 1.13,
    2021: 1.18, 2022: 1.25, 2023: 1.31, 2024: 1.36, 2025: 1.41
}

# Create adjusted deductible amount for claim
claim_df['inflation_factor'] = claim_df['Plan_year_first'].map(inflation_factors)
claim_df['adjusted_deductible'] = claim_df['DEDUCTIBLEAMOUNT_sum'] / claim_df['inflation_factor']

# Calculate claim timing relative to the year
claim_df['is_first_quarter'] = (claim_df['service_quarter'] == 1).astype(int)
claim_df['is_last_quarter'] = (claim_df['service_quarter'] == 4).astype(int)

# Find the first claim month for each member in each year
first_claims = claim_df.groupby(['MVDID_first', 'Plan_year_first']).agg(
    first_claim_month=('service_month', 'min'),
    first_claim_date=('SERVICEFROMDATE_min', 'min')
).reset_index()

# Merge check
row_count_before = len(claim_df)
claim_df = claim_df.merge(first_claims, on=['MVDID_first', 'Plan_year_first'], how='left')
assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

# Calculate days since first claim in plan year
claim_df['days_since_first_claim'] = (claim_df['SERVICEFROMDATE_min'] - claim_df['first_claim_date']).dt.days

# Flag if this is the member's first claim in the plan year
claim_df['is_first_claim_in_year'] = (claim_df['SERVICEFROMDATE_min'] == claim_df['first_claim_date']).astype(int)

# Create a feature for members with family size changes
if 'family_id' in claim_df.columns:
    # Get number of years each family appears in the data
    family_years = claim_df.groupby('family_id')['Plan_year_first'].nunique().reset_index()
    family_years.columns = ['family_id', 'family_years_in_data']

    # Merge family years information
    row_count_before = len(claim_df)
    claim_df = claim_df.merge(family_years, on='family_id', how='left')
    assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

    # Only analyze families with multiple years of data
    multi_year_families = family_years[family_years['family_years_in_data'] > 1]['family_id']

    if len(multi_year_families) > 0:
        # Get one family size record per family-year
        family_size_by_year = claim_df[claim_df['family_id'].isin(multi_year_families)].groupby(
            ['family_id', 'Plan_year_first'])['family_size_first'].first().reset_index()

        # Sort by family and year
        family_size_by_year = family_size_by_year.sort_values(['family_id', 'Plan_year_first'])

        # Calculate if family size changed from previous year
        family_size_by_year['prev_family_size'] = family_size_by_year.groupby('family_id')['family_size_first'].shift(1)
        family_size_by_year['family_size_changed'] = (~(family_size_by_year['family_size_first'] ==
                                                     family_size_by_year['prev_family_size'])).astype(int)
        family_size_by_year = family_size_by_year[['family_id', 'Plan_year_first', 'family_size_changed']]

        # Merge back to claim data
        row_count_before = len(claim_df)
        claim_df = claim_df.merge(family_size_by_year, on=['family_id', 'Plan_year_first'], how='left')
        assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

        # Fill missing values
        claim_df['family_size_changed'] = claim_df['family_size_changed'].fillna(0).astype(int)
    else:
        claim_df['family_size_changed'] = 0

################ PART 6. Company and Plan-Level Features at Claim Level ##################

# Calculate company-level statistics using claim-level data
company_deduct_stats = claim_df.groupby(['COMPANY_KEY_first', 'Plan_year_first']).agg({
    'DEDUCTIBLEAMOUNT_sum': ['mean', 'std', 'median'],
    'member_fam_deduct_amount': ['mean', 'std', 'median'],
    'fam_total_deduct_amount': ['mean', 'std', 'median'],
    'family_size_first': 'mean'
})

# Flatten the column hierarchy
company_deduct_stats.columns = ['_'.join(col).strip() for col in company_deduct_stats.columns.values]
company_deduct_stats = company_deduct_stats.reset_index()

# Calculate coefficient of variation
company_deduct_stats['company_deductible_cv'] = (
    company_deduct_stats['DEDUCTIBLEAMOUNT_sum_std'] /
    company_deduct_stats['DEDUCTIBLEAMOUNT_sum_mean'].replace(0, np.nan)
)
company_deduct_stats['company_member_deductible_cv'] = (
    company_deduct_stats['member_fam_deduct_amount_std'] /
    company_deduct_stats['member_fam_deduct_amount_mean'].replace(0, np.nan)
)
company_deduct_stats['company_family_deductible_cv'] = (
    company_deduct_stats['fam_total_deduct_amount_std'] /
    company_deduct_stats['fam_total_deduct_amount_mean'].replace(0, np.nan)
)

# Keep only computed columns to merge back
company_cols = [
    'COMPANY_KEY_first', 'Plan_year_first',
    'company_deductible_cv', 'company_member_deductible_cv', 'company_family_deductible_cv',
    'DEDUCTIBLEAMOUNT_sum_mean', 'member_fam_deduct_amount_mean',
    'fam_total_deduct_amount_mean', 'family_size_first_mean'
]
company_deduct_stats = company_deduct_stats[company_cols]

# Merge back to main dataset with check
row_count_before = len(claim_df)
claim_df = claim_df.merge(company_deduct_stats, on=['COMPANY_KEY_first', 'Plan_year_first'], how='left')
assert row_count_before == len(claim_df), f"Merge changed row count from {row_count_before} to {len(claim_df)}"

# Calculate claim deductible relative to company average
claim_df['deductible_to_company_avg'] = (
    claim_df['DEDUCTIBLEAMOUNT_sum'] /
    claim_df['DEDUCTIBLEAMOUNT_sum_mean'].replace(0, np.nan)
)
claim_df['member_deductible_to_company_avg'] = (
    claim_df['member_fam_deduct_amount'] /
    claim_df['member_fam_deduct_amount_mean'].replace(0, np.nan)
)
claim_df['family_deductible_to_company_avg'] = (
    claim_df['fam_total_deduct_amount'] /
    claim_df['fam_total_deduct_amount_mean'].replace(0, np.nan)
)

# Fill NaN values with 0 for ratio columns
ratio_cols = [
    'company_deductible_cv', 'company_member_deductible_cv', 'company_family_deductible_cv',
    'deductible_to_company_avg', 'member_deductible_to_company_avg', 'family_deductible_to_company_avg'
]
claim_df[ratio_cols] = claim_df[ratio_cols].fillna(0)

# Define the output path using your specified directory
output_dir = r"R:\GraduateStudents\WatsonWilliamP\ML_DL_Deduc_Classification\Data"


################ PART 7. Demographics and Comorbidity Features ##################

# GENDER FEATURES
claim_df['is_female'] = (claim_df['PATIENTGENDER_first'].str.upper() == 'F').astype(int)
claim_df['is_male'] = (claim_df['PATIENTGENDER_first'].str.upper() == 'M').astype(int)

# Count gender distribution within families
gender_by_family = claim_df.groupby(['family_id', 'PATIENTGENDER_first']).size().unstack(fill_value=0)
if 'F' in gender_by_family.columns and 'M' in gender_by_family.columns:
    gender_by_family['total'] = gender_by_family['F'] + gender_by_family['M']
    gender_by_family['female_ratio'] = gender_by_family['F'] / gender_by_family['total']
    gender_by_family['family_gender_composition'] = 'mixed'
    gender_by_family.loc[gender_by_family['female_ratio'] == 1, 'family_gender_composition'] = 'all_female'
    gender_by_family.loc[gender_by_family['female_ratio'] == 0, 'family_gender_composition'] = 'all_male'
    gender_by_family = gender_by_family.reset_index()

    # Merge to claim level
    row_count_before = len(claim_df)
    claim_df = claim_df.merge(
        gender_by_family[['family_id', 'female_ratio', 'family_gender_composition']],
        on='family_id', how='left'
    )
    assert row_count_before == len(claim_df), "Merge changed row count"

# AGE FEATURES
claim_df['age'] = claim_df['age_at_plan_year_start_first']
claim_df['age_category'] = pd.cut(
    claim_df['age'],
    bins=[0, 18, 26, 35, 50, 64, 100],
    labels=['0-18', '19-26', '27-35', '36-50', '51-64', '65+']
)

# Family age distribution
family_ages = claim_df.groupby('family_id')['age'].agg(['min', 'max', 'mean', 'std']).reset_index()
family_ages.columns = ['family_id', 'family_youngest_age', 'family_oldest_age', 'family_mean_age', 'family_age_std']
family_ages['family_age_range'] = family_ages['family_oldest_age'] - family_ages['family_youngest_age']
family_ages['family_has_children'] = (family_ages['family_youngest_age'] < 18).astype(int)
family_ages['family_has_seniors'] = (family_ages['family_oldest_age'] >= 65).astype(int)

# Create family life stage variable
family_ages['family_life_stage'] = 'other'
family_ages.loc[(family_ages['family_has_children'] == 1), 'family_life_stage'] = 'has_children'
family_ages.loc[(family_ages['family_mean_age'] < 30) & (family_ages['family_has_children'] == 0), 'family_life_stage'] = 'young_adults'
family_ages.loc[(family_ages['family_mean_age'] >= 30) & (family_ages['family_mean_age'] < 50) & (family_ages['family_has_children'] == 0), 'family_life_stage'] = 'middle_age'
family_ages.loc[(family_ages['family_mean_age'] >= 50) & (family_ages['family_has_seniors'] == 0), 'family_life_stage'] = 'mature'
family_ages.loc[(family_ages['family_has_seniors'] == 1), 'family_life_stage'] = 'senior'

row_count_before = len(claim_df)
claim_df = claim_df.merge(family_ages, on='family_id', how='left')
assert row_count_before == len(claim_df), "Merge changed row count"

# COMORBIDITY FEATURES
try:
    from pyhealth.medcode import InnerMap
    from pyhealth.medcode.utils import charlson_comorbidity

    icd2comorbid = InnerMap.load("icd10cm")

    # Process claim-level diagnosis codes
    icd_codes = claim_df.groupby('CLAIMNUMBER')['CODEVALUE_first'].apply(list).reset_index()

    # Function to calculate Charlson comorbidity score and identify conditions
    def get_charlson_features(codes):
        # Filter and standardize codes
        valid_codes = [str(code).strip().upper() for code in codes if isinstance(code, (str, int, float))]

        if not valid_codes:
            # Default values if no valid codes
            result = {
                'charlson_score': 0,
                'charlson_conditions': 0
            }
            return pd.Series(result)

        # Get Charlson comorbidities
        charlson_dict = charlson_comorbidity(valid_codes, code_type="icd10cm", use_mapping=icd2comorbid)

        # Calculate score and condition count
        result = {
            'charlson_score': sum(charlson_dict.values()),
            'charlson_conditions': sum(1 for val in charlson_dict.values() if val > 0)
        }

        return pd.Series(result)

    # Apply to each claim
    charlson_results = icd_codes.apply(lambda row: get_charlson_features(row['CODEVALUE_first']), axis=1)

    # Add claim number
    charlson_results['CLAIMNUMBER'] = icd_codes['CLAIMNUMBER']

    # Merge to claim data
    row_count_before = len(claim_df)
    claim_df = claim_df.merge(charlson_results, on='CLAIMNUMBER', how='left')
    assert row_count_before == len(claim_df), "Merge changed row count"

    # Fill NaN values
    comorbidity_cols = [col for col in charlson_results.columns if col != 'CLAIMNUMBER']
    claim_df[comorbidity_cols] = claim_df[comorbidity_cols].fillna(0)

    # Create simplified comorbidity category
    claim_df['comorbidity_level'] = pd.cut(
        claim_df['charlson_score'],
        bins=[-0.1, 0, 1, 2, 100],
        labels=['none', 'mild', 'moderate', 'severe']
    )

except (ImportError, ModuleNotFoundError):
    # Handle case where pyhealth is not available
    claim_df['charlson_score'] = 0
    claim_df['charlson_conditions'] = 0
    claim_df['comorbidity_level'] = 'none'


# ONE-HOT ENCODING FOR CATEGORICAL VARIABLES
# This creates binary indicator columns for each category

# For age categories
age_dummies = pd.get_dummies(claim_df['age_category'], prefix='age', drop_first=False)
claim_df = pd.concat([claim_df, age_dummies], axis=1)

# For family gender composition
gender_comp_dummies = pd.get_dummies(claim_df['family_gender_composition'], prefix='gender_comp', drop_first=False)
claim_df = pd.concat([claim_df, gender_comp_dummies], axis=1)

# For family life stage
life_stage_dummies = pd.get_dummies(claim_df['family_life_stage'], prefix='life_stage', drop_first=False)
claim_df = pd.concat([claim_df, life_stage_dummies], axis=1)

# For comorbidity level
comorbidity_dummies = pd.get_dummies(claim_df['comorbidity_level'], prefix='comorbidity', drop_first=False)
claim_df = pd.concat([claim_df, comorbidity_dummies], axis=1)

# Set the output directory to the specified network path
output_dir = r"R:\GraduateStudents\WatsonWilliamP\ML_DL_Deduc_Classification\Data"

# Save the claim-level dataset
csv_path = os.path.join(output_dir, "claim_level_features.csv")
claim_df.to_csv(csv_path, index=False)
print(f"Claim-level dataset with {claim_df.shape[1]} features saved to {csv_path}")

