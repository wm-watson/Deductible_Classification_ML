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


######## PART 1 - Claimline, family size, companypspecific#######################

# Make claimline number counter
df_clean['claim_line_number'] = df_clean.groupby('CLAIMNUMBER').cumcount() + 1

# Calculate total lines per claim and add to each row
claim_counts = df_clean.groupby('CLAIMNUMBER').size().to_dict()
df_clean['total_claim_lines'] = df_clean['CLAIMNUMBER'].map(claim_counts)

df_clean.info()

# Create family identifier
df_clean['family_id'] = df_clean['SUBSCRIBERID']

# Count unique individuals in each family
family_member_counts = df_clean.groupby('family_id')['MVDID'].nunique().to_dict()
df_clean['unique_family_members'] = df_clean['family_id'].map(family_member_counts)

# Flag whether family_size matches actual unique members seen in claims
df_clean['family_size_matches_members'] = (df_clean['family_size'] == df_clean['unique_family_members']).astype(int)

# Company-specific
company_deductible_types = df_clean.groupby(['COMPANY_KEY', 'Plan_year'])['deductible_types'].nunique().reset_index()
company_deductible_types.rename(columns={'deductible_types': 'company_deductible_types_count'}, inplace=True)
df_clean = df_clean.merge(company_deductible_types, on=['COMPANY_KEY', 'Plan_year'], how='left')

# Flag companies with multiple deductible types in same year
df_clean['company_has_multiple_deductible_types'] = (df_clean['company_deductible_types_count'] > 1).astype(int)


######## Part 2. Financial Ratios & Patterns      ############################

# Financial ratios
df_clean['deductible_to_allowed_ratio'] = df_clean['DEDUCTIBLEAMOUNT'] / df_clean['ALLOWEDAMOUNT'].replace(0, np.nan)
df_clean['patient_responsibility'] = df_clean['COINSURANCEAMOUNT'] + df_clean['COPAYAMOUNT'] + df_clean['DEDUCTIBLEAMOUNT']
df_clean['deductible_to_responsibility_ratio'] = df_clean['DEDUCTIBLEAMOUNT'] / df_clean['patient_responsibility'].replace(0, np.nan)
df_clean['deductible_to_billed_ratio'] = df_clean['DEDUCTIBLEAMOUNT'] / df_clean['BILLEDAMOUNT'].replace(0, np.nan)
df_clean['patient_share'] = df_clean['patient_responsibility'] / df_clean['ALLOWEDAMOUNT'].replace(0, np.nan)
df_clean['insurance_covered_ratio'] = df_clean['PAIDAMOUNT'] / df_clean['ALLOWEDAMOUNT'].replace(0, np.nan)

# Flag claims where coinsurance is present - this indicates when deductible has been met
df_clean['has_coinsurance'] = (df_clean['COINSURANCEAMOUNT'] > 0).astype(int)
df_clean['has_copay'] = (df_clean['COPAYAMOUNT'] > 0).astype(int)
df_clean['has_deductible'] = (df_clean['DEDUCTIBLEAMOUNT'] > 0).astype(int)

# Calculate total deductible paid by each member for each plan year
member_deductibles = df_clean.groupby(['MVDID', 'Plan_year'])['DEDUCTIBLEAMOUNT'].sum().reset_index()
member_deductibles.rename(columns={'DEDUCTIBLEAMOUNT': 'member_total_deductible'}, inplace=True)

# Merge back to main dataframe
df_clean = df_clean.merge(member_deductibles, on=['MVDID', 'Plan_year'], how='left')

# Calculate average financials per claim
claim_financials = df_clean.groupby('CLAIMNUMBER').agg({
    'BILLEDAMOUNT': 'sum',
    'ALLOWEDAMOUNT': 'sum',
    'PAIDAMOUNT': 'sum',
    'DEDUCTIBLEAMOUNT': 'sum',
    'COINSURANCEAMOUNT': 'sum',
    'COPAYAMOUNT': 'sum'
}).reset_index()

claim_financials['claim_deductible_to_allowed_ratio'] = claim_financials['DEDUCTIBLEAMOUNT'] / claim_financials['ALLOWEDAMOUNT'].replace(0, np.nan)
claim_financials['claim_patient_responsibility'] = claim_financials['DEDUCTIBLEAMOUNT'] + claim_financials['COINSURANCEAMOUNT'] + claim_financials['COPAYAMOUNT']
claim_financials['claim_patient_share'] = claim_financials['claim_patient_responsibility'] / claim_financials['ALLOWEDAMOUNT'].replace(0, np.nan)

# Keep cal columns
claim_financials = claim_financials[['CLAIMNUMBER', 'claim_deductible_to_allowed_ratio', 'claim_patient_responsibility', 'claim_patient_share']]
df_clean = df_clean.merge(claim_financials, on='CLAIMNUMBER', how='left')


######## Part 2. Time-based Feature Construction#########################

# Extract time-based features
df_clean['service_year'] = df_clean['SERVICEFROMDATE'].dt.year
df_clean['service_month'] = df_clean['SERVICEFROMDATE'].dt.month
df_clean['service_quarter'] = df_clean['SERVICEFROMDATE'].dt.quarter
df_clean['service_day_of_week'] = df_clean['SERVICEFROMDATE'].dt.dayofweek
df_clean['service_day_of_year'] = df_clean['SERVICEFROMDATE'].dt.dayofyear
df_clean['is_beginning_of_year'] = (df_clean['service_month'] <= 3).astype(int)
df_clean['is_end_of_year'] = (df_clean['service_month'] >= 10).astype(int)

# Sort claims chronologically
df_sorted = df_clean.sort_values(['MVDID', 'Plan_year', 'SERVICEFROMDATE'])

# Add sequence numbers
df_sorted['patient_claim_sequence'] = df_sorted.groupby('MVDID').cumcount() + 1
df_sorted['patient_plan_year_claim_sequence'] = df_sorted.groupby(['MVDID', 'Plan_year']).cumcount() + 1
df_clean = df_sorted.copy()

# Flag the first claim with coinsurance for each member in each plan year
df_clean['cumulative_coinsurance'] = df_clean.groupby(['MVDID', 'Plan_year'])['has_coinsurance'].cumsum()
df_clean['first_coinsurance'] = (df_clean['cumulative_coinsurance'] == 1) & (df_clean['has_coinsurance'] == 1)

# Flag if member ever has coinsurance in the plan year
member_has_coins = df_clean.groupby(['MVDID', 'Plan_year'])['has_coinsurance'].max().reset_index()
member_has_coins.rename(columns={'has_coinsurance': 'ever_has_coinsurance'}, inplace=True)
df_clean = df_clean.merge(member_has_coins, on=['MVDID', 'Plan_year'], how='left')

# For members who have coinsurance, calculate deductible before first coinsurance
before_coinsurance = df_clean.loc[
    (df_clean['ever_has_coinsurance'] == 1) &  # Only consider members who eventually have coinsurance
    (df_clean.groupby(['MVDID', 'Plan_year'])['cumulative_coinsurance'].transform('cumsum') == 0)
    # Before first coinsurance
    ]

if len(before_coinsurance) > 0:
    deduct_before_coins = before_coinsurance.groupby(['MVDID', 'Plan_year'])['DEDUCTIBLEAMOUNT'].sum().reset_index()
    deduct_before_coins.rename(columns={'DEDUCTIBLEAMOUNT': 'deductible_before_coinsurance'}, inplace=True)

    # Merge back
    df_clean = df_clean.merge(deduct_before_coins, on=['MVDID', 'Plan_year'], how='left')
else:
    df_clean['deductible_before_coinsurance'] = np.nan

# For members without coinsurance, set their deductible_before_coinsurance to their total deductible
no_coins_members = df_clean[df_clean['ever_has_coinsurance'] == 0]
if len(no_coins_members) > 0:
    no_coins_deduct = no_coins_members.groupby(['MVDID', 'Plan_year'])['DEDUCTIBLEAMOUNT'].sum().reset_index()
    no_coins_deduct.rename(columns={'DEDUCTIBLEAMOUNT': 'no_coins_total_deductible'}, inplace=True)

    # Merge back
    df_clean = df_clean.merge(no_coins_deduct, on=['MVDID', 'Plan_year'], how='left')

    # Fill in deductible_before_coinsurance with total deductible for no-coinsurance members
    df_clean.loc[df_clean['ever_has_coinsurance'] == 0, 'deductible_before_coinsurance'] = df_clean[
        'no_coins_total_deductible']

    # Drop the temporary column
    df_clean = df_clean.drop(columns=['no_coins_total_deductible'])

# Create flag for members who never have coinsurance
df_clean['never_has_coinsurance'] = (df_clean['ever_has_coinsurance'] == 0).astype(int)

# For members with coinsurance, calculate the ratio of deductible before coinsurance to total deductible
# This calculation should be done at member-year level to avoid duplicating calculations
member_deduct_ratio = df_clean.drop_duplicates(subset=['MVDID', 'Plan_year']).copy()
member_deduct_ratio['deductible_before_coins_ratio'] = np.nan  # Initialize with NaN

# Only calculate for members with coinsurance and positive total deductible
valid_mask = (member_deduct_ratio['ever_has_coinsurance'] == 1) & (member_deduct_ratio['member_total_deductible'] > 0)

# Ensure deductible_before_coinsurance is not NaN for the calculation
calc_mask = valid_mask & member_deduct_ratio['deductible_before_coinsurance'].notna()

if calc_mask.any():
    member_deduct_ratio.loc[calc_mask, 'deductible_before_coins_ratio'] = (
        member_deduct_ratio.loc[calc_mask, 'deductible_before_coinsurance'] /
        member_deduct_ratio.loc[calc_mask, 'member_total_deductible']
    )

# Keep only the columns we need
member_deduct_ratio = member_deduct_ratio[['MVDID', 'Plan_year', 'deductible_before_coins_ratio']]

# Merge back to the main dataframe
df_clean = df_clean.merge(member_deduct_ratio, on=['MVDID', 'Plan_year'], how='left')

df_clean.info()

######## Part 4: Individual-to-Family Contribution Analysis #########################

# Step 1: Calculate clean versions of family and member deductibles
# Calculate total deductible paid by each member within family for each plan year
member_fam_deduct = df_clean.groupby(['family_id', 'MVDID', 'Plan_year'])['DEDUCTIBLEAMOUNT'].sum().reset_index()
member_fam_deduct.rename(columns={'DEDUCTIBLEAMOUNT': 'member_fam_deduct_amount'}, inplace=True)

# Merge back to get individual contribution
df_clean = df_clean.merge(member_fam_deduct, on=['family_id', 'MVDID', 'Plan_year'], how='left')

# Calculate family total deductible (ensuring we avoid duplicates)
family_deduct_temp = member_fam_deduct.groupby(['family_id', 'Plan_year'])[
    'member_fam_deduct_amount'].sum().reset_index()
family_deduct_temp.rename(columns={'member_fam_deduct_amount': 'fam_total_deduct_amount'}, inplace=True)

# Merge family totals
df_clean = df_clean.merge(family_deduct_temp, on=['family_id', 'Plan_year'], how='left')

# Step 2: Calculate family contribution metrics
# Calculate member's contribution percentage to family deductible
df_clean['member_deduct_contribution_pct'] = df_clean['member_fam_deduct_amount'] / df_clean[
    'fam_total_deduct_amount'].replace(0, np.nan)

# Calculate the max individual deductible within family
family_max = member_fam_deduct.groupby(['family_id', 'Plan_year'])['member_fam_deduct_amount'].max().reset_index()
family_max.rename(columns={'member_fam_deduct_amount': 'family_max_member_deduct'}, inplace=True)
df_clean = df_clean.merge(family_max, on=['family_id', 'Plan_year'], how='left')


# Calculate variation metrics - using a function to avoid repetitive calculations
def calc_family_deduct_stats(x):
    if len(x) <= 1:
        return pd.Series({
            'fam_deduct_variation': 0,
            'fam_deduct_max_to_mean': 1,
            'fam_deduct_skew': 0,
            'fam_deduct_gini': 0
        })

    mean = x.mean()
    if mean == 0:
        return pd.Series({
            'fam_deduct_variation': 0,
            'fam_deduct_max_to_mean': 1 if len(x) > 0 else 0,
            'fam_deduct_skew': 0,
            'fam_deduct_gini': 0
        })

    # Coefficient of variation
    cv = x.std() / mean if mean > 0 else 0

    # Max to mean ratio
    max_to_mean = x.max() / mean if mean > 0 else 0

    # Simple skew calculation
    median = x.median()
    skew = (mean - median) / mean if mean > 0 else 0

    # Simple Gini coefficient (measure of inequality)
    sorted_x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * sorted_x)) / (n * np.sum(sorted_x)) if np.sum(sorted_x) > 0 else 0

    return pd.Series({
        'fam_deduct_variation': cv,
        'fam_deduct_max_to_mean': max_to_mean,
        'fam_deduct_skew': skew,
        'fam_deduct_gini': gini
    })


# Calculate the statistics at the family-year level
family_stats = member_fam_deduct.groupby(['family_id', 'Plan_year'])['member_fam_deduct_amount'].apply(
    calc_family_deduct_stats).reset_index()
df_clean = df_clean.merge(family_stats, on=['family_id', 'Plan_year'], how='left')

# Step 3: Calculate additional family structure metrics
# Ratio of max individual deductible to family total
df_clean['max_to_family_deduct_ratio'] = df_clean['family_max_member_deduct'] / df_clean[
    'fam_total_deduct_amount'].replace(0, np.nan)

# Ratio of total family deductible to family size
df_clean['deduct_per_family_member'] = df_clean['fam_total_deduct_amount'] / df_clean['family_size']

# Ratio of member's deductible to average deductible per family member
df_clean['member_to_avg_deduct_ratio'] = df_clean['member_fam_deduct_amount'] / df_clean[
    'deduct_per_family_member'].replace(0, np.nan)


# Step 4: Features that might indicate aggregate vs embedded structure
# In aggregate deductibles, we expect more concentrated spending (one person pays most of the deductible)
# In embedded deductibles, we expect more distributed spending (each person has their own deductible)

# Calculate concentration ratios (similar to Herfindahl index)
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
family_concentration = member_fam_deduct.groupby(['family_id', 'Plan_year']).apply(calc_concentration).reset_index()
df_clean = df_clean.merge(family_concentration, on=['family_id', 'Plan_year'], how='left')

# Step 5: Calculate features related to deductible payment patterns
# For aggregate deductibles, we might see family members waiting until one member meets most of the deductible
# For embedded deductibles, payment patterns should be more independent

# First, get each member's first deductible payment date in the plan year
member_first_deduct = df_clean[df_clean['DEDUCTIBLEAMOUNT'] > 0].groupby(['family_id', 'MVDID', 'Plan_year'])[
    'SERVICEFROMDATE'].min().reset_index()
member_first_deduct.rename(columns={'SERVICEFROMDATE': 'first_deduct_date'}, inplace=True)

# Get family's first deductible date
family_first_deduct = member_first_deduct.groupby(['family_id', 'Plan_year'])['first_deduct_date'].min().reset_index()
family_first_deduct.rename(columns={'first_deduct_date': 'family_first_deduct_date'}, inplace=True)

# Merge both back
df_clean = df_clean.merge(member_first_deduct, on=['family_id', 'MVDID', 'Plan_year'], how='left')
df_clean = df_clean.merge(family_first_deduct, on=['family_id', 'Plan_year'], how='left')

# Calculate days between family's first deductible and member's first deductible
df_clean['days_after_family_first_deduct'] = (
            df_clean['first_deduct_date'] - df_clean['family_first_deduct_date']).dt.days


# Calculate variance in first deductible dates within family
def calc_date_variance(dates):
    if len(dates) <= 1:
        return 0
    # Convert to days since minimum date
    min_date = min(dates)
    days_diff = [(date - min_date).days for date in dates]
    return np.var(days_diff)


family_date_var = member_first_deduct.groupby(['family_id', 'Plan_year'])['first_deduct_date'].apply(
    lambda x: calc_date_variance(x)).reset_index()
family_date_var.rename(columns={'first_deduct_date': 'family_deduct_date_variance'}, inplace=True)
df_clean = df_clean.merge(family_date_var, on=['family_id', 'Plan_year'], how='left')

# Step 6: Create features based on family member utilization patterns
# Calculate what percentage of family members have deductible payments
family_member_with_deduct = member_fam_deduct[member_fam_deduct['member_fam_deduct_amount'] > 0].groupby(
    ['family_id', 'Plan_year']).size().reset_index()
family_member_with_deduct.rename(columns={0: 'members_with_deduct'}, inplace=True)
df_clean = df_clean.merge(family_member_with_deduct, on=['family_id', 'Plan_year'], how='left')
df_clean['pct_family_with_deduct'] = df_clean['members_with_deduct'] / df_clean['family_size']

# Fill NAs with appropriate values for all created columns
deduct_cols = [col for col in df_clean.columns if
               'deduct' in col.lower() and df_clean[col].dtype in [np.float64, np.float32]]
for col in deduct_cols:
    if col.startswith('pct_') or col.endswith('_ratio') or col.endswith('_share'):
        df_clean[col] = df_clean[col].fillna(0)
