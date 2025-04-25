##### Part 5. MULTI-Year Analysis Features######

# Calculate consistency of deductible structure for each member across years
yearly_deductibles = df_clean.groupby(['MVDID', 'Plan_year'])['DEDUCTIBLEAMOUNT'].sum().reset_index()

# Calculate variation across years for the same member
member_years = yearly_deductibles.groupby('MVDID')['Plan_year'].count().reset_index()
member_years.rename(columns={'Plan_year': 'member_years_in_data'}, inplace=True)
df_clean = df_clean.merge(member_years, on='MVDID', how='left')

# Only calculate year-over-year metrics for members with multiple years
multi_year_members = yearly_deductibles[yearly_deductibles['MVDID'].isin(
    member_years[member_years['member_years_in_data'] > 1]['MVDID'])]

if len(multi_year_members) > 0:
    member_year_variation = multi_year_members.groupby('MVDID').agg(
        deduct_std=('DEDUCTIBLEAMOUNT', 'std'),
        deduct_mean=('DEDUCTIBLEAMOUNT', 'mean')
    ).reset_index()

    member_year_variation['year_over_year_variation'] = member_year_variation['deduct_std'] / member_year_variation['deduct_mean'].replace(0, np.nan)
    member_year_variation = member_year_variation[['MVDID', 'year_over_year_variation']]

    # Merge back
    df_clean = df_clean.merge(member_year_variation, on='MVDID', how='left')
else:
    df_clean['year_over_year_variation'] = np.nan

# Track if deductible_types changed for a member across years
deduct_type_by_year = df_clean.groupby(['MVDID', 'Plan_year'])['deductible_types'].first().reset_index()

# Calculate number of unique deductible types a member had over all years
member_deduct_types = deduct_type_by_year.groupby('MVDID')['deductible_types'].nunique().reset_index()
member_deduct_types.rename(columns={'deductible_types': 'deductible_type_changes'}, inplace=True)

# Merge back - higher values may indicate plan changes
df_clean = df_clean.merge(member_deduct_types, on='MVDID', how='left')

# Flag if this year's deductible type is different from previous year for this member
deduct_type_by_year = deduct_type_by_year.sort_values(['MVDID', 'Plan_year'])
deduct_type_by_year['prev_deductible_type'] = deduct_type_by_year.groupby('MVDID')['deductible_types'].shift(1)
deduct_type_by_year['plan_changed'] = (~(deduct_type_by_year['deductible_types'] == deduct_type_by_year['prev_deductible_type'])).astype(int)
deduct_type_by_year = deduct_type_by_year[['MVDID', 'Plan_year', 'plan_changed']]

# Merge back
df_clean = df_clean.merge(deduct_type_by_year, on=['MVDID', 'Plan_year'], how='left')
df_clean['plan_changed'] = df_clean['plan_changed'].fillna(0).astype(int)

# Simplified inflation adjustment (use actual healthcare inflation data if available)
inflation_factors = {
    2016: 1.00,
    2017: 1.03,
    2018: 1.06,
    2019: 1.10,
    2020: 1.13,
    2021: 1.18,
    2022: 1.25,
    2023: 1.31
}

# Create adjusted deductible amount
df_clean['inflation_factor'] = df_clean['Plan_year'].map(inflation_factors)
df_clean['adjusted_deductible'] = df_clean['DEDUCTIBLEAMOUNT'] / df_clean['inflation_factor']

# Find first month with claims for each member in each year
first_claim_month = df_clean.groupby(['MVDID', 'Plan_year'])['service_month'].min().reset_index()
first_claim_month.rename(columns={'service_month': 'first_claim_month'}, inplace=True)

# Calculate average first claim month across years
avg_first_month = first_claim_month.groupby('MVDID')['first_claim_month'].mean().reset_index()
avg_first_month.rename(columns={'first_claim_month': 'avg_first_claim_month'}, inplace=True)

# Merge both
df_clean = df_clean.merge(first_claim_month, on=['MVDID', 'Plan_year'], how='left')
df_clean = df_clean.merge(avg_first_month, on='MVDID', how='left')

# Find the month when coinsurance first appeared each year (if available)
coins_members = df_clean[df_clean['has_coinsurance'] == 1]
if len(coins_members) > 0:
    coins_months = coins_members.groupby(['MVDID', 'Plan_year'])['service_month'].min().reset_index()
    coins_months.rename(columns={'service_month': 'first_coins_month'}, inplace=True)

    # Merge back
    df_clean = df_clean.merge(coins_months, on=['MVDID', 'Plan_year'], how='left')

    # Calculate for members with coinsurance in multiple years
    multi_year_coins = coins_months.groupby('MVDID')['Plan_year'].count()
    multi_year_coins = multi_year_coins[multi_year_coins > 1].index

    if len(multi_year_coins) > 0:
        coins_month_std = coins_months[coins_months['MVDID'].isin(multi_year_coins)].groupby('MVDID')['first_coins_month'].std().reset_index()
        coins_month_std.rename(columns={'first_coins_month': 'coins_month_variation'}, inplace=True)

        # Merge back
        df_clean = df_clean.merge(coins_month_std, on='MVDID', how='left')
    else:
        df_clean['coins_month_variation'] = np.nan
else:
    df_clean['first_coins_month'] = np.nan
    df_clean['coins_month_variation'] = np.nan

# Check if family composition changed over years
family_years = df_clean.groupby('family_id')['Plan_year'].nunique().reset_index()
family_years.rename(columns={'Plan_year': 'family_years_in_data'}, inplace=True)
df_clean = df_clean.merge(family_years, on='family_id', how='left')

multi_year_families = family_years[family_years['family_years_in_data'] > 1]['family_id']

if len(multi_year_families) > 0:
    family_size_by_year = df_clean[df_clean['family_id'].isin(multi_year_families)].groupby(['family_id', 'Plan_year'])['family_size'].first().reset_index()
    family_size_by_year = family_size_by_year.sort_values(['family_id', 'Plan_year'])

    # Calculate if family size changed
    family_size_by_year['prev_family_size'] = family_size_by_year.groupby('family_id')['family_size'].shift(1)
    family_size_by_year['family_size_changed'] = (~(family_size_by_year['family_size'] == family_size_by_year['prev_family_size'])).astype(int)
    family_size_by_year = family_size_by_year[['family_id', 'Plan_year', 'family_size_changed']]

    # Merge back
    df_clean = df_clean.merge(family_size_by_year, on=['family_id', 'Plan_year'], how='left')
    df_clean['family_size_changed'] = df_clean['family_size_changed'].fillna(0).astype(int)
else:
    df_clean['family_size_changed'] = 0

# Calculate what percentage of yearly deductible is paid in first quarter
df_clean['is_first_quarter'] = (df_clean['service_quarter'] == 1).astype(int)

q1_deductibles = df_clean[df_clean['is_first_quarter'] == 1].groupby(['MVDID', 'Plan_year'])['DEDUCTIBLEAMOUNT'].sum().reset_index()
q1_deductibles.rename(columns={'DEDUCTIBLEAMOUNT': 'q1_deductible'}, inplace=True)

yearly_deductibles = df_clean.groupby(['MVDID', 'Plan_year'])['DEDUCTIBLEAMOUNT'].sum().reset_index()
yearly_deductibles.rename(columns={'DEDUCTIBLEAMOUNT': 'yearly_deductible'}, inplace=True)

# Merge both
deductible_timing = q1_deductibles.merge(yearly_deductibles, on=['MVDID', 'Plan_year'], how='right')
deductible_timing['q1_deductible'] = deductible_timing['q1_deductible'].fillna(0)
deductible_timing['q1_deductible_pct'] = deductible_timing['q1_deductible'] / deductible_timing['yearly_deductible'].replace(0, np.nan)
deductible_timing = deductible_timing[['MVDID', 'Plan_year', 'q1_deductible_pct']]

# Merge back
df_clean = df_clean.merge(deductible_timing, on=['MVDID', 'Plan_year'], how='left')

################     PART 6. Company and Plan-Level Features       ##################
# Analyze deductible patterns at the company level
company_deduct_stats = df_clean.groupby(['COMPANY_KEY', 'Plan_year']).agg({
    'DEDUCTIBLEAMOUNT': ['mean', 'std', 'median'],
    'member_total_deductible': ['mean', 'std', 'median'],
    'family_total_deductible': ['mean', 'std', 'median'],
    'family_size': 'mean',
    'unique_family_members': 'mean'
})

# Flatten the column hierarchy
company_deduct_stats.columns = ['_'.join(col).strip() for col in company_deduct_stats.columns.values]
company_deduct_stats = company_deduct_stats.reset_index()

# Calculate coefficient of variation
company_deduct_stats['company_deductible_cv'] = company_deduct_stats['DEDUCTIBLEAMOUNT_std'] / company_deduct_stats['DEDUCTIBLEAMOUNT_mean'].replace(0, np.nan)
company_deduct_stats['company_member_deductible_cv'] = company_deduct_stats['member_total_deductible_std'] / company_deduct_stats['member_total_deductible_mean'].replace(0, np.nan)
company_deduct_stats['company_family_deductible_cv'] = company_deduct_stats['family_total_deductible_std'] / company_deduct_stats['family_total_deductible_mean'].replace(0, np.nan)

# Keep only computed columns to merge back
company_cols = ['COMPANY_KEY', 'Plan_year', 'company_deductible_cv', 'company_member_deductible_cv',
                'company_family_deductible_cv', 'DEDUCTIBLEAMOUNT_mean', 'member_total_deductible_mean',
                'family_total_deductible_mean', 'family_size_mean']
company_deduct_stats = company_deduct_stats[company_cols]

# Merge back to main dataset
df_clean = df_clean.merge(company_deduct_stats, on=['COMPANY_KEY', 'Plan_year'], how='left')

# Calculate deductible amount relative to company average
df_clean['deductible_to_company_avg'] = df_clean['DEDUCTIBLEAMOUNT'] / df_clean['DEDUCTIBLEAMOUNT_mean'].replace(0, np.nan)
df_clean['member_deductible_to_company_avg'] = df_clean['member_total_deductible'] / df_clean['member_total_deductible_mean'].replace(0, np.nan)
df_clean['family_deductible_to_company_avg'] = df_clean['family_total_deductible'] / df_clean['family_total_deductible_mean'].replace(0, np.nan)
