import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Define path
output_dir = r"R:\GraduateStudents\WatsonWilliamP\ML_DL_Deduc_Classification\Data"

# Load the feature importance results
feature_importance_path = os.path.join(output_dir, "feature_importance_agg_vs_emb.csv")
all_scores = pd.read_csv(feature_importance_path)

# Get top 20 features
top_features = all_scores.head(20)['Feature'].tolist()
print(f"Using these top 20 features: {top_features}")

# Load the preprocessed dataset
claim_df = pd.read_csv(os.path.join(output_dir, "claim_level_features.csv"), low_memory=False)

# Filter to just Aggregate and Embedded
mask = claim_df['DEDUCTIBLE_CATEGORY_first'].isin(['Aggregate', 'Embedded'])
filtered_df = claim_df[mask].copy()
filtered_df['target'] = (filtered_df['DEDUCTIBLE_CATEGORY_first'] == 'Embedded').astype(int)

# Get feature columns
feature_columns = [col for col in top_features if col in filtered_df.columns]
print(f"Found {len(feature_columns)}/{len(top_features)} features in the dataset")

# Prepare X and y
X = filtered_df[feature_columns].copy()
y = filtered_df['target']

# Apply the same preprocessing (imputation and scaling)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=feature_columns,
    index=X.index
)

# Normalize features
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_imputed),
    columns=feature_columns,
    index=X.index
)

# Split into train, validation, and test sets (60/20/20)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Save the prepared datasets for later use
train_data = pd.concat([X_train, y_train], axis=1)
val_data = pd.concat([X_val, y_val], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv(os.path.join(output_dir, "train_data_top20.csv"), index=False)
val_data.to_csv(os.path.join(output_dir, "val_data_top20.csv"), index=False)
test_data.to_csv(os.path.join(output_dir, "test_data_top20.csv"), index=False)

print("Datasets prepared and saved successfully!")
