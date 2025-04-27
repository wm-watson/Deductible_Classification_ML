import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, \
    precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import os

# Define the directory path where your data is stored
output_dir = r"R:\GraduateStudents\WatsonWilliamP\ML_DL_Deduc_Classification\Data"

# Load the previously saved claim-level dataset
csv_path = os.path.join(output_dir, "claim_level_features.csv")
print(f"Loading data from {csv_path}...")

# Load the dataset with optimized dtypes to reduce memory usage
claim_df = pd.read_csv(csv_path, low_memory=False)

# Step 1: Analyze the imbalance
print("Analyzing target distribution...")
target_column = 'DEDUCTIBLE_CATEGORY_first'
mask = claim_df[target_column].isin(['Aggregate', 'Embedded'])
filtered_df = claim_df[mask].copy()
filtered_df['target'] = (filtered_df[target_column] == 'Embedded').astype(int)

print(f"Original dataset size: {len(filtered_df)}")
print(f"Target distribution: {filtered_df['target'].value_counts()}")
print(
    f"Aggregate class: {filtered_df['target'].value_counts()[0]} samples ({filtered_df['target'].value_counts()[0] / len(filtered_df) * 100:.2f}%)")
print(
    f"Embedded class: {filtered_df['target'].value_counts()[1]} samples ({filtered_df['target'].value_counts()[1] / len(filtered_df) * 100:.2f}%)")
print(f"Imbalance ratio: 1:{filtered_df['target'].value_counts()[1] / filtered_df['target'].value_counts()[0]:.1f}")

# Step 2: Identify columns to exclude
print("\nIdentifying columns to exclude...")
exclude_patterns = ['CLAIMNUMBER', 'MVDID', 'SUBSCRIBERID', 'COMPANY_KEY', 'family_id',
                    'deductible_type', 'DEDUCTIBLE_CATEGORY', 'deductible_types_first',
                    'unique_deductible_types']
exclude_columns = [col for col in filtered_df.columns if any(pattern in col for pattern in exclude_patterns)]

# Also exclude datetime columns and target
datetime_columns = [col for col in filtered_df.columns if pd.api.types.is_datetime64_dtype(filtered_df[col])]
exclude_columns.extend(datetime_columns)
exclude_columns.extend(['target'])

print(f"Excluding {len(exclude_columns)} columns: {exclude_columns[:10]}...")
if len(exclude_columns) > 10:
    print(f"...and {len(exclude_columns) - 10} more")

# Step 3: Select features for analysis
feature_columns = [col for col in filtered_df.columns
                   if col not in exclude_columns
                   and pd.api.types.is_numeric_dtype(filtered_df[col])]
print(f"\nSelected {len(feature_columns)} numeric features for analysis")

# Step 4: Handle missing values and prepare data
print("\nPreparing features...")
X = filtered_df[feature_columns].copy()
y = filtered_df['target']

# Check for any columns with too many missing values
missing_percentages = X.isnull().mean() * 100
high_missing = missing_percentages[missing_percentages > 50].index.tolist()
if high_missing:
    print(f"Removing {len(high_missing)} columns with >50% missing values")
    X = X.drop(columns=high_missing)
    feature_columns = [col for col in feature_columns if col not in high_missing]

# Step 5: Create a balanced sample for feature selection
print("\nCreating a balanced sample for feature selection...")

# Options for handling imbalance:
# 1. Undersampling the majority class
# 2. Oversampling the minority class with SMOTE
# 3. Combination approach

# For initial feature selection, we'll use a balanced sample
# Strategy: Undersample majority class to get a manageable dataset size

# Determine sample sizes
n_minority = filtered_df['target'].value_counts()[0]  # Aggregate class count
sample_size_majority = min(n_minority * 5, filtered_df['target'].value_counts()[1])  # Cap at 5x minority

print(f"Using {n_minority} Aggregate samples and {sample_size_majority} Embedded samples for feature selection")

# Create mask for Aggregate samples (all of them)
agg_mask = filtered_df['target'] == 0
agg_indices = filtered_df[agg_mask].index

# Randomly sample from Embedded class
emb_mask = filtered_df['target'] == 1
emb_indices = filtered_df[emb_mask].index
np.random.seed(42)
sampled_emb_indices = np.random.choice(emb_indices, size=sample_size_majority, replace=False)

# Combine indices for balanced sample
balanced_indices = np.concatenate([agg_indices, sampled_emb_indices])
balanced_df = filtered_df.loc[balanced_indices].copy()

print(f"Balanced sample size: {len(balanced_df)}")
print(f"New target distribution: {balanced_df['target'].value_counts()}")

# Use the balanced dataset for feature selection
X_balanced = balanced_df[feature_columns].copy()
y_balanced = balanced_df['target']

# Handle missing values
print("Handling missing values...")
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X_balanced),
    columns=feature_columns
)

# Normalize features
print("Normalizing features...")
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_imputed),
    columns=feature_columns
)

# Step 6: Split balanced data for feature selection validation
print("\nSplitting balanced data for validation...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Step 7: Feature selection using multiple methods
print("\nPerforming feature selection using multiple methods...")

# Method 1: Univariate Feature Selection (ANOVA F-value)
print("\n1. Univariate Feature Selection (ANOVA)...")
k_features = min(50, len(feature_columns))
selector_f = SelectKBest(f_classif, k=k_features)
selector_f.fit(X_train, y_train)
anova_scores = pd.DataFrame({
    'Feature': feature_columns,
    'ANOVA_Score': selector_f.scores_,
    'ANOVA_P_Value': selector_f.pvalues_
})
anova_scores = anova_scores.sort_values('ANOVA_Score', ascending=False)
print(f"Top 10 features by ANOVA F-test:")
print(anova_scores.head(10))

# Method 2: Mutual Information
print("\n2. Mutual Information...")
selector_mi = SelectKBest(mutual_info_classif, k=k_features)
selector_mi.fit(X_train, y_train)
mi_scores = pd.DataFrame({
    'Feature': feature_columns,
    'MI_Score': selector_mi.scores_
})
mi_scores = mi_scores.sort_values('MI_Score', ascending=False)
print(f"Top 10 features by Mutual Information:")
print(mi_scores.head(10))

# Method 3: Random Forest Feature Importance
print("\n3. Random Forest Feature Importance...")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_importances = pd.DataFrame({
    'Feature': feature_columns,
    'RF_Importance': rf.feature_importances_
})
rf_importances = rf_importances.sort_values('RF_Importance', ascending=False)
print(f"Top 10 features by Random Forest Importance:")
print(rf_importances.head(10))

# Method 4: Logistic Regression with L1 regularization
print("\n4. Logistic Regression Coefficients (L1 regularization)...")
log_reg = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, class_weight='balanced', random_state=42)
log_reg.fit(X_train, y_train)
log_reg_coefs = pd.DataFrame({
    'Feature': feature_columns,
    'LR_Coefficient': np.abs(log_reg.coef_[0])
})
log_reg_coefs = log_reg_coefs.sort_values('LR_Coefficient', ascending=False)
print(f"Top 10 features by Logistic Regression L1:")
print(log_reg_coefs.head(10))

# Step 8: Aggregate results from all methods
print("\nAggregating results from all methods...")
# Normalize scores for each method
for df, score_col in [(anova_scores, 'ANOVA_Score'),
                      (mi_scores, 'MI_Score'),
                      (rf_importances, 'RF_Importance'),
                      (log_reg_coefs, 'LR_Coefficient')]:
    df[f'{score_col}_Normalized'] = df[score_col] / df[score_col].max()

# Merge all results
all_scores = anova_scores[['Feature', 'ANOVA_Score_Normalized']]
all_scores = all_scores.merge(
    mi_scores[['Feature', 'MI_Score_Normalized']],
    on='Feature', how='left'
)
all_scores = all_scores.merge(
    rf_importances[['Feature', 'RF_Importance_Normalized']],
    on='Feature', how='left'
)
all_scores = all_scores.merge(
    log_reg_coefs[['Feature', 'LR_Coefficient_Normalized']],
    on='Feature', how='left'
)

# Calculate average normalized score
all_scores['Average_Score'] = all_scores[[
    'ANOVA_Score_Normalized',
    'MI_Score_Normalized',
    'RF_Importance_Normalized',
    'LR_Coefficient_Normalized'
]].mean(axis=1)

# Rank features by average score
all_scores = all_scores.sort_values('Average_Score', ascending=False)
print("\nTop 20 features based on average importance across all methods:")
print(all_scores.head(20))

# Step 9: Save feature importance results
feature_importance_path = os.path.join(output_dir, "feature_importance_agg_vs_emb.csv")
all_scores.to_csv(feature_importance_path, index=False)
print(f"\nFeature importance results saved to {feature_importance_path}")

# Step 10: Visualize top features
plt.figure(figsize=(12, 8))
top_20 = all_scores.head(20).copy()
top_20['Feature'] = top_20['Feature'].str.replace('_', ' ').str.capitalize()
sns.barplot(x='Average_Score', y='Feature', data=top_20)
plt.title('Top 20 Features for Distinguishing Aggregate vs. Embedded Deductibles')
plt.tight_layout()
viz_path = os.path.join(output_dir, "top_features_agg_vs_emb.png")
plt.savefig(viz_path)
print(f"Visualization saved to {viz_path}")

# Step 11: Evaluate top features on imbalanced data using cross-validation
print("\nEvaluating feature importance with cross-validation on imbalanced data...")

# Prepare full dataset using same preprocessing steps
X_full = filtered_df[feature_columns].copy()
y_full = filtered_df['target']

# Handle missing values
X_full_imputed = pd.DataFrame(
    imputer.transform(X_full),  # Use transform, not fit_transform
    columns=feature_columns
)

# Normalize features
X_full_scaled = pd.DataFrame(
    scaler.transform(X_full_imputed),  # Use transform, not fit_transform
    columns=feature_columns
)

# Select feature sets for evaluation
feature_sets = {
    'top_5': all_scores.head(5)['Feature'].tolist(),
    'top_10': all_scores.head(10)['Feature'].tolist(),
    'top_20': all_scores.head(20)['Feature'].tolist(),
    'top_50': all_scores.head(50)['Feature'].tolist() if len(all_scores) >= 50 else all_scores['Feature'].tolist()
}


# Define the evaluation function using SMOTE and cross-validation
def evaluate_features(X, y, features, n_splits=5):
    X_selected = X[features]

    # Metrics to track
    aucs = []
    avg_precisions = []
    recalls = []
    specificities = []

    # Define cross-validation strategy
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Create a pipeline with SMOTE and Random Forest
    # We'll use SMOTE for minority class oversampling
    # SMOTE should only be applied to the training data in each fold
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)

    fold = 1
    for train_idx, test_idx in skf.split(X_selected, y):
        X_train_fold, X_test_fold = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        # Apply SMOTE to training data
        smote = SMOTE(random_state=42, sampling_strategy=0.5)  # Create minority class at 50% of majority
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)

        # Train model on resampled data
        model.fit(X_train_resampled, y_train_resampled)

        # Predict on test set
        y_pred_proba = model.predict_proba(X_test_fold)[:, 1]
        y_pred = model.predict(X_test_fold)

        # Calculate metrics
        auc = roc_auc_score(y_test_fold, y_pred_proba)
        avg_precision = average_precision_score(y_test_fold, y_pred_proba)

        # Calculate confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred).ravel()
        recall = tp / (tp + fn)  # Sensitivity
        specificity = tn / (tn + fp)

        # Store metrics
        aucs.append(auc)
        avg_precisions.append(avg_precision)
        recalls.append(recall)
        specificities.append(specificity)

        print(f"Fold {fold}: AUC={auc:.4f}, Avg Precision={avg_precision:.4f}, "
              f"Sensitivity={recall:.4f}, Specificity={specificity:.4f}")
        fold += 1

    # Return average metrics
    return {
        'AUC': np.mean(aucs),
        'AUC_std': np.std(aucs),
        'Avg_Precision': np.mean(avg_precisions),
        'Avg_Precision_std': np.std(avg_precisions),
        'Sensitivity': np.mean(recalls),
        'Sensitivity_std': np.std(recalls),
        'Specificity': np.mean(specificities),
        'Specificity_std': np.std(specificities)
    }


# Evaluate each feature set
results = []
for name, features in feature_sets.items():
    print(f"\nEvaluating {name} features: {len(features)} features")

    # Get CV results
    cv_results = evaluate_features(X_full_scaled, y_full, features)

    # Add to results
    results.append({
        'Feature_Set': name,
        'Num_Features': len(features),
        'AUC': cv_results['AUC'],
        'AUC_std': cv_results['AUC_std'],
        'Avg_Precision': cv_results['Avg_Precision'],
        'Avg_Precision_std': cv_results['Avg_Precision_std'],
        'Sensitivity': cv_results['Sensitivity'],
        'Sensitivity_std': cv_results['Sensitivity_std'],
        'Specificity': cv_results['Specificity'],
        'Specificity_std': cv_results['Specificity_std']
    })

# Save results
results_df = pd.DataFrame(results)
results_path = os.path.join(output_dir, "feature_evaluation_imbalanced_cv.csv")
results_df.to_csv(results_path, index=False)
print(f"\nCross-validation results saved to {results_path}")

# Plot performance metrics
plt.figure(figsize=(12, 8))
x = range(len(results_df))
width = 0.2
metrics = ['AUC', 'Avg_Precision', 'Sensitivity', 'Specificity']
colors = ['blue', 'green', 'orange', 'red']

for i, metric in enumerate(metrics):
    plt.bar([pos + width * i for pos in x], results_df[metric], width,
            yerr=results_df[f'{metric}_std'], label=metric, color=colors[i], capsize=5)

plt.xlabel('Feature Set')
plt.ylabel('Score')
plt.title('Model Performance with Different Feature Sets (5-fold CV)')
plt.xticks([pos + width * 1.5 for pos in x], results_df['Feature_Set'])
plt.legend()
plt.ylim(0, 1.0)
perf_path = os.path.join(output_dir, "feature_performance_imbalanced_cv.png")
plt.savefig(perf_path)
print(f"Performance chart saved to {perf_path}")

# Save the top features for future use
top_features = all_scores.head(20)['Feature'].tolist()
with open(os.path.join(output_dir, "top_20_features.txt"), 'w') as f:
    for feature in top_features:
        f.write(f"{feature}\n")

print("\nFeature selection analysis complete!")
print(f"Top 20 features: {top_features}")
