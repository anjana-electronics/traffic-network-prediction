import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, classification_report


# ----------------------------------
# 1. Load datasets
# ----------------------------------

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

TARGET = "Target"
ID = "id"

if TARGET not in train.columns:
    raise ValueError("Target column is missing from train.csv")

if ID not in train.columns or ID not in test.columns:
    raise ValueError("The id column is missing")

print("\nTraining class distribution:")
print(train[TARGET].value_counts())
print("\nTraining class proportions:")
print(train[TARGET].value_counts(normalize=True).round(4))


# ----------------------------------
# 2. Prepare features and target
# ----------------------------------

# Do not use id as a feature.
feature_cols = [
    col for col in train.columns
    if col not in [TARGET, ID] and col in test.columns
]

X = train[feature_cols].copy()
y = train[TARGET].copy()
X_test = test[feature_cols].copy()

# Detect numeric and categorical columns.
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [
    col for col in X.columns if col not in numeric_cols
]

# Handle missing values and categorical data.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])


# ----------------------------------
# 3. Split data for validation
# ----------------------------------

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ----------------------------------
# 4. Compare models
# ----------------------------------

models = {
    "ExtraTrees_balanced": ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "ExtraTrees_unbalanced": ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight=None,
        random_state=42,
        n_jobs=-1
    ),

    "RandomForest_balanced": RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_score = -1
best_name = None

for name, model in models.items():
    print(f"\nTraining model: {name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    valid_predictions = pipeline.predict(X_valid)

    score = f1_score(
        y_valid,
        valid_predictions,
        average="macro",
        zero_division=0
    )

    results.append((name, score))

    print(f"Validation Macro F1: {score:.5f}")

    print(
        classification_report(
            y_valid,
            valid_predictions,
            zero_division=0
        )
    )

    if score > best_score:
        best_score = score
        best_name = name


# ----------------------------------
# 5. Display comparison
# ----------------------------------

print("\n========== MODEL COMPARISON ==========")

for name, score in sorted(
    results, key=lambda item: item[1], reverse=True
):
    print(f"{name}: Macro F1 = {score:.5f}")

print("\nBest validation model:", best_name)
print("Best validation Macro F1:", round(best_score, 5))


# ----------------------------------
# 6. Train selected model on all data
# ----------------------------------

best_model = models[best_name]

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", best_model)
])

print("\nTraining selected model on all training data...")
final_pipeline.fit(X, y)

test_predictions = final_pipeline.predict(X_test)


# ----------------------------------
# 7. Create submission.csv
# ----------------------------------

predictions = pd.DataFrame({
    ID: test[ID],
    TARGET: test_predictions
})

# Preserve the exact ID order required by sample_submission.csv.
submission = sample[[ID]].merge(
    predictions,
    on=ID,
    how="left",
    validate="one_to_one"
)

if submission[TARGET].isna().any():
    raise ValueError("Some submission IDs do not have predictions")

submission = submission[[ID, TARGET]]
submission.to_csv("submission.csv", index=False)

print("\nSubmission created successfully!")
print("Submission shape:", submission.shape)

print("\nPredicted class distribution:")
print(submission[TARGET].value_counts())

print("\nFirst five rows:")
print(submission.head())

print("\nSubmission columns:", submission.columns.tolist())