
"""
Experimental AntiFraudster model with historical behavioral features.

Run from the repository root:
    python -m src.train_evaluate_ml

This experiment does not modify the existing production pipeline.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "experimental_transactions.csv"
RESULTS_PATH = ROOT / "data" / "ml_evaluation_results.csv"

TARGET = "is_fraud"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run "
            "python -m src.generate_experimental_data first."
        )

    data = pd.read_csv(DATA_PATH)

    required = {
        "transaction_id",
        "timestamp",
        "sender_id",
        "receiver_id",
        "amount",
        "location_id",
        TARGET,
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="raise"
    )
    data["amount"] = pd.to_numeric(data["amount"], errors="raise")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="raise").astype(int)

    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only 0 and 1.")

    return data.sort_values(
        ["timestamp", "transaction_id"]
    ).reset_index(drop=True)


def add_historical_features(data):
    """
    Build features using only earlier transactions.

    Labels are never used to construct the features.
    """
    data = data.copy()
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["is_night"] = (
        (data["hour"] < 6) | (data["hour"] >= 22)
    ).astype(int)

    historical_average = []
    historical_count = []
    historical_max = []
    seconds_since_previous = []
    known_recipient = []

    # Track earlier transactions per sender in chronological order.
    sender_amounts = {}
    sender_times = {}
    sender_recipients = {}

    for row in data.itertuples(index=False):
        sender = row.sender_id
        receiver = row.receiver_id
        amount = float(row.amount)
        timestamp = row.timestamp

        amounts = sender_amounts.get(sender, [])
        times = sender_times.get(sender, [])
        recipients = sender_recipients.get(sender, set())

        # Features are computed BEFORE adding the current transaction.
        historical_count.append(len(amounts))
        historical_average.append(
            sum(amounts) / len(amounts) if amounts else float("nan")
        )
        historical_max.append(max(amounts) if amounts else float("nan"))

        seconds_since_previous.append(
            (timestamp - times[-1]).total_seconds()
            if times else float("nan")
        )
        known_recipient.append(int(receiver in recipients))

        sender_amounts.setdefault(sender, []).append(amount)
        sender_times.setdefault(sender, []).append(timestamp)
        sender_recipients.setdefault(sender, set()).add(receiver)

    data["sender_history_count"] = historical_count
    data["sender_history_avg_amount"] = historical_average
    data["sender_history_max_amount"] = historical_max
    data["seconds_since_previous"] = seconds_since_previous
    data["known_recipient"] = known_recipient

    # Compare amount to the sender's historical average.
    data["amount_to_history_ratio"] = (
        data["amount"] / data["sender_history_avg_amount"].where(
            data["sender_history_avg_amount"] > 0
        )
    ).clip(upper=100)

    # Current amount compared with the sender's previous maximum.
    data["amount_above_history_max"] = (
        data["amount"] > data["sender_history_max_amount"]
    ).astype(int)

    return data


def main():
    data = add_historical_features(load_data())

    # Exclude identifiers, target labels, and synthetic scenario labels.
    excluded = {
        "transaction_id",
        "sender_id",
        "receiver_id",
        "timestamp",
        "scenario",
        "fraud_type",
        "label",
        "target",
        TARGET,
    }

    feature_columns = [
        col for col in data.columns if col not in excluded
    ]

    X = data[feature_columns].copy()
    y = data[TARGET].copy()

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numeric_columns = [
        col for col in X.columns if col not in categorical_columns
    ]

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median"))
                    ]
                ),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    # Keep the final 25% of transactions as a future test period.
    split_index = int(len(data) * 0.75)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training period ends: {data.iloc[split_index - 1]['timestamp']}")
    print(f"Testing period starts: {data.iloc[split_index]['timestamp']}")
    

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)


    # Show which features the Random Forest relies on most.
    classifier = model.named_steps["classifier"]
    transformed_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance = pd.DataFrame({
        "feature": transformed_names,
        "importance": classifier.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTOP 10 FEATURE IMPORTANCES")
    print(importance.head(10).to_string(index=False))


    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    print("\n" + "=" * 58)
    print("ANTIFRAUDSTER — HISTORICAL FEATURE EXPERIMENT")
    print("=" * 58)
    print(f"Dataset rows:             {len(data)}")
    print(f"Training rows:            {len(X_train)}")
    print(f"Testing rows:             {len(X_test)}")
    print(f"Feature columns:          {len(feature_columns)}")
    print(f"Actual fraud in test set: {int(y_test.sum())}")
    print(f"True positives:           {tp}")
    print(f"False positives:          {fp}")
    print(f"False negatives:          {fn}")
    print(f"True negatives:           {tn}")
    print("-" * 58)
    print(f"Accuracy:                 {accuracy:.2%}")
    print(f"Precision:                {precision:.2%}")
    print(f"Recall:                   {recall:.2%}")
    print(f"F1-score:                 {f1:.2%}")
    print(f"False-positive rate:      {false_positive_rate:.2%}")
    print("-" * 58)
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    output = data.loc[X_test.index].copy()
    output["actual_fraud"] = y_test
    output["predicted_fraud"] = predictions
    output["fraud_probability"] = model.predict_proba(X_test)[:, 1]
    output.to_csv(RESULTS_PATH, index=False)

    print(f"Predictions saved to: {RESULTS_PATH}")
    print("\nReminder: these results use synthetic data only.")


if __name__ == "__main__":
    main()
