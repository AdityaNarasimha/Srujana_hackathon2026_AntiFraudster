
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from src.pipeline import analyze_transactions


TRANSACTIONS_PATH = "data/experimental_transactions.csv"
CUSTOMERS_PATH = "data/sample_customers.csv"
OUTPUT_PATH = "data/evaluation_results.csv"


def main():
    # 1. Load the original data and ground-truth labels.
    transactions = pd.read_csv(TRANSACTIONS_PATH)
    customers = pd.read_csv(CUSTOMERS_PATH)

    if transactions["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs in original data.")

    y_true_by_id = transactions.set_index("transaction_id")["is_fraud"]

    # 2. Run AntiFraudster.
    results = analyze_transactions(
        transactions.copy(),
        customers.copy()
    )

    # 3. Align results to the original transaction IDs.
    if results["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs in detector results.")

    results_by_id = results.set_index("transaction_id")

    if set(results_by_id.index) != set(y_true_by_id.index):
        raise ValueError(
            "Input and output transaction IDs do not match."
        )

    results_by_id = results_by_id.loc[y_true_by_id.index]

    # Ground truth comes ONLY from the original dataset.
    y_true = y_true_by_id.astype(int)
    y_pred = results_by_id["predicted_fraud"].astype(int)

    # 4. Calculate metrics.
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    fpr = fp / (fp + tn) if (fp + tn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    print("\n========== ANTIFRAUDSTER EVALUATION ==========")
    print(f"Total transactions:       {len(y_true)}")
    print(f"Actual fraud:             {int(y_true.sum())}")
    print(f"Actual legitimate:        {int((y_true == 0).sum())}")
    print(f"Transactions flagged:     {int(y_pred.sum())}")

    print("\n--- Confusion Matrix Counts ---")
    print(f"True positives (TP):      {tp}")
    print(f"False positives (FP):     {fp}")
    print(f"False negatives (FN):     {fn}")
    print(f"True negatives (TN):      {tn}")

    print("\n--- Metrics ---")
    print(f"Precision:                {precision:.2%}")
    print(f"Recall:                   {recall:.2%}")
    print(f"F1-score:                 {f1:.2%}")
    print(f"Accuracy:                 {accuracy:.2%}")
    print(f"False-positive rate:       {fpr:.2%}")
    print(f"Specificity:               {specificity:.2%}")

    print("\n--- Classification Report ---")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    ))

    # 5. Save aligned results, retaining original labels and scenarios.
    output = results_by_id.reset_index()
    output["actual_is_fraud"] = y_true.to_numpy()
    output["predicted_fraud"] = y_pred.to_numpy()
    output["scenario"] = transactions.set_index(
        "transaction_id"
    ).loc[y_true.index, "scenario"].to_numpy()

    output.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved corrected evaluation to {OUTPUT_PATH}")
    print("\nNote: synthetic-data evaluation only.")


if __name__ == "__main__":
    main()
