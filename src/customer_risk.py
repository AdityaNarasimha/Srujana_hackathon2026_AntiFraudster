
import pandas as pd


def assess_customer_risk(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Add customer-profile risk signals to synthetic transactions."""

    required_txn = {
        "transaction_id",
        "sender_id",
        "amount",
    }
    required_customers = {
        "customer_id",
        "customer_type",
        "expected_monthly_outflow",
        "usual_transaction_count",
        "kyc_status",
    }

    if missing := required_txn - set(transactions.columns):
        raise ValueError(f"Missing transaction columns: {sorted(missing)}")

    if missing := required_customers - set(customers.columns):
        raise ValueError(f"Missing customer columns: {sorted(missing)}")

    if transactions["transaction_id"].duplicated().any():
        raise ValueError("Transaction IDs must be unique.")

    if customers["customer_id"].duplicated().any():
        raise ValueError("Customer IDs must be unique.")

    result = transactions.copy()
    result["amount"] = pd.to_numeric(result["amount"], errors="raise")

    if result["amount"].isna().any() or (result["amount"] < 0).any():
        raise ValueError("Transaction amounts must be non-negative numbers.")

    profile = customers[
        [
            "customer_id",
            "customer_type",
            "expected_monthly_outflow",
            "usual_transaction_count",
            "kyc_status",
        ]
    ].rename(columns={"customer_id": "sender_id"})

    result = result.merge(
        profile,
        on="sender_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    result["profile_available"] = result["_merge"].eq("both")
    result = result.drop(columns="_merge")

    # Compare a single transfer against the profile's expected monthly outflow.
    # This is a screening signal, not a determination of fraud.
    result["outflow_ratio"] = (
        result["amount"]
        / result["expected_monthly_outflow"].where(
            result["expected_monthly_outflow"] > 0
        )
    )

    result["profile_risk_points"] = (
        result["outflow_ratio"].ge(0.5).fillna(False).astype(int) * 25
    )

    # Missing customer profiles and pending demo KYC are review signals.
    # Neither is proof of fraudulent activity.
    result["profile_review_reason"] = ""

    result.loc[
        ~result["profile_available"],
        "profile_review_reason",
    ] = "Customer profile unavailable"

    pending = (
        result["profile_available"]
        & result["kyc_status"].eq("pending_demo")
    )
    result.loc[
        pending,
        "profile_review_reason",
    ] = "Demo KYC status pending"

    result["profile_review_required"] = (
        ~result["profile_available"] | pending
    )

    return result
