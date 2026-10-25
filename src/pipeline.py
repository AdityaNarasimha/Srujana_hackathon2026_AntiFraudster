
import pandas as pd

from src.detection import detect_fraud
from src.customer_risk import assess_customer_risk
from src.behavior_detection import detect_behavior
from src.location_detection import detect_location_anomalies
from src.network_detection import detect_network_risks
from src.spending_baseline import detect_spending_baseline
from src.timing_detection import detect_timing_anomalies
from src.recipient_relationship import detect_recipient_relationships


def analyze_transactions(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Run the AntiFraudster transaction-risk analysis pipeline."""

    required_columns = {
        "transaction_id",
        "sender_id",
        "receiver_id",
        "amount",
    }

    missing = required_columns - set(transactions.columns)
    if missing:
        raise ValueError(
            f"Missing transaction columns: {sorted(missing)}"
        )

    profiled = detect_fraud(transactions)

    # Behavioral analysis
    if "timestamp" in profiled.columns:
        profiled = detect_behavior(profiled)
    else:
        profiled["early_transfer_warning_flag"] = False
        profiled["rapid_transfer_flag"] = False
        profiled["rapid_burst_review_flag"] = False
        profiled["new_recipient_flag"] = False
        profiled["behavior_reasons"] = ""

    # Spending-pattern analysis
    profiled = detect_spending_baseline(profiled)

    # Timing analysis
    if "timestamp" in profiled.columns:
        profiled = detect_timing_anomalies(profiled)
    else:
        profiled["unusual_time_flag"] = False
        profiled["timing_reason"] = ""

    # Recipient relationship analysis
    if "timestamp" in profiled.columns:
        profiled = detect_recipient_relationships(profiled)
    else:
        profiled["recipient_amount_deviation_flag"] = False
        profiled["recipient_relationship_reason"] = ""

    # Customer profile risk
    profiled = assess_customer_risk(profiled, customers)

    # Behavioral risk points
    rapid_flag = (
        profiled["rapid_transfer_flag"]
        .fillna(False)
        .astype(bool)
    )
    new_recipient_flag = (
        profiled["new_recipient_flag"]
        .fillna(False)
        .astype(bool)
    )

    behavior_points = (
        rapid_flag.astype(int) * 25
        + new_recipient_flag.astype(int) * 10
    ).clip(upper=35)

    # Spending risk points
    spending_flag = (
        profiled["spending_change_flag"]
        .fillna(False)
        .astype(bool)
    )
    spending_points = spending_flag.astype(int) * 15

    # Timing risk points
    unusual_time_flag = (
        profiled["unusual_time_flag"]
        .fillna(False)
        .astype(bool)
    )
    timing_points = unusual_time_flag.astype(int) * 10

    # Recipient amount deviation points
    recipient_deviation_flag = (
        profiled["recipient_amount_deviation_flag"]
        .fillna(False)
        .astype(bool)
    )
    recipient_points = recipient_deviation_flag.astype(int) * 10

    # Combined risk score
    profiled["combined_risk_score"] = (
        profiled["risk_score"]
        + profiled["profile_risk_points"]
        + behavior_points
        + spending_points
        + timing_points
        + recipient_points
    ).clip(upper=100)

    profiled["risk_level"] = pd.cut(
        profiled["combined_risk_score"],
        bins=[-1, 24, 49, 74, 100],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)

    profiled["combined_risk_level"] = profiled["risk_level"]

    # Location analysis
    if "timestamp" in profiled.columns and "location_id" in profiled.columns:
        profiled = detect_location_anomalies(profiled)
    else:
        profiled["location_anomaly_flag"] = False
        profiled["location_reason"] = ""

    # Network analysis
    profiled = detect_network_risks(profiled)

    # Compliance review signal
    profiled["compliance_review_required"] = (
        profiled["profile_review_required"]
        .fillna(False)
        .astype(bool)
    )

    # Candidate fraud prediction:
    # includes the new causal early-warning signal.
    profiled["predicted_fraud"] = (
        profiled["combined_risk_score"].ge(50)
        | rapid_flag
        | profiled["early_transfer_warning_flag"]
            .fillna(False).astype(bool)
        | profiled["spending_baseline_review"]
            .fillna(False).astype(bool)
        | recipient_deviation_flag
        | profiled["location_anomaly_flag"]
            .fillna(False).astype(bool)
        | profiled["network_review_signal"]
            .fillna(False).astype(bool)
    )

    # Human review can be broader than the fraud prediction.
    profiled["needs_human_review"] = (
        profiled["combined_risk_score"].ge(50)
        | profiled["profile_review_required"]
            .fillna(False).astype(bool)
        | rapid_flag
        | new_recipient_flag
        | profiled["spending_baseline_review"]
            .fillna(False).astype(bool)
        | recipient_deviation_flag
        | profiled["location_anomaly_flag"]
            .fillna(False).astype(bool)
        | profiled["network_review_signal"]
            .fillna(False).astype(bool)
    )

    # Collect review explanations
    reason_columns = [
        "spending_baseline_reason",
        "timing_reason",
        "recipient_relationship_reason",
        "location_reason",
        "network_review_reason",
    ]

    available_reason_columns = [
        column for column in reason_columns
        if column in profiled.columns
    ]

    if available_reason_columns:
        profiled["additional_review_reasons"] = (
            profiled[available_reason_columns]
            .fillna("")
            .astype(str)
            .apply(
                lambda row: "; ".join(
                    value for value in row if value.strip()
                ),
                axis=1,
            )
        )
    else:
        profiled["additional_review_reasons"] = ""

    return profiled
