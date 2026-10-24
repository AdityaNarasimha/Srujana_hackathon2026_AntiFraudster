

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
    """
    Combine transaction, customer, behavior, spending, timing,
    recipient relationship, location and network signals.

    Scores are demo heuristics, not fraud probabilities.
    Review signals do not prove fraud.
    """

    # 1. Transaction-level detection.
    detected = detect_fraud(transactions)

    # 2. Behavior detection.
    if "timestamp" in detected.columns:
        behavior = detect_behavior(detected)
    else:
        behavior = detected.copy()
        behavior["rapid_transfer_flag"] = False
        behavior["new_recipient_flag"] = False
        behavior["behavior_reasons"] = ""

    # 3. Personal spending baseline.
    profiled = detect_spending_baseline(behavior)

    # 4. Personal timing baseline.
    if "timestamp" in profiled.columns:
        profiled = detect_timing_anomalies(profiled)
    else:
        profiled["unusual_time_flag"] = False
        profiled["historical_timing_count"] = 0
        profiled["historical_active_hours"] = ""
        profiled["timing_reason"] = (
            "Timestamp unavailable; timing not assessed"
        )

    # 5. Recipient relationship history.
    if "timestamp" in profiled.columns:
        profiled = detect_recipient_relationships(profiled)
    else:
        # Relationship history requires chronological ordering.
        # Without timestamps, preserve existing new-recipient signals
        # and explicitly mark relationship analysis as unavailable.
        profiled["recipient_prior_transaction_count"] = 0
        profiled["recipient_prior_total_amount"] = 0.0
        profiled["recipient_prior_average_amount"] = float("nan")
        profiled["recipient_relationship_new"] = False
        profiled["recipient_amount_deviation_flag"] = False
        profiled["recipient_relationship_reason"] = (
            "Timestamp unavailable; relationship history not assessed"
        )

    # 6. Customer profile context.
    profiled = assess_customer_risk(profiled, customers)

    # 7. Behavior risk points.
    profiled["behavior_risk_points"] = (
        profiled["rapid_transfer_flag"].astype(int) * 25
        + profiled["new_recipient_flag"].astype(int) * 10
    ).clip(upper=35)

    # 8. Review-oriented score contributions.
    profiled["spending_baseline_risk_points"] = (
        profiled["spending_change_flag"].astype(int) * 15
    )

    profiled["timing_risk_points"] = (
        profiled["unusual_time_flag"].astype(int) * 10
    )

    # Recipient amount deviation is a separate review signal.
    profiled["recipient_relationship_risk_points"] = (
        profiled["recipient_amount_deviation_flag"].astype(int) * 10
    )

    # 9. Combined heuristic score.
    profiled["combined_risk_score"] = (
        profiled["risk_score"]
        + profiled["profile_risk_points"]
        + profiled["behavior_risk_points"]
        + profiled["spending_baseline_risk_points"]
        + profiled["timing_risk_points"]
        + profiled["recipient_relationship_risk_points"]
    ).clip(upper=100)

    profiled["combined_risk_level"] = pd.cut(
        profiled["combined_risk_score"],
        bins=[-1, 19, 49, 79, 100],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)

    # 10. Location review signals.
    if (
        "timestamp" in profiled.columns
        and "location_id" in profiled.columns
    ):
        profiled = detect_location_anomalies(profiled)
    else:
        profiled["location_anomaly_flag"] = False
        profiled["location_distance_km"] = float("nan")
        profiled["location_elapsed_minutes"] = float("nan")
        profiled["location_reason"] = ""

    # 11. Network review signals.
    profiled = detect_network_risks(profiled)

    # 12. Human review recommendation.
    profiled["needs_human_review"] = (
        profiled["combined_risk_score"].ge(50)
        | profiled["profile_review_required"]
        | profiled["rapid_transfer_flag"]
        | profiled["spending_baseline_review"]
        | profiled["unusual_time_flag"]
        | profiled["recipient_amount_deviation_flag"]
        | profiled["location_anomaly_flag"]
        | profiled["network_review_signal"]
    )

    # 13. Collect additional review reasons.
    reason_columns = [
        "spending_change_reason",
        "timing_reason",
        "recipient_relationship_reason",
        "location_reason",
        "network_reason",
    ]

    reasons = []
    for _, row in profiled[reason_columns].fillna("").iterrows():
        parts = [
            str(row[column]).strip()
            for column in reason_columns
            if str(row[column]).strip()
            and str(row[column]).strip() not in {
                "No spending-baseline deviation detected",
                "Insufficient personal spending history",
                "Insufficient personal timing history",
                "Transaction hour observed in prior history",
                "Timestamp unavailable; timing not assessed",
                "Timestamp unavailable; relationship history not assessed",
            }
        ]
        reasons.append("; ".join(parts))

    profiled["additional_review_reasons"] = reasons

    return profiled
