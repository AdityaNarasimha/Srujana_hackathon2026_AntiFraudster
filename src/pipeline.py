
import pandas as pd

from src.detection import detect_fraud
from src.customer_risk import assess_customer_risk
from src.behavior_detection import detect_behavior
from src.location_detection import detect_location_anomalies
from src.network_detection import detect_network_risks
from src.spending_baseline import detect_spending_baseline
from src.timing_detection import detect_timing_anomalies


def analyze_transactions(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine transaction, customer, behavior, spending, timing,
    location and network signals.

    Scores are demo heuristics, not fraud probabilities.
    Unusual activity is a review signal, not proof of fraud.
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

    # 5. Customer profile context.
    profiled = assess_customer_risk(profiled, customers)

    # 6. Behavior risk points.
    profiled["behavior_risk_points"] = (
        profiled["rapid_transfer_flag"].astype(int) * 25
        + profiled["new_recipient_flag"].astype(int) * 10
    ).clip(upper=35)

    # 7. Additional review-oriented score contributions.
    profiled["spending_baseline_risk_points"] = (
        profiled["spending_change_flag"].astype(int) * 15
    )

    profiled["timing_risk_points"] = (
        profiled["unusual_time_flag"].astype(int) * 10
    )

    # 8. Combined heuristic score.
    profiled["combined_risk_score"] = (
        profiled["risk_score"]
        + profiled["profile_risk_points"]
        + profiled["behavior_risk_points"]
        + profiled["spending_baseline_risk_points"]
        + profiled["timing_risk_points"]
    ).clip(upper=100)

    profiled["combined_risk_level"] = pd.cut(
        profiled["combined_risk_score"],
        bins=[-1, 19, 49, 79, 100],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)

    # 9. Location review signals.
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

    # 10. Network review signals.
    profiled = detect_network_risks(profiled)

    # 11. Recommend human review when relevant signals trigger.
    profiled["needs_human_review"] = (
        profiled["combined_risk_score"].ge(50)
        | profiled["profile_review_required"]
        | profiled["rapid_transfer_flag"]
        | profiled["spending_baseline_review"]
        | profiled["unusual_time_flag"]
        | profiled["location_anomaly_flag"]
        | profiled["network_review_signal"]
    )

    # 12. Collect additional review reasons.
    location_reason = profiled["location_reason"].fillna("")
    network_reason = profiled["network_reason"].fillna("")
    spending_reason = profiled["spending_change_reason"].fillna("")
    timing_reason = profiled["timing_reason"].fillna("")

    profiled["additional_review_reasons"] = [
        "; ".join(
            reason
            for reason in (spending, timing, location, network)
            if reason
            and reason not in {
                "No spending-baseline deviation detected",
                "Insufficient personal spending history",
                "Insufficient personal timing history",
                "Transaction hour observed in prior history",
                "Timestamp unavailable; timing not assessed",
            }
        )
        for spending, timing, location, network in zip(
            spending_reason,
            timing_reason,
            location_reason,
            network_reason,
        )
    ]

    return profiled
