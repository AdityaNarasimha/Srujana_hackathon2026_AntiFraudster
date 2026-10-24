
import pandas as pd

from src.detection import detect_fraud
from src.customer_risk import assess_customer_risk
from src.behavior_detection import detect_behavior
from src.location_detection import detect_location_anomalies
from src.network_detection import detect_network_risks
from src.spending_baseline import detect_spending_baseline


def analyze_transactions(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine transaction, customer, behavior, spending-baseline,
    location and network review signals.

    Scores are demo heuristics, not fraud probabilities.
    Unusual activity is a review signal, not proof of fraud.
    """

    # 1. Transaction-level detection.
    detected = detect_fraud(transactions)

    # 2. Behaviour detection.
    if "timestamp" in detected.columns:
        behavior = detect_behavior(detected)
    else:
        behavior = detected.copy()
        behavior["rapid_transfer_flag"] = False
        behavior["new_recipient_flag"] = False
        behavior["behavior_reasons"] = ""

    # 3. Personal spending baseline.
    profiled = detect_spending_baseline(behavior)

    # 4. Customer-profile context.
    profiled = assess_customer_risk(profiled, customers)

    # 5. Provisional behavior points.
    profiled["behavior_risk_points"] = (
        profiled["rapid_transfer_flag"].astype(int) * 25
        + profiled["new_recipient_flag"].astype(int) * 10
    ).clip(upper=35)

    # 6. Spending deviation points.
    # A flagged deviation adds a small review-oriented score contribution.
    profiled["spending_baseline_risk_points"] = (
        profiled["spending_change_flag"].astype(int) * 15
    )

    # 7. Combined score.
    profiled["combined_risk_score"] = (
        profiled["risk_score"]
        + profiled["profile_risk_points"]
        + profiled["behavior_risk_points"]
        + profiled["spending_baseline_risk_points"]
    ).clip(upper=100)

    profiled["combined_risk_level"] = pd.cut(
        profiled["combined_risk_score"],
        bins=[-1, 19, 49, 79, 100],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)

    # 8. Location signals.
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

    # 9. Network signals.
    profiled = detect_network_risks(profiled)

    # 10. Human review recommendation.
    profiled["needs_human_review"] = (
        profiled["combined_risk_score"].ge(50)
        | profiled["profile_review_required"]
        | profiled["rapid_transfer_flag"]
        | profiled["spending_baseline_review"]
        | profiled["location_anomaly_flag"]
        | profiled["network_review_signal"]
    )

    # 11. Collect additional review reasons.
    location_reason = profiled["location_reason"].fillna("")
    network_reason = profiled["network_reason"].fillna("")
    spending_reason = profiled["spending_change_reason"].fillna("")

    profiled["additional_review_reasons"] = [
        "; ".join(
            reason
            for reason in (spending, location, network)
            if reason and reason !=
            "No spending-baseline deviation detected"
            and reason != "Insufficient personal spending history"
        )
        for spending, location, network in zip(
            spending_reason,
            location_reason,
            network_reason,
        )
    ]

    return profiled
