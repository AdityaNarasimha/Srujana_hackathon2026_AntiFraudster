
import pandas as pd

from src.detection import detect_fraud
from src.customer_risk import assess_customer_risk
from src.behavior_detection import detect_behavior
from src.location_detection import detect_location_anomalies
from src.network_detection import detect_network_risks


def analyze_transactions(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine transaction, behavior, customer, location and network signals.

    Scores are demo heuristics, not fraud probabilities.
    Location/network signals are separate review signals, not proof.
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

    # 3. Customer-profile context.
    profiled = assess_customer_risk(behavior, customers)

    # 4. Provisional behaviour points.
    profiled["behavior_risk_points"] = (
        profiled["rapid_transfer_flag"].astype(int) * 25
        + profiled["new_recipient_flag"].astype(int) * 10
    ).clip(upper=35)

    # 5. Existing score remains unchanged by location/network signals.
    profiled["combined_risk_score"] = (
        profiled["risk_score"]
        + profiled["profile_risk_points"]
        + profiled["behavior_risk_points"]
    ).clip(upper=100)

    profiled["combined_risk_level"] = pd.cut(
        profiled["combined_risk_score"],
        bins=[-1, 19, 49, 79, 100],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)

    # 6. Location signals.
    if "timestamp" in profiled.columns and "location_id" in profiled.columns:
        profiled = detect_location_anomalies(profiled)
    else:
        profiled["location_anomaly_flag"] = False
        profiled["location_distance_km"] = float("nan")
        profiled["location_elapsed_minutes"] = float("nan")
        profiled["location_reason"] = ""

    # 7. VPN/proxy signals.
    profiled = detect_network_risks(profiled)

    # 8. Human review: a signal asks for review; it does not prove fraud.
    profiled["needs_human_review"] = (
        profiled["combined_risk_score"].ge(50)
        | profiled["profile_review_required"]
        | profiled["rapid_transfer_flag"]
        | profiled["location_anomaly_flag"]
        | profiled["network_review_signal"]
    )

    # 9. Collect transparent reasons for the reviewer.
    profiled["additional_review_reasons"] = ""

    location_reason = profiled["location_reason"].fillna("")
    network_reason = profiled["network_reason"].fillna("")

    profiled["additional_review_reasons"] = [
        "; ".join(
            reason for reason in (loc, net) if reason
        )
        for loc, net in zip(location_reason, network_reason)
    ]

    return profiled
