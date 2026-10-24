
import math

import pandas as pd


# Fictional coordinates for testing only; these are NOT real locations.
LOCATION_COORDINATES = {
    "LOC-001": (12.00, 77.00),
    "LOC-002": (12.05, 77.05),
    "LOC-003": (15.00, 75.00),
    "LOC-004": (19.00, 73.00),
    "LOC-005": (28.00, 77.00),
}


def _distance_km(point_a, point_b):
    """Calculate approximate great-circle distance in kilometres."""
    lat1, lon1 = map(math.radians, point_a)
    lat2, lon2 = map(math.radians, point_b)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 6371 * 2 * math.asin(min(1.0, math.sqrt(a)))


def detect_location_anomalies(
    transactions: pd.DataFrame,
    minimum_travel_speed_kmh: float = 900,
) -> pd.DataFrame:
    """
    Flag consecutive transactions implying implausibly fast travel.

    This is a review signal, not proof of fraud. Coordinates are
    fictional demo coordinates. The speed threshold is configurable
    and is not a universal fraud rule.
    """
    required = {"timestamp", "sender_id", "location_id"}
    missing = required - set(transactions.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if minimum_travel_speed_kmh <= 0:
        raise ValueError("Travel speed threshold must be positive.")

    result = transactions.copy()
    result["timestamp"] = pd.to_datetime(
        result["timestamp"], errors="raise"
    )

    if result["timestamp"].isna().any():
        raise ValueError("Timestamps must not be missing.")

    if result["sender_id"].isna().any():
        raise ValueError("Sender IDs must not be missing.")

    result["location_anomaly_flag"] = False
    result["location_distance_km"] = float("nan")
    result["location_elapsed_minutes"] = float("nan")
    result["location_reason"] = ""

    # Stable ordering makes results reproducible when timestamps tie.
    result["_original_order"] = range(len(result))
    ordered = result.sort_values(
        ["sender_id", "timestamp", "_original_order"],
        kind="stable",
    )

    previous_by_sender = {}

    for idx, row in ordered.iterrows():
        sender = row["sender_id"]
        location = row["location_id"]
        timestamp = row["timestamp"]

        # Unknown or missing location: do not infer suspicious travel.
        if pd.isna(location) or location not in LOCATION_COORDINATES:
            continue

        current_point = LOCATION_COORDINATES[location]
        previous = previous_by_sender.get(sender)

        if previous is not None:
            previous_time, previous_point = previous
            elapsed_minutes = (
                timestamp - previous_time
            ).total_seconds() / 60

            # Tied or out-of-order events cannot establish travel speed.
            if elapsed_minutes > 0:
                distance_km = _distance_km(
                    previous_point, current_point
                )
                implied_speed = distance_km / (
                    elapsed_minutes / 60
                )

                result.at[idx, "location_distance_km"] = round(
                    distance_km, 2
                )
                result.at[idx, "location_elapsed_minutes"] = round(
                    elapsed_minutes, 2
                )

                if implied_speed > minimum_travel_speed_kmh:
                    result.at[idx, "location_anomaly_flag"] = True
                    result.at[idx, "location_reason"] = (
                        "Consecutive recorded locations imply "
                        "an unusually fast journey; verify the "
                        "location data and transaction context"
                    )

        previous_by_sender[sender] = (timestamp, current_point)

    return result.drop(columns=["_original_order"])
