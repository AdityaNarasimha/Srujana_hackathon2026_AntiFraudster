
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_CUSTOMERS = 100
N_NORMAL = 900

START = pd.Timestamp("2026-09-01")
END = pd.Timestamp("2026-10-01")

LOCATIONS = [
    "LOC-001",
    "LOC-002",
    "LOC-003",
    "LOC-004",
    "LOC-005",
]

RAPID_WINDOW_MINUTES = 5
RAPID_BURST_COUNT = 25
TRANSACTIONS_PER_BURST = 3


def generate_experimental_data() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    transaction_number = 1

    normal_receivers = {
        customer: [
            f"MERCHANT-{customer:03d}-{i:02d}"
            for i in range(1, 4)
        ]
        for customer in range(1, N_CUSTOMERS + 1)
    }

    # 1. Generate legitimate baseline history.
    for _ in range(N_NORMAL):
        sender_num = int(rng.integers(1, N_CUSTOMERS + 1))
        sender = f"ACC-{sender_num:03d}"

        timestamp = START + pd.Timedelta(
            minutes=int(
                rng.integers(
                    0, int((END - START).total_seconds() / 60)
                )
            )
        )

        rows.append({
            "transaction_id": f"TXN-{transaction_number:05d}",
            "timestamp": timestamp,
            "sender_id": sender,
            "receiver_id": str(
                rng.choice(normal_receivers[sender_num])
            ),
            "amount": round(
                float(rng.lognormal(mean=4.5, sigma=0.65)), 2
            ),
            "location_id": str(rng.choice(LOCATIONS)),
            "is_fraud": 0,
            "scenario": "normal",
        })
        transaction_number += 1

    normal_df = pd.DataFrame(rows)

    # 2. Find customers with sufficient legitimate history.
    eligible_senders = []

    for sender, group in normal_df.groupby("sender_id"):
        if len(group) >= 5:
            eligible_senders.append(sender)

    if not eligible_senders:
        raise RuntimeError(
            "Not enough sender history for new-recipient scenarios."
        )

    # 3. Generate non-rapid scenarios first.
    scenario_names = (
        ["unusual_amount"] * 25
        + ["new_recipient"] * 25
        + ["large_legitimate"] * 15
        + ["nighttime_legitimate"] * 10
    )
    rng.shuffle(scenario_names)

    for scenario in scenario_names:
        sender_num = int(rng.integers(1, N_CUSTOMERS + 1))
        sender = f"ACC-{sender_num:03d}"

        timestamp = START + pd.Timedelta(
            minutes=int(
                rng.integers(
                    0, int((END - START).total_seconds() / 60)
                )
            )
        )

        receiver = str(rng.choice(normal_receivers[sender_num]))
        amount = round(
            float(rng.lognormal(mean=4.5, sigma=0.65)), 2
        )
        is_fraud = 1
        location_id = str(rng.choice(LOCATIONS))

        if scenario == "unusual_amount":
            amount = round(float(rng.uniform(5000, 15000)), 2)

        elif scenario == "new_recipient":
            # Select a sender with at least five normal transactions.
            sender = str(rng.choice(eligible_senders))
            sender_num = int(sender.split("-")[1])

            receiver = f"NEW-{transaction_number:05d}"
            amount = round(float(rng.uniform(100, 1000)), 2)

            # Place the event after the sender's latest normal event.
            sender_normal = normal_df[
                normal_df["sender_id"] == sender
            ]
            latest_normal_time = sender_normal["timestamp"].max()

            # Keep the event inside the generated time period.
            timestamp = latest_normal_time + pd.Timedelta(seconds=1)
            if timestamp >= END:
                timestamp = END - pd.Timedelta(seconds=1)

        elif scenario == "large_legitimate":
            amount = round(float(rng.uniform(5000, 15000)), 2)
            is_fraud = 0

        elif scenario == "nighttime_legitimate":
            day = int(rng.integers(0, 30))
            timestamp = START + pd.Timedelta(days=day, hours=2)
            amount = round(float(rng.uniform(50, 1000)), 2)
            is_fraud = 0

        rows.append({
            "transaction_id": f"TXN-{transaction_number:05d}",
            "timestamp": timestamp,
            "sender_id": sender,
            "receiver_id": receiver,
            "amount": amount,
            "location_id": location_id,
            "is_fraud": is_fraud,
            "scenario": scenario,
        })
        transaction_number += 1

    # 4. Generate rapid bursts last. Choose sender/time combinations
    # that do not overlap existing transactions from the same sender.
    existing = pd.DataFrame(rows)
    rapid_bursts_created = 0
    attempts = 0
    max_attempts = 100_000

    while (
        rapid_bursts_created < RAPID_BURST_COUNT
        and attempts < max_attempts
    ):
        attempts += 1

        sender_num = int(rng.integers(1, N_CUSTOMERS + 1))
        sender = f"ACC-{sender_num:03d}"

        # Allow the full four-minute burst to finish before END.
        latest_start = END - pd.Timedelta(minutes=4)
        available_minutes = int(
            (latest_start - START).total_seconds() // 60
        )
        start_time = START + pd.Timedelta(
            minutes=int(rng.integers(0, available_minutes + 1))
        )

        # Check the complete five-minute window, not just exact
        # timestamp matches, against all existing same-sender events.
        sender_times = pd.to_datetime(
            existing.loc[
                existing["sender_id"] == sender, "timestamp"
            ]
        )

        window_start = start_time - pd.Timedelta(
            minutes=RAPID_WINDOW_MINUTES
        )
        window_end = start_time + pd.Timedelta(
            minutes=RAPID_WINDOW_MINUTES
        )

        overlaps = (
            (sender_times >= window_start)
            & (sender_times <= window_end)
        ).any()

        if overlaps:
            continue

        location_id = str(rng.choice(LOCATIONS))

        for offset in range(TRANSACTIONS_PER_BURST):
            timestamp = start_time + pd.Timedelta(
                minutes=offset * 2
            )

            rows.append({
                "transaction_id": f"TXN-{transaction_number:05d}",
                "timestamp": timestamp,
                "sender_id": sender,
                "receiver_id": f"RAPID-{transaction_number:05d}",
                "amount": round(float(rng.uniform(50, 300)), 2),
                "location_id": location_id,
                "is_fraud": 1,
                "scenario": "rapid_transfers",
            })
            transaction_number += 1

        rapid_bursts_created += 1

        # Update the lookup so later bursts cannot overlap this burst.
        new_rows = rows[-TRANSACTIONS_PER_BURST:]
        existing = pd.concat(
            [existing, pd.DataFrame(new_rows)],
            ignore_index=True,
        )

    if rapid_bursts_created != RAPID_BURST_COUNT:
        raise RuntimeError(
            "Could not generate all rapid-transfer bursts "
            "without sender/time overlap."
        )

    # 5. Validate and save.
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["timestamp", "transaction_id"]
    ).reset_index(drop=True)

    assert len(df) == 1050, f"Unexpected row count: {len(df)}"
    assert df["transaction_id"].is_unique
    assert df["amount"].ge(0).all()
    assert df["is_fraud"].isin([0, 1]).all()
    assert df["location_id"].notna().all()

    expected_counts = {
        "normal": 900,
        "rapid_transfers": 75,
        "new_recipient": 25,
        "unusual_amount": 25,
        "large_legitimate": 15,
        "nighttime_legitimate": 10,
    }
    actual_counts = df["scenario"].value_counts().to_dict()
    assert actual_counts == expected_counts, (
        f"Unexpected scenario counts: {actual_counts}"
    )

    output_path = Path("data/experimental_transactions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    data = generate_experimental_data()

    print(f"Generated {len(data)} transactions")
    print(f"Fraud-labelled: {int(data['is_fraud'].sum())}")
    print("\nScenario counts:")
    print(data["scenario"].value_counts())
    print("\nLabel counts:")
    print(data["is_fraud"].value_counts())
    print("\nSaved to data/experimental_transactions.csv")
