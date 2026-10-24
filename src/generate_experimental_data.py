
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_CUSTOMERS = 100
N_NORMAL = 900

START = pd.Timestamp("2026-09-01")
END = pd.Timestamp("2026-10-01")

# Fictional location identifiers for the prototype.
LOCATIONS = [
    "LOC-001",
    "LOC-002",
    "LOC-003",
    "LOC-004",
    "LOC-005",
]


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

    # Generate normal transaction histories first.
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

    # Find senders with enough existing history to test
    # first-time recipients fairly.
    normal_df = pd.DataFrame(rows)
    eligible_history = {}

    for sender, group in normal_df.groupby("sender_id"):
        times = group["timestamp"].sort_values().tolist()

        if len(times) >= 5:
            eligible_history[sender] = times

    if not eligible_history:
        raise RuntimeError(
            "Not enough sender history for new-recipient scenarios."
        )

    scenario_names = (
        ["unusual_amount"] * 25
        + ["rapid_transfers"] * 25
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

        receiver = str(
            rng.choice(normal_receivers[sender_num])
        )
        amount = round(
            float(rng.lognormal(mean=4.5, sigma=0.65)), 2
        )
        is_fraud = 1
        location_id = str(rng.choice(LOCATIONS))

        if scenario == "unusual_amount":
            amount = round(float(rng.uniform(5000, 15000)), 2)

        elif scenario == "rapid_transfers":
            # Three transactions from one sender within four minutes.
            timestamp = START + pd.Timedelta(
                minutes=int(rng.integers(0, 29 * 24 * 60))
            )

            for offset in range(3):
                rows.append({
                    "transaction_id": (
                        f"TXN-{transaction_number:05d}"
                    ),
                    "timestamp": timestamp
                    + pd.Timedelta(minutes=offset * 2),
                    "sender_id": sender,
                    "receiver_id": (
                        f"RAPID-{transaction_number:05d}"
                    ),
                    "amount": round(
                        float(rng.uniform(50, 300)), 2
                    ),
                    "location_id": location_id,
                    "is_fraud": 1,
                    "scenario": "rapid_transfers",
                })
                transaction_number += 1

            continue

        elif scenario == "new_recipient":
            # Select a sender with at least five earlier normal
            # transactions, then place this transaction after them.
            sender = str(rng.choice(list(eligible_history)))
            sender_history = eligible_history[sender]

            fifth_history_time = sender_history[4]
            timestamp = fifth_history_time + pd.Timedelta(seconds=1)

            # Ensure the generated event remains in the test period.
            if timestamp >= END:
                timestamp = END - pd.Timedelta(minutes=1)

            receiver = f"NEW-{transaction_number:05d}"
            amount = round(float(rng.uniform(100, 1000)), 2)

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

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["timestamp", "transaction_id"]
    ).reset_index(drop=True)

    assert df["transaction_id"].is_unique
    assert df["amount"].ge(0).all()
    assert df["is_fraud"].isin([0, 1]).all()
    assert df["location_id"].notna().all()

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
    print("\nLocation counts:")
    print(data["location_id"].value_counts())
    print("\nSaved to data/experimental_transactions.csv")
