
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TRANSACTIONS_FILE = ROOT / "data" / "sample_transactions.csv"
CUSTOMERS_FILE = ROOT / "data" / "sample_customers.csv"


def generate_customers() -> pd.DataFrame:
    transactions = pd.read_csv(TRANSACTIONS_FILE)

    # Generate one fictional profile for every account in the dataset.
    account_ids = sorted(
        set(transactions["sender_id"].dropna())
        | set(transactions["receiver_id"].dropna())
    )

    rng = np.random.default_rng(42)
    customer_types = ["student", "salaried", "shopkeeper", "small_business"]

    profiles = []

    for account_id in account_ids:
        customer_type = rng.choice(customer_types)

        if customer_type == "student":
            monthly_inflow = int(rng.integers(5000, 15001))
            monthly_outflow = int(rng.integers(3000, 12001))
            usual_transaction_count = int(rng.integers(10, 31))

        elif customer_type == "salaried":
            monthly_inflow = int(rng.integers(20000, 80001))
            monthly_outflow = int(rng.integers(10000, 60001))
            usual_transaction_count = int(rng.integers(15, 51))

        elif customer_type == "shopkeeper":
            monthly_inflow = int(rng.integers(30000, 150001))
            monthly_outflow = int(rng.integers(15000, 100001))
            usual_transaction_count = int(rng.integers(50, 201))

        else:
            monthly_inflow = int(rng.integers(100000, 500001))
            monthly_outflow = int(rng.integers(50000, 350001))
            usual_transaction_count = int(rng.integers(30, 151))

        profiles.append({
            "customer_id": account_id,
            "customer_type": customer_type,
            "expected_monthly_inflow": monthly_inflow,
            "expected_monthly_outflow": monthly_outflow,
            "usual_transaction_count": usual_transaction_count,
            "kyc_status": rng.choice(
                ["verified_demo", "pending_demo"],
                p=[0.9, 0.1],
            ),
        })

    return pd.DataFrame(profiles)


if __name__ == "__main__":
    CUSTOMERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    customers = generate_customers()
    customers.to_csv(CUSTOMERS_FILE, index=False)
    print(f"Generated {len(customers)} fictional customer profiles.")
    print(customers.head())
