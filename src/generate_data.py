
import numpy as np
import pandas as pd

# Make the sample reproducible
rng = np.random.default_rng(42)

n_normal = 980
n_suspicious = 20
n = n_normal + n_suspicious

# Generate fictional transaction amounts
normal_amounts = rng.lognormal(
    mean=5.5, sigma=0.8, size=n_normal
).round(2)

suspicious_amounts = rng.uniform(
    10000, 50000, size=n_suspicious
).round(2)

amounts = np.concatenate([
    normal_amounts,
    suspicious_amounts
])

# Generate timestamps across several days
timestamps = pd.Timestamp("2026-10-01") + pd.to_timedelta(
    rng.integers(0, 7 * 24 * 60, size=n), unit="m"
)

# Make the example suspicious transactions occur at night
timestamps = pd.Series(timestamps)
timestamps.iloc[n_normal:] = (
    pd.Timestamp("2026-10-07")
    + pd.to_timedelta(
        rng.integers(0, 4 * 60, size=n_suspicious), unit="m"
    )
)

df = pd.DataFrame({
    "transaction_id": [
        f"TXN-{i:04d}" for i in range(1, n + 1)
    ],
    "timestamp": timestamps,
    "sender_id": [
        f"ACC-{x:03d}" for x in rng.integers(1, 101, size=n)
    ],
    "receiver_id": [
        f"ACC-{x:03d}" for x in rng.integers(101, 151, size=n)
    ],
    "amount": amounts,
    "is_fraud": [0] * n_normal + [1] * n_suspicious
})

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/sample_transactions.csv", index=False)

print("Sample dataset created!")
print(f"Total transactions: {len(df)}")
print(f"Example fraud labels: {df['is_fraud'].sum()}")
print(df.head())
