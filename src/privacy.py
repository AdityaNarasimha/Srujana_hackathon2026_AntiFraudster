
import hashlib
import secrets

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "sender_id",
    "receiver_id",
    "amount",
}

OPTIONAL_COLUMNS = {
    "transaction_id",
    "is_fraud",
}


def protect_transactions(df: pd.DataFrame):
    """Minimize columns and pseudonymize identifiers."""

    df = df.copy()

    # Standardize column names
    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Keep only fields required by the detector.
    allowed = REQUIRED_COLUMNS | OPTIONAL_COLUMNS
    removed_columns = [
        column for column in df.columns
        if column not in allowed
    ]

    df = df[
        [column for column in df.columns if column in allowed]
    ].copy()

    # Validate transaction values.
    df["amount"] = pd.to_numeric(
        df["amount"], errors="raise"
    )

    if df["amount"].isna().any():
        raise ValueError("Transaction amounts cannot be empty.")

    if (df["amount"] < 0).any():
        raise ValueError("Negative transaction amounts detected.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="raise"
    )

    # Generate a temporary secret for this processing run.
    # IDs stay linkable within this dataset but not across runs.
    salt = secrets.token_bytes(32)

    def pseudonymize(value):
        if pd.isna(value):
            return "MISSING"

        raw = str(value).encode("utf-8")
        digest = hashlib.sha256(salt + raw).hexdigest()

        return "P-" + digest[:16]

    for column in [
        "transaction_id",
        "sender_id",
        "receiver_id",
    ]:
        if column in df.columns:
            df[column] = df[column].map(pseudonymize)

    return df, removed_columns
