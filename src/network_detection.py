
import pandas as pd


SUPPORTED_NETWORK_TYPES = {
    "vpn",
    "proxy",
    "tor",
    "hosting",
    "residential",
    "unknown",
}


def detect_network_risks(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Interpret network classifications from an authorized source.

    Expected optional column:
        network_type

    Accepted values:
        vpn, proxy, tor, hosting, residential, unknown

    Missing data is treated as unknown, not as suspicious.
    This function does not independently verify IP addresses.
    """
    result = transactions.copy()

    if "network_type" not in result.columns:
        result["network_type"] = "unknown"

    network_type = (
        result["network_type"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid = ~network_type.isin(SUPPORTED_NETWORK_TYPES)

    if invalid.any():
        bad_values = sorted(network_type[invalid].unique())
        raise ValueError(
            f"Unsupported network_type values: {bad_values}"
        )

    result["network_type"] = network_type
    result["vpn_detected"] = network_type.eq("vpn")
    result["proxy_detected"] = network_type.eq("proxy")
    result["tor_detected"] = network_type.eq("tor")
    result["hosting_network_detected"] = network_type.eq("hosting")

    # Network type alone does not establish fraud.
    result["network_review_signal"] = (
        result["vpn_detected"]
        | result["proxy_detected"]
        | result["tor_detected"]
        | result["hosting_network_detected"]
    )

    result["network_reason"] = ""

    reason_map = {
        "vpn": "Network source identifies a known VPN",
        "proxy": "Network source identifies a known proxy",
        "tor": "Network source identifies a Tor exit node",
        "hosting": "Network source identifies hosting-provider traffic",
    }

    result["network_reason"] = network_type.map(
        reason_map
    ).fillna("")

    return result
