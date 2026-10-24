
import pandas as pd
import pytest

from src.network_detection import detect_network_risks


def test_vpn_classification_is_reported():
    data = pd.DataFrame({"network_type": ["vpn", "residential"]})

    result = detect_network_risks(data)

    assert result["vpn_detected"].tolist() == [True, False]
    assert result["network_review_signal"].tolist() == [True, False]


def test_missing_network_data_is_unknown():
    result = detect_network_risks(pd.DataFrame({"amount": [100]}))

    assert result.iloc[0]["network_type"] == "unknown"
    assert not result.iloc[0]["network_review_signal"]


def test_invalid_network_type_is_rejected():
    with pytest.raises(ValueError):
        detect_network_risks(
            pd.DataFrame({"network_type": ["definitely_fraud"]})
        )
