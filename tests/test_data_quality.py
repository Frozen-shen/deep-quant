from pathlib import Path

import pandas as pd

import data_cache
import scripts.active.fix_data_quality as fix_data_quality


ROOT = Path(__file__).resolve().parents[1]


def test_unadjusted_directory_has_one_canonical_location():
    assert Path(data_cache.UNADJUSTED_DIR) == ROOT / "data_store" / "unadjusted"


def test_purge_invalid_rows_removes_known_corrupt_tail_and_weekend_rows():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-05-18", "2023-05-19", "2023-05-20", "2023-05-22"]
            ),
            "open": [0.4, 3290.0, 3290.0, 3250.0],
            "high": [0.4, 3295.0, 3290.0, 3260.0],
            "low": [0.4, 3255.0, 3290.0, 3240.0],
            "close": [0.4, 3260.0, 3290.0, 3250.0],
            "volume": [1000, 1000, 0, 2000],
            "amount": [400.0, 0.0, 0.0, 650.0],
        }
    )

    cleaned, report = fix_data_quality.purge_invalid_rows(df, "000540")

    assert cleaned["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2023-05-18"
    ]
    assert report["removed_rows"] == 3
    assert report["reasons"]["known_corrupt_tail"] == 3
    assert report["reasons"]["weekend"] == 1


def test_reconcile_unadjusted_fields_uses_same_date_qfq_values_only():
    unadjusted = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-04", "2022-01-05"]),
            "close": [11.6, 11.4],
            "amount": [None, None],
            "turnover": [None, 1.2],
        }
    )
    qfq = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-04", "2022-01-05"]),
            "amount": [260829736.73, 0.0],
            "turnover": [3.0665, 0.0],
        }
    )

    repaired, report = fix_data_quality.reconcile_unadjusted_fields(
        unadjusted, qfq
    )

    assert repaired.loc[0, "amount"] == 260829736.73
    assert repaired.loc[0, "turnover"] == 3.0665
    assert pd.isna(repaired.loc[1, "amount"])
    assert repaired.loc[1, "turnover"] == 1.2
    assert report["filled"]["amount"] == 1
    assert report["filled"]["turnover"] == 1
