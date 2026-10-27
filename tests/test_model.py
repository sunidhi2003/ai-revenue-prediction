from src.features import make_features

def test_feature_generation():
    row = make_features("2026-01-10", "Canada", ["Canada"])
    assert row["year"] == 2026
    assert row["country"] == "Canada"
    assert row["is_weekend"] in (0, 1)
