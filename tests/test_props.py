"""Props model: quantile ordering and over-probability helper."""
import numpy as np
import pytest

from src.models.props_model import PropsModel


def test_over_probability_at_median_is_half():
    assert PropsModel.over_probability(10, 20, 30, 20) == pytest.approx(0.5)


def test_over_probability_at_quantiles():
    assert PropsModel.over_probability(10, 20, 35, 10) == pytest.approx(0.9, abs=1e-6)
    assert PropsModel.over_probability(10, 20, 35, 35) == pytest.approx(0.1, abs=1e-6)


def test_quantile_model_outputs_ordered_intervals(features):
    p = features.players
    model = PropsModel(stats=["pts", "reb"], n_estimators=30).fit(p)
    out = model.predict(p[p["is_upcoming"] == 1])
    for s in ["pts", "reb"]:
        assert (out[f"{s}_p10"] <= out[f"{s}_p50"]).all()
        assert (out[f"{s}_p50"] <= out[f"{s}_p90"]).all()
        assert (out[f"{s}_p10"] >= 0).all()
