import numpy as np
import pytest

from magictools.nulls import gaussian_log_null, rank_exposures, update_nulls


def test_bayes_update_against_hand_calculation():
    result = update_nulls([0.25, 0.75], np.log([[0.2, 0.8], [0.5, 0.25]]))
    expected = np.array([0.25 * 0.2 * 0.5, 0.75 * 0.8 * 0.25])
    np.testing.assert_allclose(result["posterior_weights"], expected / expected.sum())
    assert result["log_evidence"] == pytest.approx(np.log(expected.sum()))


def test_off_chip_does_not_exclude_and_half_depth_is_half():
    values = gaussian_log_null([1000, 5, 5], 5, [0, 1, 0.4])
    np.testing.assert_allclose(np.exp(values), [1, 0.5, 0.8])


def test_extreme_likelihood_stays_in_log_space():
    result = update_nulls([1, 1], gaussian_log_null([[1000, 1001]], 5))
    assert np.isfinite(result["log_evidence"])
    assert result["posterior_weights"][0] > 0.999


def test_all_impossible_is_error():
    with pytest.raises(ValueError, match="zero probability"):
        update_nulls([1, 1], [[-np.inf, -np.inf]])


def test_zero_prior_stays_zero_and_no_evidence_preserves_prior():
    np.testing.assert_allclose(update_nulls([0, 1, 3], np.empty((0, 3)))["posterior_weights"], [0, 0.25, 0.75])


def test_information_gain_ranks_discriminating_frame():
    ranked = rank_exposures([1, 1], [[0.99, 0.99], [1, 0], [0, 0]])
    assert ranked[0]["index"] == 1
    assert ranked[0]["expected_information_nats"] == pytest.approx(np.log(2))
    assert ranked[1]["expected_information_nats"] == pytest.approx(0)
    assert ranked[2]["expected_information_nats"] == pytest.approx(0)


@pytest.mark.parametrize("prior,likelihood", [([0, 0], [[0, 0]]), ([-1, 2], [[0, 0]]), ([1, 1], [[0.1, 0]]), ([1, 1], [[np.nan, 0]]), ([1, 1], [0, 0])])
def test_bad_null_inputs(prior, likelihood):
    with pytest.raises(ValueError):
        update_nulls(prior, likelihood)
