"""Partial caching must preserve likelihoods across its memory boundary."""

import numpy as np
import pytest

pytest.importorskip("flint")
pytest.importorskip("finufft")

from research.kepler_transform.detector_time.fields import CommonDetector
from research.kepler_transform.detector_time.scaling import BudgetPixelCache, streamed


def test_partial_cache_and_uncached_numerator_agree_with_independent_pixels():
    rng = np.random.default_rng(1298)
    shape = (3, 7, 7)
    images = rng.normal(size=shape)
    detector = CommonDetector(rng.uniform(.5, 2., shape), [1., 1.4],
                              [[.9, .2], [-.15, 1.1]],
                              masks=rng.random(shape)<.1, source_half_width=4.)
    pos = rng.uniform(-3.9, 3.9, (17, 3, 2))
    dn, di = detector.direct(images, pos)
    np.testing.assert_allclose(streamed(detector, images, pos, batch=5), [dn,di], atol=1e-14)
    base = pos.nbytes + di.nbytes
    for cached_count in (0, 1, 6, 17):
        budget = base + cached_count * 3 * 7 * 2 * 8
        cache = BudgetPixelCache(detector, pos, di, budget, batch=4)
        assert cache.cached_queries == cached_count
        assert cache.persistent_bytes <= budget
        for data in (images, .7*images + rng.normal(size=shape)):
            np.testing.assert_allclose(cache.apply(data), detector.direct(data,pos), atol=2e-14)
    with pytest.raises(ValueError, match="budget"):
        BudgetPixelCache(detector, pos, di, base-1)
    with pytest.raises(ValueError, match="information"):
        BudgetPixelCache(detector, pos, di[:2], base)
