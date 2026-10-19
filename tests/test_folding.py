import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import ndtri

from magictools.folding import FilterFrame, FoldPlan, direct_statistics, search_fold
from magictools.fold_kepler import kepler_positions
from magictools.photometry import sampled_psf
from magictools.orbits import GM


def test_fft_matches_independent_pixel_likelihood_with_masks_and_signed_psf():
    rng = np.random.default_rng(16)
    shape = (19, 23)
    image = rng.normal(size=shape) + 9
    variance = rng.uniform(.1, 4, shape)
    mask = rng.uniform(size=shape) < .2
    image[1, 1] = np.nan
    mask[1, 1] = True
    kernel = rng.normal(.2, .4, (5, 7))
    kernel[2, 3] += 5
    frame = FilterFrame(image, variance, kernel, 9, mask, 1.3)
    xy = np.array([[0, 0], [.01, .95], [21.99, 17.99], [10.7, 8.3], [-.1, 3], [22, 4]])
    result = frame.exact(xy)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for index, (x, y) in enumerate(xy[:4]):
        p = 1.3*sampled_psf(xx, yy, x, y, kernel)
        numerator = np.sum(p[~mask]*(image[~mask]-9)/variance[~mask])
        information = np.sum(p[~mask]**2/variance[~mask])
        np.testing.assert_allclose(result[index], [numerator, information], atol=1e-12)
    np.testing.assert_array_equal(result[4:], 0)


@pytest.mark.parametrize("subdivisions", [1, 2, 8])
def test_bernstein_bounds_enclose_random_positions(subdivisions):
    rng = np.random.default_rng(6)
    kernel = rng.normal(.1, .3, (5, 5))
    kernel[2, 2] += 4
    image = rng.normal(size=(20, 20))
    frame = FilterFrame(image, rng.uniform(.2, 3, image.shape), kernel, 0,
                        rng.uniform(size=image.shape) < .6)
    points = rng.uniform(0, 18.99, (500, 2))
    cells = np.floor(points*subdivisions).astype(int)
    bounds = frame.bounds(cells, subdivisions)
    for _ in range(10):
        positions = (cells+rng.uniform(size=cells.shape))/subdivisions
        exact = frame.exact(positions)
        assert np.all(exact[:, 0] >= bounds[:, 0])
        assert np.all(exact[:, 0] <= bounds[:, 1])
        assert np.all(exact[:, 1] >= bounds[:, 2])
        assert np.all(exact[:, 1] <= bounds[:, 3])


@pytest.mark.parametrize("n_frames", [1, 5, 16])
def test_shared_dag_matches_independent_bounds_and_search(n_frames):
    rng = np.random.default_rng(41)
    frames = [FilterFrame(rng.normal(size=(16, 16)), 1, np.ones((3, 3)), 0) for _ in range(n_frames)]
    positions = rng.uniform(2, 12, (160, n_frames, 2))
    positions[20:40] = positions[:20]  # exact ties remain separate trials.
    positions[-1] = -5
    positions[0, ::2] = -2  # intermittent coverage.
    plan = FoldPlan(positions, [f.shape for f in frames], 2)
    direct_bounds = np.zeros((len(positions), 4))
    for j, frame in enumerate(frames):
        cells = np.floor(positions[:, j]*2).astype(int)
        cells[~frame.valid(positions[:, j])] = -1
        direct_bounds += frame.bounds(cells, 2)
    np.testing.assert_allclose(plan.bounds(frames), direct_bounds, atol=1e-12)
    result = search_fold(frames, plan, alpha=.2, top_k=5, validate_direct=True)
    assert result["validation"]["identical_top_and_detections"]
    assert result["trials"] == 160
    assert result["work"]["screened_tracks"] > 0


def test_coverage_no_information_and_negative_scores():
    frames = [FilterFrame(-np.ones((8, 8)), 1, np.ones((3, 3)), 0)]
    positions = np.array([[[2.2, 2.4]], [[2.2, 2.4]], [[-.01, 4]]])
    plan = FoldPlan(positions, [frames[0].shape])
    result = search_fold(frames, plan, top_k=3, validate_direct=True)
    assert len(result["top"]) == 2
    assert result["top"][0]["snr"] < 0
    assert result["detections"] == []
    blank = [FilterFrame(np.ones((8, 8)), 1, np.ones((3, 3)), 0, np.ones((8, 8)))]
    assert search_fold(blank, plan, top_k=2, validate_direct=True)["top"] == []


def test_mask_holes_and_tiny_information_do_not_become_fft_phantom_fits():
    rng = np.random.default_rng(12)
    image = rng.normal(size=(32, 32))
    mask = np.zeros((32, 32))
    mask[10:20, 10:20] = 1
    positions = np.array([[14., 14.], [14.2, 14.2]])
    frame = FilterFrame(image, 1, np.ones((3, 3)), 0, mask)
    np.testing.assert_array_equal(frame.exact(positions), 0)
    np.testing.assert_array_equal(frame.bounds(np.floor(positions).astype(int), 1), 0)
    # Small but real information is evaluated directly, never discarded.
    frame = FilterFrame(image, 1e15, np.ones((3, 3)), 0)
    result = frame.exact(positions)
    yy, xx = np.mgrid[:32, :32]
    for i, (x, y) in enumerate(positions):
        p = sampled_psf(xx, yy, x, y, np.ones((3, 3)))
        np.testing.assert_allclose(result[i], [np.sum(p*image)/1e15, np.sum(p*p)/1e15], rtol=1e-13, atol=1e-30)


def test_elliptic_positions_match_independent_ode_forward_backward():
    # At perihelion on x with velocity +y; compare against a separate ODE,
    # including high eccentricity and a full orbit.
    for eccentricity in (0, .8, .95):
        a, mu = 2., GM["sun"]
        state = [a*(1-eccentricity), 0, 0, 0,
                 np.sqrt(mu/a*(1+eccentricity)/(1-eccentricity)), 0]
        period = 2*np.pi*np.sqrt(a**3/mu)
        for sign in (-1, 1):
            t = np.linspace(0, sign*period, 101)
            def derivative(_, state):
                r = state[:3]
                return np.r_[state[3:], -mu*r/np.linalg.norm(r)**3]
            reference = solve_ivp(derivative, [0, t[-1]], state, t_eval=t,
                                  method="DOP853", rtol=2e-13, atol=2e-14)
            actual = kepler_positions([[a, eccentricity, 0, 0, 0, 0]], t)[0]
            np.testing.assert_allclose(actual, reference.y[:3].T, atol=2e-8)


def test_invalid_bank_and_ellipses_are_rejected():
    with pytest.raises(ValueError):
        FoldPlan(np.ones((3, 4)), [(8, 8)])
    with pytest.raises(ValueError):
        FoldPlan(np.ones((3, 1, 2)), [(8, 8)], True)
    with pytest.raises(ValueError):
        kepler_positions([[2, 1.1, 0, 0, 0, 0]], [0])
    with pytest.raises(ValueError):
        kepler_positions([[2, .1, 0, 0, 0, 0]], [np.nan])


def test_reusing_plan_on_changed_images_and_threshold_boundary():
    rng = np.random.default_rng(17)
    positions = rng.uniform(2, 14, (200, 6, 2))
    plan = FoldPlan(positions, [(18, 18)]*6)
    for sign in (-1, 1):
        frames = [FilterFrame(sign*rng.normal(size=(18, 18)), 2, np.ones((3, 3)), 0) for _ in range(6)]
        result = search_fold(frames, plan, top_k=10, alpha=.45, validate_direct=True)
        assert result["validation"]["all_bounds_contain_direct_snr"]
    with pytest.raises(ValueError, match="shapes"):
        plan.bounds([FilterFrame(np.zeros((19, 19)), 1, np.ones((3, 3)), 0)]*6)


def test_candidates_immediately_above_and_below_threshold():
    yy, xx = np.mgrid[:20, :20]
    kernel = np.ones((3, 3))
    positions = np.array([[[3.2, 4.4]], [[12.2, 4.4]], [[-1, -1]]])
    threshold = -ndtri(.01/3)
    image = np.zeros((20, 20))
    for xy, z in zip(positions[:2, 0], [threshold+1e-7, threshold-1e-7]):
        template = sampled_psf(xx, yy, *xy, kernel)
        image += z*template/np.sqrt(np.sum(template**2))
    frames = [FilterFrame(image, 1, kernel, 0)]
    result = search_fold(frames, FoldPlan(positions, [(20, 20)]), top_k=1, validate_direct=True)
    assert [row["index"] for row in result["detections"]] == [0]
