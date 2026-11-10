import re

import matplotlib.pyplot as plt
import numpy as np
import pybaselines.whittaker
import pytest
from conftest import (
    PBSO4_BACKGROUND_COEFFICIENTS,
    PBSO4_X_RANGE,
    ExternalBackground,
    SyntheticPattern,
)
from numpy.polynomial import chebyshev

from diffpy.cmirietveld.backgrounds import ChebyshevBackground
from diffpy.srfit.fitbase import FitContribution, FitRecipe, Profile


@pytest.mark.parametrize(
    "background_kwargs, expected_names",
    [
        # Test the coefficients a new Chebyshev background has
        # C1: A user makes a background with the default number of terms.
        # Expected: It has six coefficients, b0 to b5, in order, all 0.
        ({}, ["b0", "b1", "b2", "b3", "b4", "b5"]),
        # C2: A user makes a background with three terms.
        # Expected: It has three coefficients, b0 to b2, in order, all 0.
        ({"n_terms": 3}, ["b0", "b1", "b2"]),
    ],
)
def test_coefficients(background_kwargs, expected_names):
    background = ChebyshevBackground(
        "background", x_range=PBSO4_X_RANGE, **background_kwargs
    )
    actual_names = []
    actual_values = []
    for coefficient in background.coefficients:
        actual_names.append(coefficient.name)
        actual_values.append(coefficient.value)
    expected_values = [0.0] * len(expected_names)
    assert actual_names == expected_names
    assert actual_values == expected_values


def test_coefficients_are_background_parameters():
    # Test that the coefficients are the background's own Parameters
    # Expected: Each coefficient is the Parameter of the same name on the
    # background, so setting one changes the background.
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    actual_coefficients = background.coefficients
    expected_coefficients = [background.get(f"b{i}") for i in range(6)]
    assert actual_coefficients == expected_coefficients


@pytest.mark.parametrize(
    "background_kwargs, expected_msg",
    [
        # Test that a background that can't be made raises an error saying
        # why and what to do
        # C1: A user gives a range whose start isn't below its end.
        # 1. The range is the wrong way round, 110 to 16 degrees.
        # 2. The range starts and ends at 16 degrees.
        # Expected: A ValueError names the range, and asks for it as
        # (xmin, xmax) with xmin below xmax.
        (
            {"x_range": (110.0, 16.0)},
            "ChebyshevBackground was given an x_range of (110.0, 16.0), "
            "whose start isn't below its end. Please give the range as "
            "(xmin, xmax), with xmin below xmax.",
        ),
        (
            {"x_range": (16.0, 16.0)},
            "ChebyshevBackground was given an x_range of (16.0, 16.0), "
            "whose start isn't below its end. Please give the range as "
            "(xmin, xmax), with xmin below xmax.",
        ),
        # C2: A user asks for a background with no terms.
        # Expected: A ValueError asks for at least one term.
        (
            {"x_range": PBSO4_X_RANGE, "n_terms": 0},
            "ChebyshevBackground was given n_terms of 0, but a background "
            "needs at least 1 term. Please give n_terms of 1 or more.",
        ),
    ],
)
def test_init_with_bad_arguments(background_kwargs, expected_msg):
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        ChebyshevBackground("background", **background_kwargs)


@pytest.mark.parametrize(
    "tth",
    [
        # Test that the background is the Chebyshev polynomial on the
        # background's own range, whatever range it is calculated on
        # C1: A user calculates the background over its whole range.
        # Expected: It equals numpy's Chebyshev polynomial, with 16 to 110
        # degrees mapped onto -1 to 1.
        np.arange(16.0, 110.0, 0.025),
        # C2: A user narrows the fit to 20 to 60 degrees.
        # Expected: The background still maps 16 to 110 degrees onto -1 to
        # 1, so at each angle it is the same as over the whole range.
        np.arange(20.0, 60.0, 0.025),
        # C3: A user's data start and end exactly at the background's
        # range, 16 and 110 degrees.
        # Expected: The background is calculated at both ends, which map
        # onto -1 and 1.
        np.array([16.0, 110.0]),
    ],
)
def test_call(pbso4_background, tth):
    actual_background = pbso4_background(tth)
    xmin, xmax = PBSO4_X_RANGE
    scaled_tth = 2.0 * (tth - xmin) / (xmax - xmin) - 1.0
    expected_background = chebyshev.chebval(
        scaled_tth, PBSO4_BACKGROUND_COEFFICIENTS
    )
    assert np.allclose(actual_background, expected_background)


@pytest.mark.parametrize(
    "tth, tth_span",
    [
        # Test that calculating the background outside its range raises an
        # error saying why and what to do, since a Chebyshev polynomial
        # grows quickly outside its range and would bend the fit
        # C1: A user's data reach past one end of the background's range.
        # 1. The data start at 10 degrees, below 16 degrees.
        # 2. The data end at 120 degrees, above 110 degrees.
        # Expected: A ValueError names the data's span and the range, and
        # asks for a calculation range within it or a wider background.
        (np.linspace(10.0, 60.0, 2001), "10.0 to 60.0"),
        (np.linspace(60.0, 120.0, 2401), "60.0 to 120.0"),
        # C2: A user's data reach past both ends, from 10 to 120 degrees.
        # Expected: A ValueError names the data's span and the range.
        (np.linspace(10.0, 120.0, 4401), "10.0 to 120.0"),
    ],
)
def test_call_outside_range(pbso4_background, tth, tth_span):
    expected_msg = (
        f"The background was calculated on x values from {tth_span}, "
        "outside its range of 16.0 to 110.0. Please set the profile's "
        "calculation range within the background's range, or make the "
        "background with a range that holds the data."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        pbso4_background(tth)


@pytest.mark.parametrize(
    "points",
    [
        # Test fitting a straight background to points a user gives, which
        # rises from 90 at 16 degrees to 110 at 110 degrees, so it is
        # 100 + 10 T1 on the background's range
        # C1: A user types three points into their script as a list of
        # tuples.
        # Expected: The coefficients are 100 and 10.
        [(16.0, 90.0), (63.0, 100.0), (110.0, 110.0)],
        # C2: A user loads four points from a two-column file.
        # Expected: The coefficients are 100 and 10.
        np.array([[16.0, 90.0], [39.5, 95.0], [86.5, 105.0], [110.0, 110.0]]),
        # C3: A user gives exactly two points, one for each term.
        # Expected: The coefficients are 100 and 10.
        [(16.0, 90.0), (110.0, 110.0)],
    ],
)
def test_fit_to_points(points):
    background = ChebyshevBackground(
        "background", x_range=PBSO4_X_RANGE, n_terms=2
    )
    background.fit_to_points(points)
    actual_coefficients = []
    for coefficient in background.coefficients:
        actual_coefficients.append(coefficient.value)
    expected_coefficients = [100.0, 10.0]
    assert np.allclose(actual_coefficients, expected_coefficients)


def test_fit_to_points_on_known_background(pbso4_background_points):
    # Test that fitting to points on a known six-term background recovers
    # it
    # Expected: After a user picks 15 points on the known background, the
    # coefficients are the known background's.
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    background.fit_to_points(pbso4_background_points)
    actual_coefficients = []
    for coefficient in background.coefficients:
        actual_coefficients.append(coefficient.value)
    expected_coefficients = PBSO4_BACKGROUND_COEFFICIENTS
    assert np.allclose(actual_coefficients, expected_coefficients)


def test_fit_to_points_as_recipe_starting_values(pbso4_background_points):
    # Test that fitting to points sets the starting values of a recipe
    # whose variables are the coefficients
    # Expected: After a user adds the coefficients to a recipe and then
    # fits the background to points, the recipe's values are the fitted
    # coefficients.
    # Add the coefficients of a new background to a recipe
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    recipe = FitRecipe()
    for coefficient in background.coefficients:
        recipe.add_variable(coefficient)
    # Fit the background to 15 points on the known background
    background.fit_to_points(pbso4_background_points)
    actual_values = recipe.get_values()
    expected_values = PBSO4_BACKGROUND_COEFFICIENTS
    assert np.allclose(actual_values, expected_values)


@pytest.mark.parametrize(
    "points, expected_msg",
    [
        # Test that points the background can't be fitted to raise an error
        # saying why and what to do
        # C1: A user gives points that aren't (x, intensity) pairs.
        # 1. Every point has three numbers.
        # 2. One point among pairs has three numbers.
        # 3. Only the x values are given, as a flat list.
        # Expected: A ValueError asks for each point as two numbers.
        (
            [
                (20.0, 150.0, 1.0),
                (40.0, 140.0, 1.0),
                (60.0, 130.0, 1.0),
                (80.0, 120.0, 1.0),
                (90.0, 115.0, 1.0),
                (100.0, 110.0, 1.0),
            ],
            "fit_to_points was given points that aren't all (x, intensity) "
            "pairs. Please give each point as two numbers, an x value and "
            "its intensity.",
        ),
        (
            [
                (20.0, 150.0),
                (40.0, 140.0),
                (60.0, 130.0, 1.0),
                (80.0, 120.0),
                (90.0, 115.0),
                (100.0, 110.0),
            ],
            "fit_to_points was given points that aren't all (x, intensity) "
            "pairs. Please give each point as two numbers, an x value and "
            "its intensity.",
        ),
        (
            [20.0, 40.0, 60.0, 80.0, 90.0, 100.0],
            "fit_to_points was given points that aren't all (x, intensity) "
            "pairs. Please give each point as two numbers, an x value and "
            "its intensity.",
        ),
        # C2: A user picks only four points for a background of six terms.
        # Expected: A ValueError says the coefficients can't be found, and
        # asks for more points or fewer terms.
        (
            [(20.0, 150.0), (40.0, 140.0), (60.0, 130.0), (80.0, 120.0)],
            "fit_to_points was given 4 points, fewer than the background's "
            "6 terms, so the coefficients can't be determined. Please give "
            "at least 6 points, or make the background with fewer terms.",
        ),
        # C3: A user picks points below and above the background's range.
        # 1. A point at 12 degrees, below 16 degrees.
        # 2. Points at 12 and 115 degrees, below 16 and above 110 degrees.
        # Expected: A ValueError names the points outside the range, and
        # asks for points within it or a wider range.
        (
            [
                (12.0, 150.0),
                (30.0, 145.0),
                (45.0, 140.0),
                (60.0, 135.0),
                (75.0, 130.0),
                (90.0, 125.0),
            ],
            "fit_to_points was given points outside the background's range "
            "of 16.0 to 110.0: 12.0. Please give points within the range, "
            "or make the background with a range that holds them.",
        ),
        (
            [
                (12.0, 150.0),
                (30.0, 145.0),
                (45.0, 140.0),
                (60.0, 135.0),
                (75.0, 130.0),
                (115.0, 125.0),
            ],
            "fit_to_points was given points outside the background's range "
            "of 16.0 to 110.0: 12.0, 115.0. Please give points within the "
            "range, or make the background with a range that holds them.",
        ),
    ],
)
def test_fit_to_points_with_bad_points(points, expected_msg):
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        background.fit_to_points(points)


def test_fit_to_data(synthetic_pattern):
    # Test the automatic background on a pattern whose background is known
    # Expected: After a user fits the background to a pattern of sharp
    # peaks on a known background, the background is within 1% of the
    # known one everywhere.
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    background.fit_to_data(synthetic_pattern.x, synthetic_pattern.y)
    actual_background = background(synthetic_pattern.x)
    expected_background = synthetic_pattern.background
    assert np.allclose(actual_background, expected_background, rtol=0.01)


@pytest.mark.parametrize(
    "fit_to_data_kwargs, expected_log_lambda",
    [
        # Test that the automatic background estimates the same points as
        # GSAS-II's automatic background: pybaselines' arpls with 10
        # iterations, sampled at every len(y) // 100-th point
        # C1: A user leaves the smoothness at its default.
        # Expected: The points are GSAS-II's, with its default log10 of
        # lambda for 3760 points, 5.75.
        ({}, 5.75),
        # C2: A user makes the baseline stiffer with a smoothness of 7.
        # Expected: The points are GSAS-II's with a log10 of lambda of 7.
        ({"smoothness": 7.0}, 7.0),
    ],
)
def test_fit_to_data_points_match_gsasii(
    synthetic_pattern, fit_to_data_kwargs, expected_log_lambda
):
    # Fit the background to the pattern
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    actual_points = background.fit_to_data(
        synthetic_pattern.x, synthetic_pattern.y, **fit_to_data_kwargs
    )
    # Estimate the points as GSAS-II's autoBkgCalc and calc_autobkg do
    baseline = pybaselines.whittaker.arpls(
        synthetic_pattern.y, lam=int(10**expected_log_lambda), max_iter=10
    )[0]
    step = len(synthetic_pattern.y) // 100
    expected_points = np.column_stack(
        (synthetic_pattern.x[::step], baseline[::step])
    )
    assert np.allclose(actual_points, expected_points)


def test_fit_to_data_wider_than_range():
    # Test the automatic background on data that extend past the
    # background's range, as a measured file usually does
    # Expected: The baseline is estimated from all the data from 10 to
    # 120 degrees, and only its points between 16 and 110 degrees are
    # fitted and returned.
    wide_pattern = SyntheticPattern(x_min=10.0, x_max=120.0)
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    actual_points = background.fit_to_data(wide_pattern.x, wide_pattern.y)
    # Estimate GSAS-II's points from all 4400 points of data, with its
    # default log10 of lambda for them, 5.95, then keep those inside the
    # range
    baseline = pybaselines.whittaker.arpls(
        wide_pattern.y, lam=int(10**5.95), max_iter=10
    )[0]
    step = len(wide_pattern.y) // 100
    all_points = np.column_stack((wide_pattern.x[::step], baseline[::step]))
    xmin, xmax = PBSO4_X_RANGE
    inside_range = (all_points[:, 0] >= xmin) & (all_points[:, 0] <= xmax)
    expected_points = all_points[inside_range]
    assert np.allclose(actual_points, expected_points)


@pytest.mark.parametrize(
    "fixed_point_weight",
    [
        # Test that a fixed point's weight is the number of estimated
        # points it counts as
        # C1: A user adds a fixed point at 63 degrees with the default
        # weight of 1.
        # Expected: The coefficients are those fitted to the estimated
        # points with the fixed point added once.
        1,
        # C2: A user adds the same fixed point with a weight of 3.
        # Expected: The coefficients are those fitted to the estimated
        # points with the fixed point added three times.
        3,
    ],
)
def test_fit_to_data_with_fixed_points(synthetic_pattern, fixed_point_weight):
    fixed_points = [(63.0, 200.0)]
    # Fit the background to the pattern with the fixed point
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    estimated_points = background.fit_to_data(
        synthetic_pattern.x,
        synthetic_pattern.y,
        fixed_points=fixed_points,
        fixed_point_weight=fixed_point_weight,
    )
    actual_coefficients = []
    for coefficient in background.coefficients:
        actual_coefficients.append(coefficient.value)
    # Fit a second background to the estimated points plus copies of the
    # fixed point
    repeated_fixed_points = fixed_points * fixed_point_weight
    all_points = np.vstack((estimated_points, repeated_fixed_points))
    expected_background = ChebyshevBackground(
        "background", x_range=PBSO4_X_RANGE
    )
    expected_background.fit_to_points(all_points)
    expected_coefficients = []
    for coefficient in expected_background.coefficients:
        expected_coefficients.append(coefficient.value)
    assert np.allclose(actual_coefficients, expected_coefficients)


def test_fit_to_data_with_heavy_fixed_point(synthetic_pattern):
    # Test that a heavily weighted fixed point overrides the estimate
    # Expected: After a user puts a fixed point at 200 counts at 63
    # degrees, where the estimate is about 143, with a weight of 1000,
    # the background at 63 degrees is within 1% of 200.
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    background.fit_to_data(
        synthetic_pattern.x,
        synthetic_pattern.y,
        fixed_points=[(63.0, 200.0)],
        fixed_point_weight=1000,
    )
    actual_background = background(np.array([63.0]))
    expected_background = np.array([200.0])
    assert np.allclose(actual_background, expected_background, rtol=0.01)


@pytest.mark.parametrize(
    "fit_to_data_kwargs, expected_msg",
    [
        # Test that fixed points the automatic background can't use raise
        # an error saying why and what to do
        # C1: A user gives a weight that isn't above 0.
        # 1. A weight of 0.
        # 2. A weight of -2.
        # 3. A weight of 0, with no fixed points.
        # Expected: A ValueError asks for a weight above 0.
        (
            {"fixed_points": [(63.0, 200.0)], "fixed_point_weight": 0.0},
            "fit_to_data was given a fixed_point_weight of 0.0, which is "
            "not above 0. Please give a weight above 0, such as 1 to count "
            "each fixed point as one estimated point.",
        ),
        (
            {"fixed_points": [(63.0, 200.0)], "fixed_point_weight": -2.0},
            "fit_to_data was given a fixed_point_weight of -2.0, which is "
            "not above 0. Please give a weight above 0, such as 1 to count "
            "each fixed point as one estimated point.",
        ),
        (
            {"fixed_point_weight": 0.0},
            "fit_to_data was given a fixed_point_weight of 0.0, which is "
            "not above 0. Please give a weight above 0, such as 1 to count "
            "each fixed point as one estimated point.",
        ),
        # C2: A user gives a fixed point that isn't an (x, intensity) pair.
        # Expected: A ValueError names fixed_points, so it isn't mistaken
        # for x or y, and asks for each point as two numbers.
        (
            {"fixed_points": [(63.0, 200.0, 1.0)]},
            "fit_to_data was given fixed_points that aren't all "
            "(x, intensity) pairs. Please give each fixed point as two "
            "numbers, an x value and its intensity.",
        ),
        # C3: A user gives a fixed point at 115 degrees, above the
        # background's range.
        # Expected: A ValueError names fixed_points and the point, and asks
        # for fixed points within the range or a wider range.
        (
            {"fixed_points": [(63.0, 200.0), (115.0, 150.0)]},
            "fit_to_data was given fixed_points outside the background's "
            "range of 16.0 to 110.0: 115.0. Please give fixed points within "
            "the range, or make the background with a range that holds "
            "them.",
        ),
    ],
)
def test_fit_to_data_with_bad_fixed_points(
    synthetic_pattern, fit_to_data_kwargs, expected_msg
):
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        background.fit_to_data(
            synthetic_pattern.x, synthetic_pattern.y, **fit_to_data_kwargs
        )


@pytest.mark.parametrize(
    "x, y, expected_msg",
    [
        # Test that data the automatic background can't use raise an error
        # saying why and what to do
        # C1: A user gives one more x value than intensities.
        # Expected: A ValueError gives both counts, and asks for one
        # intensity for each x value.
        (
            np.linspace(16.0, 110.0, 3761),
            np.full(3760, 100.0),
            "fit_to_data was given 3761 x values and 3760 intensities. "
            "Please give one intensity for each x value.",
        ),
        # C2: A user's data run from 109.5 to 130 degrees, so only three
        # of the estimated points, at 109.5, 109.7 and 109.9 degrees, are
        # inside the background's range.
        # Expected: A ValueError says the coefficients can't be found, and
        # asks for data covering more of the range, fixed points, or fewer
        # terms.
        (
            np.linspace(109.5, 130.0, 821),
            np.full(821, 100.0),
            "fit_to_data has 3 points inside the background's range of "
            "16.0 to 110.0, fewer than its 6 terms, so the coefficients "
            "can't be determined. Please give data that cover more of the "
            "range, add fixed points, or make the background with fewer "
            "terms.",
        ),
    ],
)
def test_fit_to_data_with_bad_data(x, y, expected_msg):
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        background.fit_to_data(x, y)


def test_plot_fit_after_fit_to_data(synthetic_pattern):
    # Test the plot a user checks after the automatic background
    # Expected: The plot shows the data, the background curve on the data's
    # x, the estimated points and the fixed points, each as given.
    # Fit the background with a fixed point and plot it
    fixed_points = np.array([(63.0, 150.0)])
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    estimated_points = background.fit_to_data(
        synthetic_pattern.x, synthetic_pattern.y, fixed_points=fixed_points
    )
    fig, ax = background.plot_fit(show=False, return_fig=True)
    # Collect what was drawn, by label
    actual_lines = {}
    for line in ax.lines:
        actual_lines[line.get_label()] = line.get_xydata()
    expected_lines = {
        "Observed": np.column_stack(
            (synthetic_pattern.x, synthetic_pattern.y)
        ),
        "Background": np.column_stack(
            (synthetic_pattern.x, background(synthetic_pattern.x))
        ),
        "Estimated points": estimated_points,
        "Fixed points": fixed_points,
    }
    assert list(actual_lines) == list(expected_lines)
    for label, expected_xydata in expected_lines.items():
        assert np.allclose(actual_lines[label], expected_xydata)
    plt.close(fig)


def test_plot_fit_without_fixed_points(synthetic_pattern):
    # Test the plot after the automatic background with no fixed points
    # Expected: The plot shows the data, the background curve and the
    # estimated points, and no fixed points.
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    background.fit_to_data(synthetic_pattern.x, synthetic_pattern.y)
    fig, ax = background.plot_fit(show=False, return_fig=True)
    actual_labels = [line.get_label() for line in ax.lines]
    expected_labels = ["Observed", "Background", "Estimated points"]
    assert actual_labels == expected_labels
    plt.close(fig)


def test_plot_fit_after_refinement(synthetic_pattern):
    # Test that the plot shows the current coefficients, as after a
    # refinement
    # Expected: After a refinement changes b0 to 160, the background curve
    # equals the background with b0 = 160, on the data of the last fit.
    # Fit the background, then change b0 as a refinement would
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    background.fit_to_data(synthetic_pattern.x, synthetic_pattern.y)
    background.b0.set_value(160.0)
    fig, ax = background.plot_fit(show=False, return_fig=True)
    actual_curve = ax.lines[1].get_xydata()
    expected_curve = np.column_stack(
        (synthetic_pattern.x, background(synthetic_pattern.x))
    )
    assert np.allclose(actual_curve, expected_curve)
    plt.close(fig)


def test_plot_fit_after_fit_to_points(pbso4_background_points):
    # Test the plot after fitting to the user's points only, which gives
    # no data
    # Expected: The plot shows the background curve from 16 to 110
    # degrees and the user's points, and no data.
    # Fit a new background to 15 points on the known background
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    background.fit_to_points(pbso4_background_points)
    fig, ax = background.plot_fit(show=False, return_fig=True)
    # Check the labels, the curve's ends and values, and the points
    actual_labels = [line.get_label() for line in ax.lines]
    expected_labels = ["Background", "Fixed points"]
    curve_x, curve_y = ax.lines[0].get_data()
    actual_curve_range = (curve_x[0], curve_x[-1])
    expected_curve_range = PBSO4_X_RANGE
    actual_points = ax.lines[1].get_xydata()
    expected_points = pbso4_background_points
    assert actual_labels == expected_labels
    assert actual_curve_range == expected_curve_range
    assert np.allclose(curve_y, background(curve_x))
    assert np.allclose(actual_points, expected_points)
    plt.close(fig)


def test_plot_fit_after_fit_to_points_with_data(
    pbso4_background_points, synthetic_pattern
):
    # Test the plot after fitting to the user's points, with data given
    # Expected: The plot shows the given data under the background curve
    # and the user's points.
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    background.fit_to_points(pbso4_background_points)
    fig, ax = background.plot_fit(
        synthetic_pattern.x, synthetic_pattern.y, show=False, return_fig=True
    )
    actual_labels = [line.get_label() for line in ax.lines]
    expected_labels = ["Observed", "Background", "Fixed points"]
    actual_data = ax.lines[0].get_xydata()
    expected_data = np.column_stack((synthetic_pattern.x, synthetic_pattern.y))
    assert actual_labels == expected_labels
    assert np.allclose(actual_data, expected_data)
    plt.close(fig)


def test_plot_fit_after_fit_to_data_then_fit_to_points(
    pbso4_background_points, synthetic_pattern
):
    # Test that the plot shows only the last fit, after a user edits the
    # estimated points and fits to them
    # Expected: The plot shows the background curve over 16 to 110
    # degrees and the points just given, and none of the data, estimated
    # points or fixed points of the earlier fit_to_data.
    # Fit the background to the pattern with a fixed point
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    background.fit_to_data(
        synthetic_pattern.x, synthetic_pattern.y, fixed_points=[(63.0, 150.0)]
    )
    # Fit it again to 15 points on the known background, and plot it
    background.fit_to_points(pbso4_background_points)
    fig, ax = background.plot_fit(show=False, return_fig=True)
    actual_labels = [line.get_label() for line in ax.lines]
    expected_labels = ["Background", "Fixed points"]
    actual_points = ax.lines[1].get_xydata()
    expected_points = pbso4_background_points
    assert actual_labels == expected_labels
    assert np.allclose(actual_points, expected_points)
    plt.close(fig)


def test_plot_fit_after_second_fit_to_data(synthetic_pattern):
    # Test that the plot shows only the last fit, after a user refits the
    # automatic background on other data without fixed points
    # Expected: The plot shows the second data and their estimated points,
    # and no fixed points from the first fit.
    # Fit the background to the pattern with a fixed point
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    background.fit_to_data(
        synthetic_pattern.x, synthetic_pattern.y, fixed_points=[(63.0, 150.0)]
    )
    # Fit it again to 20 to 100 degrees of the pattern, and plot it
    narrow_pattern = SyntheticPattern(x_min=20.0, x_max=100.0)
    estimated_points = background.fit_to_data(
        narrow_pattern.x, narrow_pattern.y
    )
    fig, ax = background.plot_fit(show=False, return_fig=True)
    actual_lines = {}
    for line in ax.lines:
        actual_lines[line.get_label()] = line.get_xydata()
    expected_lines = {
        "Observed": np.column_stack((narrow_pattern.x, narrow_pattern.y)),
        "Background": np.column_stack(
            (narrow_pattern.x, background(narrow_pattern.x))
        ),
        "Estimated points": estimated_points,
    }
    assert list(actual_lines) == list(expected_lines)
    for label, expected_xydata in expected_lines.items():
        assert np.allclose(actual_lines[label], expected_xydata)
    plt.close(fig)


def test_plot_fit_with_data_wider_than_range():
    # Test the plot after the automatic background on data that extend
    # past the background's range
    # Expected: The data are drawn from 10 to 120 degrees, and the
    # background curve only at the data's x values from 16 to 110
    # degrees, where it is calculated.
    wide_pattern = SyntheticPattern(x_min=10.0, x_max=120.0)
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    background.fit_to_data(wide_pattern.x, wide_pattern.y)
    fig, ax = background.plot_fit(show=False, return_fig=True)
    actual_data = ax.lines[0].get_xydata()
    expected_data = np.column_stack((wide_pattern.x, wide_pattern.y))
    actual_curve = ax.lines[1].get_xydata()
    xmin, xmax = PBSO4_X_RANGE
    inside_range = (wide_pattern.x >= xmin) & (wide_pattern.x <= xmax)
    curve_x = wide_pattern.x[inside_range]
    expected_curve = np.column_stack((curve_x, background(curve_x)))
    assert np.allclose(actual_data, expected_data)
    assert np.allclose(actual_curve, expected_curve)
    plt.close(fig)


def test_plot_fit_on_given_axes(synthetic_pattern):
    # Test drawing the plot on axes a user made, for example next to a
    # plot of the refinement
    # Expected: The background is drawn on the given axes, which are
    # returned.
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    background.fit_to_data(synthetic_pattern.x, synthetic_pattern.y)
    fig, given_ax = plt.subplots()
    actual_fig, actual_ax = background.plot_fit(ax=given_ax, return_fig=True)
    actual_labels = [line.get_label() for line in given_ax.lines]
    expected_labels = ["Observed", "Background", "Estimated points"]
    assert actual_ax == given_ax
    assert actual_fig == fig
    assert actual_labels == expected_labels
    plt.close(fig)


def test_plot_fit_without_return_fig(synthetic_pattern):
    # Test drawing the plot on given axes without asking for the figure
    # Expected: Nothing is returned, and the background is drawn on the
    # given axes.
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    background.fit_to_data(synthetic_pattern.x, synthetic_pattern.y)
    fig, given_ax = plt.subplots()
    actual_return = background.plot_fit(ax=given_ax)
    expected_return = None
    actual_labels = [line.get_label() for line in given_ax.lines]
    expected_labels = ["Observed", "Background", "Estimated points"]
    assert actual_return == expected_return
    assert actual_labels == expected_labels
    plt.close(fig)


@pytest.mark.parametrize(
    "plot_fit_kwargs, expected_msg",
    [
        # Test that a plot without the data to draw raises an error saying
        # why and what to do
        # C1: A user plots a background that hasn't been fitted, with no
        # data.
        # Expected: A ValueError says there is nothing to draw on, and asks
        # the user to fit the background first or give data.
        (
            {},
            "plot_fit has no data to draw the background on, because the "
            "background hasn't been fitted yet. Please call fit_to_data or "
            "fit_to_points first, or give x and y.",
        ),
        # C2: A user gives only one of x and y.
        # 1. Only x is given.
        # 2. Only y is given.
        # Expected: A ValueError names the one given, and asks for both or
        # neither.
        (
            {"x": np.linspace(16.0, 110.0, 3761)},
            "plot_fit was given x but not y. Please give both x and y, or "
            "neither to draw the data of the last fit.",
        ),
        (
            {"y": np.full(3761, 100.0)},
            "plot_fit was given y but not x. Please give both x and y, or "
            "neither to draw the data of the last fit.",
        ),
        # C3: A user gives one more x value than intensities.
        # Expected: A ValueError gives both counts, and asks for one
        # intensity for each x value.
        (
            {"x": np.linspace(16.0, 110.0, 3761), "y": np.full(3760, 100.0)},
            "plot_fit was given 3761 x values and 3760 intensities. Please "
            "give one intensity for each x value.",
        ),
    ],
)
def test_plot_fit_with_bad_data(plot_fit_kwargs, expected_msg):
    background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        background.plot_fit(**plot_fit_kwargs)


def test_register_calculator(pbso4_background, pbso4_contribution):
    # Test that registering the background puts its coefficients in the
    # FitContribution, so a recipe can refine them
    # Expected: The contribution holds the background's coefficients under
    # its name.
    pbso4_contribution.register_calculator(pbso4_background, argnames=["x"])
    actual_coefficients = []
    for index in range(6):
        coefficient = pbso4_contribution.background.get(f"b{index}")
        actual_coefficients.append(coefficient)
    expected_coefficients = pbso4_background.coefficients
    assert actual_coefficients == expected_coefficients


def test_evaluate_equation_background(
    pbso4_background, pbso4_contribution, pbso4_pattern
):
    # Test evaluating the background alone in a FitContribution, as a user
    # does to look at it during a fit
    # Expected: The background evaluated by the contribution equals the
    # background calculated on the profile's x.
    pbso4_contribution.register_calculator(pbso4_background, argnames=["x"])
    actual_background = pbso4_contribution.evaluate_equation("background")
    expected_background = pbso4_background(pbso4_pattern.tth)
    assert np.allclose(actual_background, expected_background)


def test_evaluate_with_bragg_and_background(
    pbso4_background, pbso4_contribution, pbso4_generator, pbso4_pattern
):
    # Test a FitContribution whose equation adds the background to the
    # Bragg intensity
    # Expected: The contribution's value is the generator's pattern plus
    # the background, at every point of the profile.
    pbso4_contribution.register_calculator(pbso4_background, argnames=["x"])
    pbso4_contribution.set_equation("bragg + background")
    actual_pattern = pbso4_contribution.evaluate()
    expected_pattern = pbso4_generator(pbso4_pattern.tth) + pbso4_background(
        pbso4_pattern.tth
    )
    assert np.allclose(actual_pattern, expected_pattern)


def test_evaluate_with_profile_wider_than_background(pbso4_background):
    # Test a FitContribution whose profile runs past the background's
    # range, as when a user forgets to set the calculation range
    # Expected: Evaluating the contribution raises the background's
    # ValueError, which names the profile's span and the range, before
    # any refinement runs.
    # Put the background in a contribution on 10 to 120 degrees
    profile = Profile()
    tth = np.linspace(10.0, 120.0, 4401)
    profile.set_observed_profile(tth, np.zeros_like(tth))
    contribution = FitContribution("xray")
    contribution.set_profile(profile)
    contribution.register_calculator(pbso4_background, argnames=["x"])
    contribution.set_equation("background")
    expected_msg = (
        "The background was calculated on x values from 10.0 to 120.0, "
        "outside its range of 16.0 to 110.0. Please set the profile's "
        "calculation range within the background's range, or make the "
        "background with a range that holds the data."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        contribution.evaluate()


def test_evaluate_with_external_background(
    pbso4_contribution, pbso4_generator, pbso4_pattern
):
    # Test a FitContribution with a background written outside the
    # package, as a subclass of Background
    # Expected: After a user sets its level to 100, the contribution's
    # value is the generator's pattern plus 100, and its level is a
    # Parameter of the contribution.
    external_background = ExternalBackground("background")
    external_background.level.set_value(100.0)
    pbso4_contribution.register_calculator(external_background, argnames=["x"])
    pbso4_contribution.set_equation("bragg + background")
    actual_pattern = pbso4_contribution.evaluate()
    expected_pattern = pbso4_generator(pbso4_pattern.tth) + 100.0
    actual_level = pbso4_contribution.background.level
    expected_level = external_background.level
    assert np.allclose(actual_pattern, expected_pattern)
    assert actual_level == expected_level
