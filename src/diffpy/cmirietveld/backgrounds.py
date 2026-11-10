#!/usr/bin/env python
##############################################################################
#
# (c) 2026 diffpy.cmirietveld contributors.
# All rights reserved.
#
# File coded by: Caden Myers, Simon Billinge, Billinge Group members.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/diffpy/diffpy.cmirietveld/graphs/contributors
#
# See LICENSE.rst for license information.
#
##############################################################################
"""Background calculators for the equation of a FitContribution.

A background describes a measured dataset rather than a phase. It is
registered in a FitContribution with ``register_calculator`` and added
to the Bragg intensity in the equation, for example ``"bragg +
background"``.
"""

__all__ = ["Background", "ChebyshevBackground"]

import numpy as np
from numpy.polynomial import chebyshev
from pybaselines.whittaker import arpls

from diffpy.srfit.fitbase import Calculator

N_CURVE_POINTS = 1000


class Background(Calculator):
    """Base class of the backgrounds.

    Subclasses write ``__call__`` in NumPy and create their Parameters.

    Parameters
    ----------
    name : str
        The name of the background. It must be a valid attribute name.
    """

    def __call__(self, x):
        """Calculate the background.

        Parameters
        ----------
        x : ndarray
            The values of the data axis.

        Returns
        -------
        ndarray
            The background at each value of the data axis.
        """
        raise NotImplementedError


class ChebyshevBackground(Background):
    """Background described by a Chebyshev polynomial.

    The range given at construction, ``x_range``, is mapped onto
    [-1, 1], as GSAS-II maps its limits. The range is fixed, so the
    coefficients keep their meaning when the calculation range of the
    fit changes, and the background can be fitted to points before it is
    first calculated. The background is only calculated inside the
    range, ends included, since outside it the polynomial grows quickly
    and would bend the fit.

    The background is put in a FitContribution's equation with
    ``contribution.register_calculator(background, argnames=["tth"])``,
    where "tth" is the name of the profile's x, after which its
    Parameters sit under ``contribution.background``.

    Parameters
    ----------
    name : str
        The name of the background, used in the FitContribution equation,
        for example "bragg + background". It must be a valid attribute
        name.
    x_range : tuple of float
        The (xmin, xmax) range of the data axis mapped onto [-1, 1].
    n_terms : int, optional
        The number of Chebyshev terms (default 6).

    Attributes
    ----------
    x_range : tuple of float
        The (xmin, xmax) range of the data axis mapped onto [-1, 1].
    n_terms : int
        The number of Chebyshev terms.

    Managed Parameters
    ------------------
    b0, b1, ...
        The Chebyshev coefficients, listed in order by ``coefficients``.
        They start at 0.

    Raises
    ------
    ValueError
        If the start of ``x_range`` is not below its end, or ``n_terms``
        is below 1.
    """

    def __init__(self, name, x_range, n_terms=6):
        xmin, xmax = x_range
        if not xmin < xmax:
            raise ValueError(
                f"ChebyshevBackground was given an x_range of {x_range}, "
                "whose start isn't below its end. Please give the range as "
                "(xmin, xmax), with xmin below xmax."
            )
        if n_terms < 1:
            raise ValueError(
                f"ChebyshevBackground was given n_terms of {n_terms}, but a "
                "background needs at least 1 term. Please give n_terms of 1 "
                "or more."
            )
        Background.__init__(self, name)
        self.x_range = (xmin, xmax)
        self.n_terms = n_terms
        for index in range(n_terms):
            self._new_parameter(f"b{index}", 0.0)
        self._last_data = None
        self._estimated_points = None
        self._fixed_points = None

    @property
    def coefficients(self):
        """The Parameters of the Chebyshev coefficients, in order."""
        coefficients = []
        for index in range(self.n_terms):
            coefficients.append(self.get(f"b{index}"))
        return coefficients

    def __call__(self, x):
        """Calculate the background.

        Parameters
        ----------
        x : ndarray
            The values of the data axis, for example 2theta in degrees,
            within ``x_range``.

        Returns
        -------
        ndarray
            The background at each value of the data axis.

        Raises
        ------
        ValueError
            If an x value is outside ``x_range``, for example because the
            profile's calculation range is wider than the background's.
        """
        x = np.asarray(x, dtype=float)
        xmin, xmax = self.x_range
        if x.size and (x.min() < xmin or x.max() > xmax):
            raise ValueError(
                "The background was calculated on x values from "
                f"{_format_x(x.min())} to {_format_x(x.max())}, outside its "
                f"range of {self._format_range()}. Please set the profile's "
                "calculation range within the background's range, or make "
                "the background with a range that holds the data."
            )
        background = chebyshev.chebval(
            self._scale_x(x), self._get_coefficient_values()
        )
        return background

    def fit_to_points(self, points):
        """Fit the coefficients to the user's points.

        The points are (x, intensity) pairs chosen between the peaks,
        like GSAS-II's fixed background points, and all count the same.
        The coefficients are fitted to them by linear least squares and
        set as the values of their Parameters, which become the
        refinement's starting values. ``fit_to_data`` estimates the points
        from the measured pattern instead.

        Parameters
        ----------
        points : array_like
            The (x, intensity) pairs, shaped (n, 2): a list of tuples, or
            a two-column array such as ``np.loadtxt`` reads. The x values
            are within ``x_range``.

        Raises
        ------
        ValueError
            If a point is not an (x, intensity) pair, there are fewer
            points than terms, or a point is outside ``x_range``.
        """
        points = self._read_points(points, "fit_to_points", "points", "point")
        if len(points) < self.n_terms:
            raise ValueError(
                f"fit_to_points was given {len(points)} points, fewer than "
                f"the background's {self.n_terms} terms, so the coefficients "
                f"can't be determined. Please give at least {self.n_terms} "
                "points, or make the background with fewer terms."
            )
        self._fit_coefficients(points, np.ones(len(points)))
        self._last_data = None
        self._estimated_points = None
        self._fixed_points = points

    def fit_to_data(
        self,
        x,
        y,
        fixed_points=None,
        fixed_point_weight=1.0,
        smoothness=None,
    ):
        """Fit the coefficients to a measured pattern.

        This is GSAS-II's automatic background. The baseline under the
        pattern is estimated with pybaselines' ``arpls``, and evenly
        spaced points are taken from it, every ``len(y) // 100``-th as in
        GSAS-II. The user's ``fixed_points`` are added, and the
        coefficients are fitted to all the points inside ``x_range`` by
        weighted linear least squares and set as the values of their
        Parameters, which become the refinement's starting values.

        The baseline is estimated from the data alone, so the background
        can be fitted, and checked with ``plot_fit``, before any structure
        is set up.

        Parameters
        ----------
        x : array_like
            The values of the data axis of the pattern, for example 2theta
            in degrees.
        y : array_like
            The measured intensity at each x value.
        fixed_points : array_like, optional
            The user's (x, intensity) pairs, shaped (n, 2), within
            ``x_range``. They are fitted with the estimated points.
        fixed_point_weight : float, optional
            The number of estimated points each fixed point counts as
            (default 1, so every point counts the same, as in GSAS-II). A
            larger weight lets the fixed points override the estimate
            where it is wrong.
        smoothness : float, optional
            The log10 of ``arpls``'s smoothness, lambda. A larger value
            gives a stiffer baseline that bends less into the peaks, and a
            smaller one follows a curved background more closely. The
            default, None, uses GSAS-II's value for the number of points,
            min(10, (int(10 * log10(n)**1.5) - 9.5) / 10).

        Returns
        -------
        ndarray
            The estimated points used in the fit, inside ``x_range``,
            shaped (n, 2), without the fixed points.

        Raises
        ------
        ValueError
            If ``x`` and ``y`` have different lengths, fewer points than
            terms are inside ``x_range``, ``fixed_point_weight`` is not
            above 0 (with or without fixed points), or a fixed point is
            not an (x, intensity) pair or is outside ``x_range``.
        """
        x, y = _read_data(x, y, "fit_to_data")
        if not fixed_point_weight > 0:
            raise ValueError(
                "fit_to_data was given a fixed_point_weight of "
                f"{fixed_point_weight}, which is not above 0. Please give a "
                "weight above 0, such as 1 to count each fixed point as one "
                "estimated point."
            )
        if fixed_points is None:
            fixed_points = np.empty((0, 2))
        else:
            fixed_points = self._read_points(
                fixed_points, "fit_to_data", "fixed_points", "fixed point"
            )
        if smoothness is None:
            smoothness = min(
                10, (int(10 * np.log10(len(y)) ** 1.5) - 9.5) / 10
            )
        baseline = arpls(y, lam=int(10**smoothness), max_iter=10)[0]
        step = max(len(y) // 100, 1)
        sampled_points = np.column_stack((x[::step], baseline[::step]))
        xmin, xmax = self.x_range
        inside_range = (sampled_points[:, 0] >= xmin) & (
            sampled_points[:, 0] <= xmax
        )
        estimated_points = sampled_points[inside_range]
        n_points = len(estimated_points) + len(fixed_points)
        if n_points < self.n_terms:
            raise ValueError(
                f"fit_to_data has {n_points} points inside the background's "
                f"range of {self._format_range()}, fewer than its "
                f"{self.n_terms} terms, so the coefficients can't be "
                "determined. Please give data that cover more of the range, "
                "add fixed points, or make the background with fewer terms."
            )
        all_points = np.vstack((estimated_points, fixed_points))
        weights = np.concatenate(
            (
                np.ones(len(estimated_points)),
                np.full(len(fixed_points), np.sqrt(fixed_point_weight)),
            )
        )
        self._fit_coefficients(all_points, weights)
        self._last_data = (x, y)
        self._estimated_points = estimated_points
        self._fixed_points = fixed_points if len(fixed_points) else None
        return estimated_points

    def plot_fit(self, x=None, y=None, ax=None, show=True, return_fig=False):
        """Plot the background over the data.

        The background curve is drawn from the current coefficients, so
        after a refinement it shows the refined background. The plot
        shows the last fit only: each ``fit_to_data`` or
        ``fit_to_points`` replaces the data and points drawn before. The
        data are those of the last ``fit_to_data``, unless ``x`` and
        ``y`` are given. The estimated points and the fixed points are
        drawn with different markers. After ``fit_to_points``, which has
        no data, the curve is drawn over ``x_range`` with the points.
        The data are drawn in full, and the curve only at their x values
        inside ``x_range``.

        Like srfit's ``FitRecipe.plot_recipe``, the plot uses the
        bg-mpl-stylesheets style, and is shown if ``show`` is True and no
        ``ax`` is given.

        Parameters
        ----------
        x : array_like, optional
            The values of the data axis of the pattern to draw.
        y : array_like, optional
            The measured intensity at each x value.
        ax : matplotlib.axes.Axes, optional
            The axes to draw on. By default a new figure is made.
        show : bool, optional
            The flag to show the new figure (default True). It has no
            effect when ``ax`` is given, and False suits scripts that save
            the figure instead.
        return_fig : bool, optional
            The flag to return the figure and axes (default False).

        Returns
        -------
        fig, ax : tuple
            The figure and axes, returned only if ``return_fig`` is True.

        Raises
        ------
        ValueError
            If the background has not been fitted and no ``x`` and ``y``
            are given, only one of ``x`` and ``y`` is given, or they have
            different lengths.
        """
        if (x is None) != (y is None):
            given_name, missing_name = ("x", "y") if y is None else ("y", "x")
            raise ValueError(
                f"plot_fit was given {given_name} but not {missing_name}. "
                "Please give both x and y, or neither to draw the data of the "
                "last fit."
            )
        if x is not None:
            data = _read_data(x, y, "plot_fit")
        elif self._last_data is not None:
            data = self._last_data
        elif self._fixed_points is not None:
            data = None
        else:
            raise ValueError(
                "plot_fit has no data to draw the background on, because the "
                "background hasn't been fitted yet. Please call fit_to_data "
                "or fit_to_points first, or give x and y."
            )
        xmin, xmax = self.x_range
        if data is None:
            curve_x = np.linspace(xmin, xmax, N_CURVE_POINTS)
        else:
            data_x = data[0]
            curve_x = data_x[(data_x >= xmin) & (data_x <= xmax)]
        import matplotlib.pyplot as plt
        from bg_mpl_stylesheets.styles import all_styles

        with plt.style.context(all_styles["bg-style"]):
            if ax is None:
                fig = plt.figure()
                current_ax = fig.add_subplot(111)
            else:
                current_ax = ax
                fig = current_ax.figure
            if data is not None:
                current_ax.plot(data[0], data[1], "-", label="Observed")
            current_ax.plot(curve_x, self(curve_x), "-", label="Background")
            if self._estimated_points is not None and data is not None:
                current_ax.plot(
                    self._estimated_points[:, 0],
                    self._estimated_points[:, 1],
                    "o",
                    label="Estimated points",
                )
            if self._fixed_points is not None:
                current_ax.plot(
                    self._fixed_points[:, 0],
                    self._fixed_points[:, 1],
                    "D",
                    label="Fixed points",
                )
            current_ax.legend(frameon=True)
            fig.tight_layout()
            if show and ax is None:
                plt.show()
        if return_fig:
            return fig, current_ax

    def _scale_x(self, x):
        xmin, xmax = self.x_range
        scaled_x = 2.0 * (x - xmin) / (xmax - xmin) - 1.0
        return scaled_x

    def _get_coefficient_values(self):
        values = []
        for coefficient in self.coefficients:
            values.append(coefficient.value)
        return values

    def _format_range(self):
        xmin, xmax = self.x_range
        formatted_range = f"{_format_x(xmin)} to {_format_x(xmax)}"
        return formatted_range

    def _read_points(self, points, method_name, argument_name, point_noun):
        try:
            points = np.asarray(points, dtype=float)
        except ValueError:
            points = None
        if points is None or points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"{method_name} was given {argument_name} that aren't all "
                f"(x, intensity) pairs. Please give each {point_noun} as two "
                "numbers, an x value and its intensity."
            )
        xmin, xmax = self.x_range
        outside_x = points[(points[:, 0] < xmin) | (points[:, 0] > xmax), 0]
        if len(outside_x):
            formatted_x = ", ".join(_format_x(value) for value in outside_x)
            raise ValueError(
                f"{method_name} was given {argument_name} outside the "
                f"background's range of {self._format_range()}: "
                f"{formatted_x}. Please give {point_noun}s within the range, "
                "or make the background with a range that holds them."
            )
        return points

    def _fit_coefficients(self, points, weights):
        values = chebyshev.chebfit(
            self._scale_x(points[:, 0]),
            points[:, 1],
            self.n_terms - 1,
            w=weights,
        )
        for coefficient, value in zip(self.coefficients, values):
            coefficient.set_value(value)


def _read_data(x, y, method_name):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError(
            f"{method_name} was given {len(x)} x values and {len(y)} "
            "intensities. Please give one intensity for each x value."
        )
    return x, y


def _format_x(value):
    # Round off float noise, so np.arange(16, 110, 0.025) ends at 109.975
    formatted_x = str(round(float(value), 6))
    return formatted_x


# End of file
