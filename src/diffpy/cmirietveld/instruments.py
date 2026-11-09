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
"""Instrument components, which map d-spacings onto the data axis.

An instrument describes one measured dataset, so all the generators of a
dataset, one per phase, are given the same instrument object.
"""

__all__ = [
    "Instrument",
    "ConstantWavelengthInstrument",
    "TimeOfFlightInstrument",
]

import warnings

from pyobjcryst.powderpattern import PowderPattern
from pyobjcryst.radiation import RadiationType, WavelengthType

from diffpy.cmirietveld.component import Component
from diffpy.srfit.exceptions import SrFitError

RADIATION_TYPES = {
    "X": RadiationType.RAD_XRAY,
    "N": RadiationType.RAD_NEUTRON,
    "E": RadiationType.RAD_ELECTRON,
}


class Instrument(Component):
    """Base class of the components that describe the instrument.

    A generator holds its instrument under the name ``instrument``, so
    the Parameters of an instrument are reached as
    ``generator.instrument.<name>``.
    """

    def __init__(self):
        Component.__init__(self, "instrument")


class ConstantWavelengthInstrument(Instrument):
    """Instrument of a constant-wavelength x-ray or neutron
    diffractometer.

    The wavelength has no default, because no dataset uses ObjCryst's
    default of 1 angstrom. Set it with ``set_wavelength``,
    ``set_xray_source`` or ``set_xray_wavelengths`` before calculating a
    pattern. The radiation type, the Kalpha1/Kalpha2 doublet and the
    polarization rate are settings, applied by ``install``. A doublet is
    x-ray tube radiation, so it can only be set with the radiation type
    "X".

    The wavelength setters warn when they discard something the user may
    still expect: a wavelength refined or set by hand since the last
    setter call, or a Kalpha1/Kalpha2 doublet set by another setter.

    Managed Parameters
    ------------------
    zero
        The 2theta zero shift, in degrees.
    displacement
        The sample displacement correction, in degrees.
    transparency
        The sample transparency correction, in degrees.
    mu_r
        The cylinder absorption of the sample, the product of the linear
        absorption coefficient and the radius.
    wavelength
        The wavelength in angstroms, the intensity-weighted mean for a
        Kalpha1/Kalpha2 doublet. It is refinable, for example to
        calibrate an instrument against a standard.
    """

    def __init__(self):
        Instrument.__init__(self)
        pattern = PowderPattern()
        radiation = pattern.GetRadiation()
        self.create_parameters_from(
            pattern,
            {
                "zero": "Zero",
                "displacement": "2ThetaDispl",
                "transparency": "2ThetaTransp",
                "mu_r": "MuR",
            },
            target="pattern",
        )
        self.create_parameters_from(
            radiation, {"wavelength": "Wavelength"}, target="radiation"
        )
        self.wavelength.set_value(None)
        wavelength_par = radiation.GetPar("Wavelength")
        self._wavelength_limits = (
            wavelength_par.GetHumanMin(),
            wavelength_par.GetHumanMax(),
        )
        self._radiation_type = "X"
        self._xray_source = None
        self._kalpha_wavelengths = None
        self._kalpha2_ratio = 0.5
        self._polarization_rate = 0.0
        self._last_set_wavelength = None

    def set_radiation_type(self, radiation_type):
        """Set the radiation type.

        Parameters
        ----------
        radiation_type : str
            The radiation type: "X" for x-rays, "N" for neutrons or "E" for
            electrons.

        Raises
        ------
        ValueError
            If the radiation type is not one of "X", "N" or "E", or it is
            "N" or "E" while a Kalpha1/Kalpha2 doublet is set.

        Warns
        -----
        UserWarning
            If it is "N" or "E" while a polarization rate other than 0
            is set, since ObjCryst only applies the rate to x-rays.
        """
        if radiation_type not in RADIATION_TYPES:
            raise ValueError(
                f"'{radiation_type}' is not a supported radiation type. "
                "Please use 'X' for x-rays, 'N' for neutrons or 'E' for "
                "electrons."
            )
        has_doublet = (
            self._xray_source is not None
            or self._kalpha_wavelengths is not None
        )
        if radiation_type != "X" and has_doublet:
            raise ValueError(
                f"The radiation type '{radiation_type}' can't be used with "
                "the Kalpha1/Kalpha2 doublet that is set, which needs x-rays. "
                "Please set a single wavelength with set_wavelength first."
            )
        self._warn_if_polarization_unused(
            self._polarization_rate, radiation_type
        )
        self._radiation_type = radiation_type
        self.mark_settings_changed()

    def set_wavelength(self, wavelength):
        """Set a single wavelength.

        This sets the ``wavelength`` Parameter and replaces a
        Kalpha1/Kalpha2 doublet set before. ObjCryst limits the wavelength
        to 0.05 to 20 angstroms, which leaves out the electrons of beams
        faster than about 50 kV.

        Parameters
        ----------
        wavelength : float
            The wavelength in angstroms.

        Raises
        ------
        ValueError
            If the wavelength is outside ObjCryst's limits of 0.05 to 20
            angstroms.

        Warns
        -----
        UserWarning
            If it replaces a wavelength refined or set by hand since the
            last setter call, or a Kalpha1/Kalpha2 doublet.
        """
        self._check_wavelength_limits(wavelength)
        if self._xray_source is not None:
            warnings.warn(
                "set_wavelength replaced the "
                f"{self._xray_source} Kalpha1/Kalpha2 doublet with a "
                "single wavelength. To use the doublet again, call "
                "set_xray_source.",
                UserWarning,
                stacklevel=2,
            )
        elif self._kalpha_wavelengths is not None:
            warnings.warn(
                "set_wavelength replaced the explicit Kalpha1/Kalpha2 "
                "doublet with a single wavelength. To use the doublet "
                "again, call set_xray_wavelengths.",
                UserWarning,
                stacklevel=2,
            )
        self._warn_if_wavelength_replaced("set_wavelength", wavelength)
        self._xray_source = None
        self._kalpha_wavelengths = None
        self._set_wavelength_value(wavelength)
        self.mark_settings_changed()

    def set_xray_source(self, element, kalpha2_ratio=None):
        """Set the wavelengths of an x-ray tube's Kalpha1/Kalpha2
        doublet.

        The wavelengths are ObjCryst's tabulated ones for the anode
        element. The ``wavelength`` Parameter is set to their
        intensity-weighted mean.

        Parameters
        ----------
        element : str
            The anode element: "Ag", "Co", "Cr", "Cu", "Fe" or "Mo", the
            ones ObjCryst has wavelengths for.
        kalpha2_ratio : float, optional
            The intensity ratio of Kalpha2 to Kalpha1. If it isn't given,
            the ratio set before is kept, which is 0.5 until one is given.

        Raises
        ------
        ValueError
            If ObjCryst has no tabulated wavelengths for the element, or
            the radiation type is not "X".

        Warns
        -----
        UserWarning
            If it replaces a wavelength refined or set by hand since the
            last setter call, or a doublet set with
            ``set_xray_wavelengths``.
        """
        self._check_xray_radiation()
        if kalpha2_ratio is None:
            kalpha2_ratio = self._kalpha2_ratio
        # ObjCryst only prints a warning for an unknown element and keeps
        # its wavelength, so an unchanged wavelength means it was unknown
        tube_pattern = PowderPattern()
        tube_radiation = tube_pattern.GetRadiation()
        default_wavelength = tube_radiation.GetWavelength()
        tube_pattern.SetWavelength(element, kalpha2_ratio)
        mean_wavelength = tube_radiation.GetWavelength()
        if mean_wavelength == default_wavelength:
            raise ValueError(
                f"ObjCryst has no x-ray tube wavelengths for '{element}'. "
                "Please use one of the anode elements it has, 'Ag', 'Co', "
                "'Cr', 'Cu', 'Fe' or 'Mo', or set the wavelengths with "
                "set_xray_wavelengths."
            )
        if self._kalpha_wavelengths is not None:
            warnings.warn(
                "set_xray_source replaced the explicit Kalpha1/Kalpha2 "
                f"doublet with the {element} doublet. To use the explicit "
                "wavelengths again, call set_xray_wavelengths.",
                UserWarning,
                stacklevel=2,
            )
        self._warn_if_wavelength_replaced("set_xray_source", mean_wavelength)
        self._xray_source = element
        self._kalpha_wavelengths = None
        self._kalpha2_ratio = kalpha2_ratio
        self._set_wavelength_value(mean_wavelength)
        self.mark_settings_changed()

    def set_xray_wavelengths(self, kalpha1, kalpha2, kalpha2_ratio=None):
        """Set the wavelengths of a Kalpha1/Kalpha2 doublet explicitly.

        This replaces ObjCryst's tabulated tube wavelengths, for example
        with the calibrated ones of an instrument parameter file. The
        ``wavelength`` Parameter is set to their intensity-weighted mean.

        Parameters
        ----------
        kalpha1 : float
            The Kalpha1 wavelength in angstroms.
        kalpha2 : float
            The Kalpha2 wavelength in angstroms.
        kalpha2_ratio : float, optional
            The intensity ratio of Kalpha2 to Kalpha1. If it isn't given,
            the ratio set before is kept, which is 0.5 until one is given.

        Raises
        ------
        ValueError
            If a wavelength is outside ObjCryst's limits of 0.05 to 20
            angstroms, Kalpha2 is not longer than Kalpha1, or the
            radiation type is not "X".

        Warns
        -----
        UserWarning
            If it replaces a wavelength refined or set by hand since the
            last setter call, or a doublet set with ``set_xray_source``.
        """
        self._check_xray_radiation()
        self._check_wavelength_limits(kalpha1)
        self._check_wavelength_limits(kalpha2)
        if kalpha2 <= kalpha1:
            raise ValueError(
                f"Kalpha2, {kalpha2} angstroms, is not longer than Kalpha1, "
                f"{kalpha1} angstroms. Please give the wavelengths in the "
                "order kalpha1, kalpha2."
            )
        if kalpha2_ratio is None:
            kalpha2_ratio = self._kalpha2_ratio
        mean_wavelength = kalpha1 + (kalpha2 - kalpha1) * kalpha2_ratio / (
            1 + kalpha2_ratio
        )
        if self._xray_source is not None:
            warnings.warn(
                "set_xray_wavelengths replaced the "
                f"{self._xray_source} Kalpha1/Kalpha2 doublet with the "
                f"explicit wavelengths. To use the {self._xray_source} "
                "doublet again, call set_xray_source.",
                UserWarning,
                stacklevel=2,
            )
        self._warn_if_wavelength_replaced(
            "set_xray_wavelengths", mean_wavelength
        )
        self._xray_source = None
        self._kalpha_wavelengths = (kalpha1, kalpha2)
        self._kalpha2_ratio = kalpha2_ratio
        self._set_wavelength_value(mean_wavelength)
        self.mark_settings_changed()

    def set_polarization(self, polarization_rate):
        """Set the polarization rate of the x-ray beam.

        ObjCryst only applies the rate to x-rays, so it has no effect
        with neutrons or electrons.

        Parameters
        ----------
        polarization_rate : float
            The polarization rate f used by ObjCryst, from -1 to 1. An
            unpolarized beam has f = 0. For a GSAS-style polarization P,
            f = 2P - 1.

        Raises
        ------
        ValueError
            If the rate is outside [-1, 1].

        Warns
        -----
        UserWarning
            If the rate is not 0 and the radiation type is "N" or "E".
        """
        if not -1 <= polarization_rate <= 1:
            raise ValueError(
                f"The polarization rate {polarization_rate} is outside "
                "[-1, 1]. Please give ObjCryst's rate f; for a GSAS "
                "polarization P, f = 2P - 1."
            )
        self._warn_if_polarization_unused(
            polarization_rate, self._radiation_type
        )
        self._polarization_rate = polarization_rate
        self.mark_settings_changed()

    def get_settings(self):
        """Return the settings in use, for showing to a user.

        The radiation type and the polarization rate are always listed.
        A tube adds ``xray_source`` and ``kalpha2_ratio``, and explicit
        wavelengths add ``kalpha1``, ``kalpha2`` and ``kalpha2_ratio``.
        A single wavelength adds nothing, since it is the ``wavelength``
        Parameter.

        Returns
        -------
        dict of str to object
            The value of each setting in use, by the name of the setter
            argument that sets it.
        """
        settings = {"radiation_type": self._radiation_type}
        if self._xray_source is not None:
            settings["xray_source"] = self._xray_source
            settings["kalpha2_ratio"] = self._kalpha2_ratio
        elif self._kalpha_wavelengths is not None:
            kalpha1, kalpha2 = self._kalpha_wavelengths
            settings["kalpha1"] = kalpha1
            settings["kalpha2"] = kalpha2
            settings["kalpha2_ratio"] = self._kalpha2_ratio
        settings["polarization_rate"] = self._polarization_rate
        return settings

    def install(self, pattern, diffraction):
        """Set the radiation, wavelengths and polarization of one
        generator.

        The radiation type is set first, then the wavelength or the
        Kalpha1/Kalpha2 doublet, then the polarization rate, and the
        pattern is prepared last. ObjCryst resets the rate when a tube is
        set, and only recalculates for a new rate after the wavelength is
        set again, so this order is what makes a change to any setting
        take effect. The ``wavelength`` value itself is set by ``update``.

        Parameters
        ----------
        pattern : pyobjcryst.powderpattern.PowderPattern
            The pattern of the generator.
        diffraction : pyobjcryst.powderpattern.PowderPatternDiffraction
            The diffraction component of the generator's structure.
        """
        radiation = pattern.GetRadiation()
        pattern.SetRadiationType(RADIATION_TYPES[self._radiation_type])
        if self._xray_source is not None:
            pattern.SetWavelength(self._xray_source, self._kalpha2_ratio)
        elif self._kalpha_wavelengths is not None:
            # Only SetWavelength(element, ratio) sets a ratio other than
            # 0.5, so a tube sets up the doublet, and its Kalpha2 - Kalpha1
            # is replaced
            kalpha1, kalpha2 = self._kalpha_wavelengths
            pattern.SetWavelength("Cu", self._kalpha2_ratio)
            delta_lambda_par = radiation.GetPar("XRayTubeDeltaLambda")
            delta_lambda_par.SetHumanValue(kalpha2 - kalpha1)
        else:
            pattern.SetWavelength(self.wavelength.value)
            # SetWavelength(wavelength) keeps a doublet set before
            radiation.SetWavelengthType(
                WavelengthType.WAVELENGTH_MONOCHROMATIC
            )
        radiation.SetLinearPolarRate(self._polarization_rate)
        pattern.Prepare()

    def update(self, pattern, diffraction):
        """Push the Parameter values into the pattern and its radiation.

        Parameters
        ----------
        pattern : pyobjcryst.powderpattern.PowderPattern
            The pattern of the generator.
        diffraction : pyobjcryst.powderpattern.PowderPatternDiffraction
            The diffraction component of the generator's structure.

        Raises
        ------
        ValueError
            If a Parameter value is outside the limits of its
            ``RefinablePar``.
        """
        self.push_parameters(pattern, target="pattern")
        self.push_parameters(pattern.GetRadiation(), target="radiation")

    def _validate(self):
        if self.wavelength.value is None:
            raise SrFitError(
                "The instrument has no wavelength. Please set one with "
                "set_wavelength, set_xray_source or set_xray_wavelengths."
            )
        Instrument._validate(self)

    def _set_wavelength_value(self, wavelength):
        self.wavelength.set_value(wavelength)
        self._last_set_wavelength = wavelength

    def _warn_if_wavelength_replaced(self, setter_name, new_wavelength):
        current_wavelength = self.wavelength.value
        if current_wavelength is None:
            return
        if current_wavelength == self._last_set_wavelength:
            return
        warnings.warn(
            f"{setter_name} replaced the wavelength "
            f"{current_wavelength:g} angstroms, which was refined or set "
            f"by hand, with {new_wavelength:g} angstroms. To keep "
            f"{current_wavelength:g} angstroms, set instrument.wavelength "
            f"again after {setter_name}.",
            UserWarning,
            stacklevel=3,
        )

    def _warn_if_polarization_unused(self, polarization_rate, radiation_type):
        if polarization_rate == 0 or radiation_type == "X":
            return
        warnings.warn(
            f"The polarization rate {polarization_rate} has no effect with "
            f"the radiation type '{radiation_type}', because ObjCryst only "
            "applies it to x-rays. Please set a polarization rate only for "
            "x-ray data.",
            UserWarning,
            stacklevel=3,
        )

    def _check_xray_radiation(self):
        if self._radiation_type != "X":
            raise ValueError(
                "A Kalpha1/Kalpha2 doublet needs x-rays, but the radiation "
                f"type is '{self._radiation_type}'. Please set the radiation "
                "type to 'X' first, or set a single wavelength with "
                "set_wavelength."
            )

    def _check_wavelength_limits(self, wavelength):
        lower_limit, upper_limit = self._wavelength_limits
        if not lower_limit <= wavelength <= upper_limit:
            raise ValueError(
                f"The wavelength {wavelength} angstroms is outside "
                f"ObjCryst's limits of {lower_limit:g} to {upper_limit:g} "
                "angstroms. Please give a wavelength within these limits."
            )


class TimeOfFlightInstrument(Instrument):
    """Instrument of a time-of-flight neutron diffractometer.

    Peak positions follow t = DIFC * d + DIFA * d**2 + zero.

    Managed Parameters
    ------------------
    DIFC
        The linear conversion from d-spacing to time of flight.
    DIFA
        The quadratic conversion from d-spacing to time of flight.
    zero
        The time-of-flight zero shift.
    """


# End of file
