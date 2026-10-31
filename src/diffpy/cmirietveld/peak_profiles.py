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
"""Peak profile components, which set the shape of each Bragg peak."""

__all__ = [
    "PeakProfile",
    "PseudoVoigtProfile",
    "TCHProfile",
    "BackToBackExponentialProfile",
]

from pyobjcryst.reflectionprofile import (
    ReflectionProfilePseudoVoigt,
    ReflectionProfilePseudoVoigtTCH,
)

from diffpy.cmirietveld.component import Component


class PeakProfile(Component):
    """Base class of the components that set the peak shape.

    A generator holds its peak profile under the name ``peak_profile``,
    so the Parameters of a profile are reached as
    ``generator.peak_profile.<name>``.
    """

    def __init__(self):
        Component.__init__(self, "peak_profile")


class PseudoVoigtProfile(PeakProfile):
    """Pseudo-Voigt peak profile with Caglioti widths and asymmetry.

    This is the default peak profile of a generator. It wraps pyobjcryst's
    ``ReflectionProfilePseudoVoigt``.

    Managed Parameters
    ------------------
    U, V, W
        The Caglioti peak width terms, in degrees squared.
    eta0, eta1
        The pseudo-Voigt mixing terms.
    asym0, asym1, asym2
        The peak asymmetry terms.
    """

    def __init__(self):
        PeakProfile.__init__(self)
        self.create_parameters_from(
            ReflectionProfilePseudoVoigt(),
            {
                "U": "U",
                "V": "V",
                "W": "W",
                "eta0": "Eta0",
                "eta1": "Eta1",
                "asym0": "Asym0",
                "asym1": "Asym1",
                "asym2": "Asym2",
            },
        )

    def install(self, pattern, diffraction):
        """Install a fresh pyobjcryst pseudo-Voigt profile in one
        generator.

        The diffraction component gets a new
        ``ReflectionProfilePseudoVoigt``, replacing the profile installed
        before. Its values are set by ``update``.

        Parameters
        ----------
        pattern : pyobjcryst.powderpattern.PowderPattern
            The pattern of the generator.
        diffraction : pyobjcryst.powderpattern.PowderPatternDiffraction
            The diffraction component of the generator's structure.
        """
        diffraction.SetProfile(ReflectionProfilePseudoVoigt())

    def update(self, pattern, diffraction):
        """Push the Parameter values into the installed pseudo-Voigt
        profile.

        ``SetProfile`` installs a copy, so the values are written into
        ``diffraction.GetProfile()``, not into the profile given to
        ``SetProfile``.

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
        self.push_parameters(diffraction.GetProfile())


class TCHProfile(PeakProfile):
    """Symmetric Thompson-Cox-Hastings pseudo-Voigt peak profile.

    It wraps pyobjcryst's ``ReflectionProfilePseudoVoigtTCH``, which has
    no asymmetry terms.

    Managed Parameters
    ------------------
    U, V, W
        The Gaussian width terms, in degrees squared.
    X, Y, Z
        The Lorentzian width terms, in degrees.
    P
        The Gaussian size broadening term.
    LGmix
        The mixing of the Gaussian and Lorentzian size broadening.
    """

    def __init__(self):
        PeakProfile.__init__(self)
        self.create_parameters_from(
            ReflectionProfilePseudoVoigtTCH(),
            {
                "U": "U",
                "V": "V",
                "W": "W",
                "X": "X",
                "Y": "Y",
                "Z": "Z",
                "P": "P",
                "LGmix": "LGmix",
            },
        )

    def install(self, pattern, diffraction):
        """Install a fresh pyobjcryst TCH profile in one generator.

        The diffraction component gets a new
        ``ReflectionProfilePseudoVoigtTCH``, replacing the profile installed
        before. Its values are set by ``update``.

        Parameters
        ----------
        pattern : pyobjcryst.powderpattern.PowderPattern
            The pattern of the generator.
        diffraction : pyobjcryst.powderpattern.PowderPatternDiffraction
            The diffraction component of the generator's structure.
        """
        diffraction.SetProfile(ReflectionProfilePseudoVoigtTCH())

    def update(self, pattern, diffraction):
        """Push the Parameter values into the installed TCH profile.

        ``SetProfile`` installs a copy, so the values are written into
        ``diffraction.GetProfile()``, not into the profile given to
        ``SetProfile``.

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
        self.push_parameters(diffraction.GetProfile())


class BackToBackExponentialProfile(PeakProfile):
    """Peak profile of a time-of-flight pattern.

    It is a pseudo-Voigt convolved with back-to-back exponentials, and
    wraps ObjCryst's ``ReflectionProfileDoubleExponentialPseudoVoigt``.
    """


# End of file
