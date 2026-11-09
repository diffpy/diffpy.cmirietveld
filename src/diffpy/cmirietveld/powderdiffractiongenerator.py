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
"""Powder diffraction profile generator.

The PowderDiffractionGenerator class calculates the Bragg intensity of
one phase of a powder diffraction pattern with pyobjcryst. It mirrors
srfit's PDFGenerator, so a structure ParameterSet can be shared between
PowderDiffractionGenerators for several datasets and a PDFGenerator.
"""

__all__ = ["PowderDiffractionGenerator"]

import warnings
from pathlib import Path

import numpy as np
from pyobjcryst import ObjCrystException
from pyobjcryst.crystal import Crystal, create_crystal_from_cif
from pyobjcryst.powderpattern import PowderPattern

from diffpy.cmirietveld.instruments import Instrument
from diffpy.cmirietveld.peak_profiles import PeakProfile
from diffpy.cmistructure import structure_to_parameter_set
from diffpy.cmistructure.basestructureparset import BaseStructureParSet
from diffpy.srfit.exceptions import SrFitError
from diffpy.srfit.fitbase import ProfileGenerator

PLACEHOLDER_TTH = np.linspace(5.0, 175.0, 1000)


class PowderDiffractionGenerator(ProfileGenerator):
    """Calculate the Bragg intensity of one phase of a powder pattern
    with pyobjcryst.

    The generator is assembled from components, one per stage of the
    calculation: an intensity source, an instrument, zero or more
    intensity corrections and a peak profile. The Parameters of each
    component sit under its name, for example
    ``generator.peak_profile.U`` and ``generator.instrument.zero``. The
    generator has no defaults: the structure, the instrument and the peak
    profile are each set before a pattern is calculated, so a script
    always shows which models it refines.

    The generator calculates the Bragg intensity only; a background is
    added in the FitContribution equation.

    Parameters
    ----------
    name : str, optional
        The name of the generator (default "bragg", the name used in
        FitContribution equations such as "bragg + background"). It must
        be a valid attribute name.

    Attributes
    ----------
    phase : ParameterSet or None
        The structure ParameterSet, or None until a structure is set.
    peak_profile : PeakProfile or None
        The peak profile component, or None until one is set.
    instrument : Instrument or None
        The instrument component, or None until one is set.

    Managed Parameters
    ------------------
    scale
        The scale factor, applied to the calculated pattern.
    """

    def __init__(self, name="bragg"):
        ProfileGenerator.__init__(self, name)
        self.phase = None
        self._pattern = None
        self._diffraction = None
        self._last_tth = np.empty(0)
        self._installed_settings = {}
        self._new_parameter("scale", 1.0)
        self.instrument = None
        self.peak_profile = None

    def set_structure(self, structure, name=None):
        """Set the structure used to calculate the pattern.

        The structure is a pyobjcryst Crystal, the path to a CIF file, or a
        ParameterSet that adapts a Crystal. A Crystal or a path is adapted
        to a new ParameterSet, and a path is first read into a new Crystal.
        A ParameterSet is used as it is, with its own name.

        Giving the same ParameterSet to several generators shares the
        structure between them. Giving them the same path does not: each
        call reads the file again, so each generator refines its own
        structure.

        Parameters
        ----------
        structure : Crystal, pathlib.Path, str or ObjCrystCrystalParSet
            The pyobjcryst Crystal, the path to a CIF file holding it, or
            the diffpy.cmistructure ParameterSet that adapts it.
        name : str, optional
            The name of the ParameterSet made for a Crystal or a path
            (default "phase"). With a ParameterSet, it must match the
            ParameterSet's own name.

        Raises
        ------
        FileNotFoundError
            If ``structure`` is a path to a file that does not exist.
        ValueError
            If ``structure`` is a path to a file that isn't a CIF file
            holding a crystal structure, a structure or ParameterSet that
            ObjCryst can't calculate with, or a ParameterSet given with a
            ``name`` other than its own.

        Warns
        -----
        UserWarning
            If a Parameter of the replaced structure is a variable of a fit
            recipe or constrained in one, since the recipe would refine a
            Parameter that no longer changes the pattern.
        """
        if isinstance(structure, BaseStructureParSet):
            if name is not None and name != structure.name:
                raise ValueError(
                    f"set_structure was given the name '{name}' with a "
                    "ParameterSet, which keeps its own name, "
                    f"'{structure.name}'. Please leave out the name, or give "
                    "it when you create the ParameterSet."
                )
            crystal = structure.structure
        elif isinstance(structure, (str, Path)):
            crystal = _read_crystal_from_cif(Path(structure))
        else:
            crystal = structure
        if not isinstance(crystal, Crystal):
            raise ValueError(
                f"set_structure was given a {type(structure).__name__}, "
                "which the powder diffraction generator can't calculate "
                "with. Please give a pyobjcryst Crystal, the path to a CIF "
                "file, or a ParameterSet made from a Crystal."
            )
        if isinstance(structure, BaseStructureParSet):
            phase = structure
        else:
            phase = structure_to_parameter_set(name or "phase", crystal)
        self._replace_phase(phase, "set_structure")

    def set_structure_from_parset(self, phase):
        """Set the structure from an existing ParameterSet.

        This is the same as ``set_structure`` with a ParameterSet. It is
        kept so the generator has the same methods as
        diffpy.cmipdf's PDF generators.

        Parameters
        ----------
        phase : ObjCrystCrystalParSet
            The diffpy.cmistructure ParameterSet that adapts a pyobjcryst
            Crystal.

        Raises
        ------
        ValueError
            If ``phase`` does not adapt a pyobjcryst Crystal.

        Warns
        -----
        UserWarning
            If a Parameter of the replaced structure is a variable of a fit
            recipe or constrained in one.
        """
        if not isinstance(phase, BaseStructureParSet) or not isinstance(
            phase.structure, Crystal
        ):
            raise ValueError(
                f"set_structure_from_parset was given a "
                f"{type(phase).__name__}, which the powder diffraction "
                "generator can't calculate with. Please give a ParameterSet "
                "made from a pyobjcryst Crystal."
            )
        self._replace_phase(phase, "set_structure_from_parset")

    def set_peak_profile(self, peak_profile):
        """Set the peak profile component.

        The peak profile replaces the one set before, with its Parameters
        under ``peak_profile``. It is installed once the generator has a
        structure, so it can be set before or after the structure.

        Parameters
        ----------
        peak_profile : PeakProfile
            The peak profile. It can be shared with other generators.

        Raises
        ------
        ValueError
            If ``peak_profile`` is not a PeakProfile.

        Warns
        -----
        UserWarning
            If a Parameter of the replaced peak profile is a variable of a
            fit recipe or constrained in one, since the recipe would refine
            a Parameter that no longer changes the pattern.
        """
        self._set_component(
            "peak_profile",
            peak_profile,
            PeakProfile,
            "PseudoVoigtProfile or TCHProfile",
        )

    def set_instrument(self, instrument):
        """Set the instrument component.

        The instrument replaces the one set before, with its Parameters
        under ``instrument``. It is installed once the generator has a
        structure, so it can be set before or after the structure.

        Parameters
        ----------
        instrument : Instrument
            The instrument. All the generators of one dataset share it.

        Raises
        ------
        ValueError
            If ``instrument`` is not an Instrument.

        Warns
        -----
        UserWarning
            If a Parameter of the replaced instrument is a variable of a
            fit recipe or constrained in one, since the recipe would refine
            a Parameter that no longer changes the pattern.
        """
        self._set_component(
            "instrument",
            instrument,
            Instrument,
            "ConstantWavelengthInstrument",
        )

    def add_intensity_correction(self, correction):
        """Add an intensity correction component.

        Parameters
        ----------
        correction : IntensityCorrection
            The correction. Its Parameters sit under its name.

        Raises
        ------
        ValueError
            If ``correction`` is not an IntensityCorrection.
        """
        raise NotImplementedError

    def set_intensity_source(self, intensity_source):
        """Set the intensity source component.

        Parameters
        ----------
        intensity_source : IntensitySource
            The intensity source, which replaces the structure.

        Raises
        ------
        ValueError
            If ``intensity_source`` is not an IntensitySource.
        """
        raise NotImplementedError

    def print_info(self):
        """Print what the generator holds.

        The scale is printed first, then the structure, the instrument
        and the peak profile, each under the name its Parameters sit
        under, so a user can check a setup before refining it. The
        structure shows its space group and lattice parameters, with the
        angles in degrees, rather than all its Parameters. Each
        component shows its type, then its settings from
        ``get_settings`` on one line if it has any, then the current
        value of each of its Parameters. Anything that isn't set yet
        shows "not set".
        """
        lines = [
            f"PowderDiffractionGenerator '{self.name}'",
            f"  scale = {_format_value(self.scale.value)}",
        ]
        if self.phase is None:
            lines.append("  phase: not set")
        else:
            crystal = self.phase.structure
            lines.append(
                f"  {self.phase.name}: Crystal '{crystal.GetName()}', "
                f"space group {crystal.GetSpaceGroup().GetName()}"
            )
            for length_name in ["a", "b", "c"]:
                length = self.phase.get(length_name).value
                lines.append(f"    {length_name} = {_format_value(length)}")
            for angle_name in ["alpha", "beta", "gamma"]:
                angle = np.degrees(self.phase.get(angle_name).value)
                lines.append(f"    {angle_name} = {_format_value(angle)}")
        components = {
            "instrument": self.instrument,
            "peak_profile": self.peak_profile,
        }
        for component_name, component in components.items():
            if component is None:
                lines.append(f"  {component_name}: not set")
                continue
            lines.append(f"  {component_name}: {type(component).__name__}")
            settings = []
            for setting_name, value in component.get_settings().items():
                settings.append(f"{setting_name} = {_format_value(value)}")
            if settings:
                lines.append(f"    settings: {', '.join(settings)}")
            for parameter in component.iterate_over_parameters():
                lines.append(
                    f"    {parameter.name} = {_format_value(parameter.value)}"
                )
        print("\n".join(lines))

    def __call__(self, tth):
        """Calculate the Bragg intensity.

        Each component whose settings changed since it was last installed
        is installed again. Then each component pushes its Parameter
        values into pyobjcryst, and the pattern is calculated and
        multiplied by ``scale``.

        Parameters
        ----------
        tth : ndarray
            The 2theta values in degrees at which to calculate the
            pattern.

        Returns
        -------
        ndarray
            The calculated Bragg intensity at each 2theta value.

        Raises
        ------
        SrFitError
            If no structure, instrument or peak profile is set, or the
            instrument has no wavelength.
        ValueError
            If a component's Parameter value is outside the limits of its
            ObjCryst ``RefinablePar``.
        """
        self._check_setup()
        components = [self.instrument, self.peak_profile]
        for component in components:
            component._validate()
        if not np.array_equal(tth, self._last_tth):
            self._set_tth(tth)
        for component in components:
            installed_component, installed_version = (
                self._installed_settings.get(component.name, (None, None))
            )
            if (
                installed_component is not component
                or installed_version != component.settings_version
            ):
                component.install(self._pattern, self._diffraction)
                self._installed_settings[component.name] = (
                    component,
                    component.settings_version,
                )
        for component in components:
            component.update(self._pattern, self._diffraction)
        self._pattern.Prepare()
        calculated_pattern = np.array(self._pattern.GetPowderPatternCalc())
        bragg_intensity = self.scale.value * calculated_pattern
        return bragg_intensity

    def _validate(self):
        self._check_setup()
        ProfileGenerator._validate(self)

    def _check_setup(self):
        if self.phase is None:
            raise SrFitError(
                "The generator has no structure. Please set one with "
                "set_structure or set_structure_from_parset."
            )
        if self.instrument is None:
            raise SrFitError(
                "The generator has no instrument. Please set one with "
                "set_instrument, such as a ConstantWavelengthInstrument."
            )
        if self.peak_profile is None:
            raise SrFitError(
                "The generator has no peak profile. Please set one with "
                "set_peak_profile, such as a PseudoVoigtProfile or a "
                "TCHProfile."
            )

    def _replace_phase(self, phase, setter_name):
        if self.phase is not None:
            _warn_if_parameters_used(self.phase, setter_name, "structure")
            self.remove_parameter_set(self.phase)
        self.phase = phase
        self.add_parameter_set(phase)
        self._pattern = PowderPattern()
        self._set_tth(PLACEHOLDER_TTH)
        self._diffraction = self._pattern.AddPowderPatternDiffraction(
            phase.structure
        )
        self._installed_settings = {}
        self._flush(other=(self,))

    def _set_component(self, component_name, component, base_class, examples):
        setter_name = f"set_{component_name}"
        kind = component_name.replace("_", " ")
        if not isinstance(component, base_class):
            given_name = type(component).__name__
            raise ValueError(
                f"{setter_name} was given {_get_article(given_name)} "
                f"{given_name}, which is not {_get_article(kind)} {kind}. "
                f"Please give {_get_article(base_class.__name__)} "
                f"{base_class.__name__}, such as {examples}."
            )
        old_component = getattr(self, component_name)
        if old_component is not None:
            _warn_if_parameters_used(old_component, setter_name, kind)
            self.remove_parameter_set(old_component)
        setattr(self, component_name, component)
        self.add_parameter_set(component)
        self._flush(other=(self,))

    def _set_tth(self, tth):
        self._last_tth = np.array(tth, dtype=float)
        self._pattern.SetPowderPatternX(np.radians(self._last_tth))
        self._pattern.SetPowderPatternObs(np.zeros_like(self._last_tth))


def _read_crystal_from_cif(cif_path):
    # create_crystal_from_cif takes no Path, and raises ObjCrystException
    # without the file name for a file with no structure
    try:
        crystal = create_crystal_from_cif(str(cif_path))
    except ObjCrystException:
        raise ValueError(
            f"No crystal structure could be read from {cif_path}. "
            "Please give a CIF file that holds a crystal structure."
        ) from None
    return crystal


def _warn_if_parameters_used(parameter_set, setter_name, kind):
    # A Parameter is observed by its ParameterSet alone, unless a recipe
    # variable proxies it, which adds the recipe's observer
    used_names = []
    for parameter in parameter_set.iterate_over_parameters():
        if len(parameter._observers) > 1 or parameter.constrained:
            if parameter.name not in used_names:
                used_names.append(parameter.name)
    if not used_names:
        return
    warnings.warn(
        f"{setter_name} replaced {_get_article(kind)} {kind} whose "
        f"Parameters are used in a fit recipe: {', '.join(used_names)}. "
        "The recipe would still refine them, but they no longer change the "
        f"pattern. Please call {setter_name} before adding the {kind}'s "
        "Parameters to the recipe.",
        UserWarning,
        stacklevel=4,
    )


def _format_value(value):
    if value is None:
        formatted_value = "not set"
    elif isinstance(value, str):
        formatted_value = value
    else:
        formatted_value = f"{value:g}"
    return formatted_value


def _get_article(word):
    article = "an" if word[0].lower() in "aeiou" else "a"
    return article


# End of file
