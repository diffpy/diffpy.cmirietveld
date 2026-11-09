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
"""Base class of the components that configure a
PowderDiffractionGenerator.

A component describes one stage of the powder pattern calculation, such
as the peak profile or the instrument. It holds its refinable values as
srfit Parameters and configures the pyobjcryst objects of a generator.
"""

__all__ = ["Component"]

from diffpy.srfit.fitbase.parameter import Parameter
from diffpy.srfit.fitbase.parameterset import ParameterSet


class Component(ParameterSet):
    """Configure one stage of a PowderDiffractionGenerator.

    Subclasses write ``install`` and ``update`` for the pyobjcryst model
    they configure. They do not keep a reference to the pyobjcryst objects
    they are given, so one component can serve several generators.

    A component has two kinds of configuration:

    - **Parameters** are the values a fit can refine, such as a zero shift
      or a peak width. They are created from a pyobjcryst model with
      ``create_parameters_from``, and ``update`` writes them into the
      model with ``push_parameters`` before every calculation. A component
      that configures more than one pyobjcryst object, such as a pattern
      and its radiation, creates the Parameters of each object under its
      own target and pushes them one target at a time.
    - **Settings** are the configuration that is not refined, such as the
      radiation type or the x-ray tube. ``install`` applies them. A method
      that changes a setting calls ``mark_settings_changed``, so every
      generator holding the component runs ``install`` again before its
      next calculation.

    Parameters
    ----------
    name : str
        The name of the component. It must be a valid attribute name.

    Examples
    --------
    An instrument whose ``zero`` belongs to the pattern and whose
    ``wavelength`` belongs to the pattern's radiation:

    >>> class MyInstrument(Component):
    ...     def __init__(self):
    ...         super().__init__("instrument")
    ...         self.create_parameters_from(
    ...             PowderPattern(), {"zero": "Zero"}, target="pattern"
    ...         )
    ...         self.create_parameters_from(
    ...             Radiation(),
    ...             {"wavelength": "Wavelength"},
    ...             target="radiation",
    ...         )
    ...
    ...     def update(self, pattern, diffraction):
    ...         self.push_parameters(pattern, target="pattern")
    ...         self.push_parameters(
    ...             pattern.GetRadiation(), target="radiation"
    ...         )
    """

    def __init__(self, name):
        ParameterSet.__init__(self, name)
        self._refinable_par_names = {}
        self._settings_version = 0

    @property
    def settings_version(self):
        """The number of times the settings of the component changed.

        A generator compares it with the value it last installed, and
        runs ``install`` again when it differs. It starts at 0, and only
        ``mark_settings_changed`` changes it: a change to a Parameter
        value is not a settings change.
        """
        return self._settings_version

    def mark_settings_changed(self):
        """Record that a setting of the component changed.

        Settings are the configuration of a component that is not a
        Parameter, such as the radiation type of an instrument. Call
        this from every method that changes a setting. It bumps
        ``settings_version`` and notifies the component's observers, so
        every generator holding the component drops its cached pattern
        and runs ``install`` again before its next calculation.

        Examples
        --------
        A setting method of an instrument:

        >>> def set_xray_source(self, element):
        ...     self._xray_source = element
        ...     self.mark_settings_changed()
        """
        self._settings_version += 1
        self._flush(other=(self,))

    def get_settings(self):
        """Return the settings of the component, for showing to a user.

        A generator's ``print_info`` shows them next to the Parameters.
        A component with settings overrides this to list the settings in
        use, each under the name of the setter argument that sets it, so
        a user sees what to pass to change it.

        Returns
        -------
        dict of str to object
            The value of each setting in use, by name. It is empty for a
            component without settings.
        """
        settings = {}
        return settings

    def install(self, pattern, diffraction):
        """Set the model up in one generator.

        The generator runs this once it has both the component and a
        structure, and again after a setting of the component changes, so
        it must be safe to repeat.

        Parameters
        ----------
        pattern : pyobjcryst.powderpattern.PowderPattern
            The pattern of the generator.
        diffraction : pyobjcryst.powderpattern.PowderPatternDiffraction
            The diffraction component of the generator's structure.
        """
        raise NotImplementedError

    def update(self, pattern, diffraction):
        """Push the Parameter values into the pyobjcryst objects.

        The generator runs this before every calculation.

        Parameters
        ----------
        pattern : pyobjcryst.powderpattern.PowderPattern
            The pattern of the generator.
        diffraction : pyobjcryst.powderpattern.PowderPatternDiffraction
            The diffraction component of the generator's structure.
        """
        raise NotImplementedError

    def create_parameters_from(self, refinable_obj, names, target="default"):
        """Create the Parameters of the component from a pyobjcryst
        model.

        Each Parameter takes the default value of the matching
        ``RefinablePar`` and its limits as bounds. ObjCryst silently clamps
        a value outside its limits, so the bounds keep a refinement within
        the values ObjCryst actually uses.

        The Parameter names are mapped to the ``RefinablePar`` names
        rather than copied, so the names users see are this package's,
        independent of ObjCryst's. Some ObjCryst names are not valid
        Parameter names (``2ThetaDispl``). The mapping keeps ObjCryst
        parameters out of refinement scripts. The mapping is stored under
        ``target``, so ``push_parameters`` writes each value back to the
        same ``RefinablePar``.

        Parameters
        ----------
        refinable_obj : pyobjcryst.refinableobj.RefinableObj
            A fresh instance of the pyobjcryst model.
        names : dict of str to str
            The ``RefinablePar`` name for each Parameter name, for example
            ``{"eta0": "Eta0"}``. Only the named ``RefinablePar`` objects
            get a Parameter.
        target : str, optional
            The name of the pyobjcryst object the Parameters belong to
            (default "default"). A component that configures more than one
            pyobjcryst object, such as a pattern and its radiation, creates
            the Parameters of each object under its own target.

        Raises
        ------
        ValueError
            If the model has no ``RefinablePar`` with one of the names.

        Examples
        --------
        In the ``__init__`` of a peak profile:

        >>> self.create_parameters_from(
        ...     ReflectionProfilePseudoVoigt(),
        ...     {"U": "U", "V": "V", "W": "W", "eta0": "Eta0"},
        ... )
        """
        available_names = [
            refinable_obj.GetPar(index).GetName()
            for index in range(refinable_obj.GetNbPar())
        ]
        for refinable_par_name in names.values():
            if refinable_par_name not in available_names:
                raise ValueError(
                    f"{type(refinable_obj).__name__} has no refinable "
                    f"parameter named '{refinable_par_name}'. Please use one "
                    f"of its parameter names: {', '.join(available_names)}."
                )
        for parameter_name, refinable_par_name in names.items():
            refinable_par = refinable_obj.GetPar(refinable_par_name)
            parameter = Parameter(
                parameter_name, refinable_par.GetHumanValue()
            )
            if refinable_par.IsLimited():
                parameter.bound_range(
                    refinable_par.GetHumanMin(), refinable_par.GetHumanMax()
                )
            self.add_parameter(parameter)
        self._refinable_par_names.setdefault(target, {}).update(names)

    def push_parameters(self, refinable_obj, target="default"):
        """Push the Parameter values into a pyobjcryst model.

        Each value of the target's Parameters is written to the
        ``RefinablePar`` that ``create_parameters_from`` mapped it to. The
        Parameters of other targets are neither checked nor pushed.

        A value outside the limits of its ``RefinablePar`` raises an error
        instead of being clamped, because ObjCryst would otherwise use a
        different value from the one the Parameter reports. The check uses
        ObjCryst's limits, not the Parameter's bounds: a value outside
        narrowed bounds is still pushed, and widened bounds don't let a
        value past a limit. An optimizer run without the recipe's bounds
        stops with this error the first time it steps past a limit.

        All the values of the target are checked before any is pushed,
        so an error leaves the model unchanged rather than half updated.
        A component with several targets pushes them one at a time, so an
        error in a later target leaves the earlier targets' objects
        updated. The calculation stops with the error, and the next one
        pushes every value again.

        Parameters
        ----------
        refinable_obj : pyobjcryst.refinableobj.RefinableObj
            The pyobjcryst model to update.
        target : str, optional
            The target whose Parameters are pushed (default "default"), as
            given to ``create_parameters_from``. Only that target's values
            are checked and pushed.

        Raises
        ------
        ValueError
            If the component has no Parameters for the target, or a
            Parameter value is outside the limits of its ``RefinablePar``.
        """
        if target not in self._refinable_par_names:
            raise ValueError(
                f"The component '{self.name}' has no Parameters for the "
                f"target '{target}'. Please use one of its targets: "
                f"{', '.join(self._refinable_par_names)}."
            )
        refinable_par_names = self._refinable_par_names[target]
        values = {}
        refinable_pars = {}
        for parameter_name, refinable_par_name in refinable_par_names.items():
            values[parameter_name] = self.get(parameter_name).value
            refinable_par = refinable_obj.GetPar(refinable_par_name)
            refinable_pars[parameter_name] = refinable_par
        for parameter_name, refinable_par in refinable_pars.items():
            if not refinable_par.IsLimited():
                continue
            value = values[parameter_name]
            lower_limit = refinable_par.GetHumanMin()
            upper_limit = refinable_par.GetHumanMax()
            if not lower_limit <= value <= upper_limit:
                used_value = min(max(value, lower_limit), upper_limit)
                raise ValueError(
                    f"The value of '{parameter_name}', {value}, is outside "
                    f"its limits [{lower_limit}, {upper_limit}]. ObjCryst "
                    f"would silently use {used_value} instead, so the refined "
                    "value would not be the one used to calculate the "
                    "pattern. Please keep the value within the limits, for "
                    "example by passing the recipe's bounds to the optimizer: "
                    "least_squares(..., bounds=recipe.get_bounds_array())."
                )
        for parameter_name, refinable_par in refinable_pars.items():
            refinable_par.SetHumanValue(values[parameter_name])


# End of file
