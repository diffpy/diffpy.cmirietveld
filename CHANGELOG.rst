=============
Release notes
=============

.. current developments

0.1.0
=====

**Added:**

* Add ``ConstantWavelengthInstrument``, the instrument of a constant-wavelength x-ray, neutron or electron diffractometer, with refinable ``zero``, ``displacement``, ``transparency``, ``mu_r`` and ``wavelength``, and settings for the radiation type, an x-ray source's Kalpha1/Kalpha2 doublet and the polarization rate.
* Warn when a ``ConstantWavelengthInstrument`` setter replaces a refined wavelength or a Kalpha1/Kalpha2 doublet, or when a polarization rate has no effect because the radiation isn't x-rays.
* Add ``Component.create_parameters_from``, which creates a component's Parameters from a pyobjcryst model, with the model's default values and its limits as bounds, and ``Component.push_parameters``, which writes the values back and raises a ``ValueError`` for a value outside the limits instead of letting ObjCryst clamp it silently.
* Add the skeleton of the package: ``PowderDiffractionGenerator``, the ``Component`` base class, the peak profile, instrument, intensity correction and intensity source components, ``ChebyshevBackground``, and the parsers of data and instrument parameter files. The classes are stubs that raise ``NotImplementedError`` until they are filled in.
* Add ``Component.settings_version`` and ``Component.mark_settings_changed``, so a generator can run ``install`` again after a setting of a component changes.
* Add a ``target`` argument to ``Component.create_parameters_from`` and ``Component.push_parameters``, so one component can configure several pyobjcryst objects, such as a pattern and its radiation.
* Add the ``PseudoVoigtProfile`` and ``TCHProfile`` peak profile components, which create their Parameters from pyobjcryst's ``ReflectionProfilePseudoVoigt`` and ``ReflectionProfilePseudoVoigtTCH`` and configure a pyobjcryst powder pattern with ``install`` and ``update``.
* ``GSASDataParser`` reads GSAS STD and ESD powder data files, with counting-statistics uncertainties for STD files.
* Add ``PowderDiffractionGenerator``, which calculates one phase's Bragg intensity with pyobjcryst from a structure set with ``set_structure`` (a pyobjcryst ``Crystal``, the path to a CIF file or a ParameterSet), a peak profile set with ``set_peak_profile`` and an instrument set with ``set_instrument``, and a refinable ``scale``. It has no default peak profile or instrument, so all three must be set before a pattern is calculated.
* Add ``PowderDiffractionGenerator.print_info``, which prints the generator's structure, instrument and peak profile with their settings and Parameter values.
* Add ``Component.get_settings``, which returns a component's settings for showing to a user, and its ``ConstantWavelengthInstrument`` version, which lists the radiation type, the x-ray source or Kalpha1/Kalpha2 wavelengths, and the polarization rate.
* Warn when ``set_structure``, ``set_peak_profile`` or ``set_instrument`` replaces a structure or component whose Parameters are used in a fit recipe.

