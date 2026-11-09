**Added:**

* Add ``PowderDiffractionGenerator``, which calculates one phase's Bragg intensity with pyobjcryst from a structure set with ``set_structure`` (a pyobjcryst ``Crystal``, the path to a CIF file or a ParameterSet), a peak profile set with ``set_peak_profile`` and an instrument set with ``set_instrument``, and a refinable ``scale``. It has no default peak profile or instrument, so all three must be set before a pattern is calculated.
* Add ``PowderDiffractionGenerator.print_info``, which prints the generator's structure, instrument and peak profile with their settings and Parameter values.
* Add ``Component.get_settings``, which returns a component's settings for showing to a user, and its ``ConstantWavelengthInstrument`` version, which lists the radiation type, the x-ray source or Kalpha1/Kalpha2 wavelengths, and the polarization rate.
* Warn when ``set_structure``, ``set_peak_profile`` or ``set_instrument`` replaces a structure or component whose Parameters are used in a fit recipe.

**Changed:**

* <news item>

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
