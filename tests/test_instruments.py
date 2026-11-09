import re
import warnings

import numpy as np
import pytest
from pyobjcryst.radiation import RadiationType

from diffpy.cmirietveld.instruments import ConstantWavelengthInstrument
from diffpy.srfit.exceptions import SrFitError

# ObjCryst limits the 2theta corrections to +-0.05 rad, in degrees
TWO_THETA_CORRECTION_LIMIT = np.degrees(0.05)
# ObjCryst's tabulated Cu tube wavelengths, in angstroms. ObjCryst
# stores them in single precision, so they aren't 1.540560 and
# 1.544390 exactly
CU_KALPHA1 = 1.540560007095337
CU_KALPHA2 = 1.5443899631500244
CU_MEAN_WAVELENGTH = 1.5418366591135662
# ObjCryst's Cu tube mean wavelength for a Kalpha2/Kalpha1 ratio of 0.4
CU_MEAN_WAVELENGTH_RATIO_0_4 = 1.5416542802538191


def test_parameter_names():
    # Test the name and Parameters of the instrument
    # Expected: It is named instrument and has the 2theta corrections, the
    # absorption and the wavelength.
    instrument = ConstantWavelengthInstrument()
    actual_name = instrument.name
    actual_parameter_names = sorted(instrument.get_names())
    expected_name = "instrument"
    expected_parameter_names = [
        "displacement",
        "mu_r",
        "transparency",
        "wavelength",
        "zero",
    ]
    assert actual_name == expected_name
    assert actual_parameter_names == expected_parameter_names


@pytest.mark.parametrize(
    "parameter_name, expected_value, expected_bounds",
    [
        # Test that the pattern Parameters take ObjCryst's defaults and
        # limits
        # C1: The zero shift is created from the pattern's Zero.
        # Expected: Its value is 0 and its bounds are +-0.05 rad, in degrees.
        (
            "zero",
            0.0,
            [-TWO_THETA_CORRECTION_LIMIT, TWO_THETA_CORRECTION_LIMIT],
        ),
        # C2: The displacement is created from the pattern's 2ThetaDispl.
        # Expected: Its value is 0 and its bounds are +-0.05 rad, in degrees.
        (
            "displacement",
            0.0,
            [-TWO_THETA_CORRECTION_LIMIT, TWO_THETA_CORRECTION_LIMIT],
        ),
        # C3: The transparency is created from the pattern's 2ThetaTransp.
        # Expected: Its value is 0 and its bounds are +-0.05 rad, in degrees.
        (
            "transparency",
            0.0,
            [-TWO_THETA_CORRECTION_LIMIT, TWO_THETA_CORRECTION_LIMIT],
        ),
        # C4: The absorption is created from the pattern's MuR.
        # Expected: Its value is 0 and its bounds are [0, 1000].
        ("mu_r", 0.0, [0.0, 1000.0]),
    ],
)
def test_pattern_parameter_values_and_bounds(
    parameter_name, expected_value, expected_bounds
):
    instrument = ConstantWavelengthInstrument()
    parameter = instrument.get(parameter_name)
    actual_value = parameter.value
    actual_bounds = list(parameter.bounds)
    assert actual_value == pytest.approx(expected_value)
    assert actual_bounds == pytest.approx(expected_bounds)


def test_wavelength_before_set():
    # Test the wavelength of a new instrument
    # Expected: It has no value, because no dataset uses ObjCryst's default
    # of 1 angstrom, and its bounds are the Wavelength limits [0.05, 20].
    instrument = ConstantWavelengthInstrument()
    actual_value = instrument.wavelength.value
    actual_bounds = list(instrument.wavelength.bounds)
    expected_value = None
    expected_bounds = [0.05, 20.0]
    assert actual_value == expected_value
    assert actual_bounds == pytest.approx(expected_bounds)


def test_validate_before_wavelength_set():
    # Test validating an instrument before a wavelength is set, as srfit
    # does before a fit
    # Expected: An SrFitError says that the wavelength is missing and how
    # to set it.
    instrument = ConstantWavelengthInstrument()
    expected_msg = (
        "The instrument has no wavelength. Please set one with "
        "set_wavelength, set_xray_source or set_xray_wavelengths."
    )
    with pytest.raises(SrFitError, match=re.escape(expected_msg)):
        instrument._validate()


@pytest.mark.parametrize(
    "setter_name, setter_args, expected_wavelength",
    [
        # Test that each way of setting the wavelength sets the wavelength
        # Parameter, so the instrument validates
        # C1: A single wavelength is set.
        # Expected: The wavelength is that wavelength.
        ("set_wavelength", [1.5406], 1.5406),
        # C2: A Cu tube is set.
        # Expected: The wavelength is the intensity-weighted mean of
        # ObjCryst's Cu Kalpha1 and Kalpha2, as ObjCryst stores it.
        ("set_xray_source", ["Cu"], CU_MEAN_WAVELENGTH),
        # C3: The Cu Kalpha1 and Kalpha2 wavelengths are set explicitly.
        # Expected: The wavelength is their intensity-weighted mean,
        # Kalpha1 + (Kalpha2 - Kalpha1) * 0.5 / 1.5.
        (
            "set_xray_wavelengths",
            [CU_KALPHA1, CU_KALPHA2],
            CU_MEAN_WAVELENGTH,
        ),
        # C4: A Cu tube is set with a Kalpha2/Kalpha1 ratio of 0.4.
        # Expected: The wavelength is the mean weighted by that ratio,
        # Kalpha1 + (Kalpha2 - Kalpha1) * 0.4 / 1.4, as ObjCryst stores it.
        ("set_xray_source", ["Cu", 0.4], CU_MEAN_WAVELENGTH_RATIO_0_4),
        # C5: The Cu Kalpha1 and Kalpha2 wavelengths are set explicitly
        # with a Kalpha2/Kalpha1 ratio of 0.4.
        # Expected: The wavelength is the mean weighted by that ratio.
        (
            "set_xray_wavelengths",
            [CU_KALPHA1, CU_KALPHA2, 0.4],
            CU_MEAN_WAVELENGTH_RATIO_0_4,
        ),
    ],
)
def test_wavelength_after_setter(
    setter_name, setter_args, expected_wavelength
):
    instrument = ConstantWavelengthInstrument()
    getattr(instrument, setter_name)(*setter_args)
    actual_wavelength = instrument.wavelength.value
    assert actual_wavelength == pytest.approx(expected_wavelength)
    instrument._validate()


@pytest.mark.parametrize(
    "setter_name, setter_args, expected_settings",
    [
        # Test the settings an instrument reports to print_info. Only the
        # settings in use are listed, each under the name of the setter
        # argument that sets it
        # C1: A user sets a single wavelength.
        # Expected: The radiation type and the polarization rate.
        (
            "set_wavelength",
            [1.5406],
            {"radiation_type": "X", "polarization_rate": 0.0},
        ),
        # C2: A user sets a Cu tube.
        # Expected: The tube and its Kalpha2/Kalpha1 ratio are listed too.
        (
            "set_xray_source",
            ["Cu"],
            {
                "radiation_type": "X",
                "xray_source": "Cu",
                "kalpha2_ratio": 0.5,
                "polarization_rate": 0.0,
            },
        ),
        # C3: A user sets the Kalpha1 and Kalpha2 wavelengths from their
        # instrument file, with a ratio of 0.4.
        # Expected: The two wavelengths and the ratio are listed.
        (
            "set_xray_wavelengths",
            [CU_KALPHA1, CU_KALPHA2, 0.4],
            {
                "radiation_type": "X",
                "kalpha1": CU_KALPHA1,
                "kalpha2": CU_KALPHA2,
                "kalpha2_ratio": 0.4,
                "polarization_rate": 0.0,
            },
        ),
        # C4: A user switches to neutrons.
        # Expected: The radiation type is N.
        (
            "set_radiation_type",
            ["N"],
            {"radiation_type": "N", "polarization_rate": 0.0},
        ),
        # C5: A user sets a polarization rate of 0.95.
        # Expected: The polarization rate is 0.95.
        (
            "set_polarization",
            [0.95],
            {"radiation_type": "X", "polarization_rate": 0.95},
        ),
    ],
)
def test_get_settings(setter_name, setter_args, expected_settings):
    instrument = ConstantWavelengthInstrument()
    getattr(instrument, setter_name)(*setter_args)
    actual_settings = instrument.get_settings()
    assert actual_settings == expected_settings


@pytest.mark.parametrize(
    "setter_name, setter_args",
    [
        # Test that each setting method is a settings change, so every
        # generator holding the instrument runs install again
        # C1: The radiation type is set.
        # Expected: settings_version goes from 0 to 1.
        ("set_radiation_type", ["N"]),
        # C2: A single wavelength is set, which also sets the wavelength
        # type.
        # Expected: settings_version goes from 0 to 1.
        ("set_wavelength", [1.5406]),
        # C3: A Cu tube is set.
        # Expected: settings_version goes from 0 to 1.
        ("set_xray_source", ["Cu"]),
        # C4: The Kalpha1 and Kalpha2 wavelengths are set explicitly.
        # Expected: settings_version goes from 0 to 1.
        ("set_xray_wavelengths", [CU_KALPHA1, CU_KALPHA2]),
        # C5: The polarization rate is set.
        # Expected: settings_version goes from 0 to 1.
        ("set_polarization", [0.95]),
    ],
)
def test_settings_version_after_setter(setter_name, setter_args):
    instrument = ConstantWavelengthInstrument()
    getattr(instrument, setter_name)(*setter_args)
    actual_settings_version = instrument.settings_version
    expected_settings_version = 1
    assert actual_settings_version == expected_settings_version


@pytest.mark.parametrize(
    "setter_name, first_args, refined_wavelength, second_args, expected_msg",
    [
        # Test that a setter warns when it replaces a wavelength refined
        # since it was last called, so a calibration isn't lost without the
        # user knowing
        # C1: A user sets the wavelength to the synchrotron's wavelength of
        # 0.4592 angstroms, refines it to 0.45903 against a calibrant,
        # then reruns the setup cell of their script, which sets 0.4592
        # again.
        # Expected: A UserWarning says that the refined 0.45903 was
        # replaced by 0.4592, and how to keep it.
        (
            "set_wavelength",
            [0.4592],
            0.45903,
            [0.4592],
            "set_wavelength replaced the wavelength 0.45903 angstroms, which "
            "was refined or set by hand, with 0.4592 angstroms. To keep "
            "0.45903 angstroms, set instrument.wavelength again after "
            "set_wavelength.",
        ),
        # C2: A user sets a Cu tube, refines the wavelength to 1.5422
        # angstroms against a calibrant, then sets the tube again with a
        # different Kalpha2/Kalpha1 ratio.
        # Expected: A UserWarning says that the refined 1.5422 was replaced
        # by the tube's mean for that ratio, 1.54175, and how to keep it.
        (
            "set_xray_source",
            ["Cu"],
            1.5422,
            ["Cu", 0.45],
            "set_xray_source replaced the wavelength 1.5422 angstroms, which "
            "was refined or set by hand, with 1.54175 angstroms. To keep "
            "1.5422 angstroms, set instrument.wavelength again after "
            "set_xray_source.",
        ),
        # C3: A user sets their own Kalpha1 and Kalpha2 wavelengths, refines
        # the wavelength to 1.5422 angstroms against a calibrant, then
        # reruns the setup cell of their script, which sets the same
        # wavelengths again.
        # Expected: A UserWarning says that the refined 1.5422 was replaced
        # by the mean of the two wavelengths, 1.54184, and how to keep it.
        (
            "set_xray_wavelengths",
            [CU_KALPHA1, CU_KALPHA2],
            1.5422,
            [CU_KALPHA1, CU_KALPHA2],
            "set_xray_wavelengths replaced the wavelength 1.5422 angstroms, "
            "which was refined or set by hand, with 1.54184 angstroms. To "
            "keep 1.5422 angstroms, set instrument.wavelength again after "
            "set_xray_wavelengths.",
        ),
    ],
)
def test_setter_warns_wavelength_replaced(
    setter_name, first_args, refined_wavelength, second_args, expected_msg
):
    # Set up the instrument's wavelength
    instrument = ConstantWavelengthInstrument()
    getattr(instrument, setter_name)(*first_args)
    # Refine the wavelength against the calibrant, here set by hand
    instrument.wavelength.set_value(refined_wavelength)
    # Call the setter again
    with pytest.warns(UserWarning, match=re.escape(expected_msg)):
        getattr(instrument, setter_name)(*second_args)


@pytest.mark.parametrize(
    "first_setter_name, first_args, second_setter_name, second_args, "
    "expected_msg",
    [
        # Test that a setter warns when it replaces a Kalpha1/Kalpha2
        # doublet set by another setter, so the doublet isn't lost without
        # the user knowing
        # C1: A user sets a Cu tube, then changes their mind and sets a
        # single wavelength of 1.5406 angstroms.
        # Expected: A UserWarning says that the Cu doublet was replaced by
        # a single wavelength, and how to use the doublet again.
        (
            "set_xray_source",
            ["Cu"],
            "set_wavelength",
            [1.5406],
            "set_wavelength replaced the Cu Kalpha1/Kalpha2 doublet with a "
            "single wavelength. To use the doublet again, call "
            "set_xray_source.",
        ),
        # C2: A user sets their own Kalpha1 and Kalpha2 wavelengths, then
        # changes their mind and sets a single wavelength of 1.5406
        # angstroms.
        # Expected: A UserWarning says that the doublet was replaced by a
        # single wavelength, and how to use the doublet again.
        (
            "set_xray_wavelengths",
            [CU_KALPHA1, CU_KALPHA2],
            "set_wavelength",
            [1.5406],
            "set_wavelength replaced the explicit Kalpha1/Kalpha2 doublet "
            "with a single wavelength. To use the doublet again, call "
            "set_xray_wavelengths.",
        ),
        # C3: A user sets a Cu tube, then sets their own Kalpha1 and Kalpha2
        # wavelengths instead.
        # Expected: A UserWarning says that the Cu doublet was replaced,
        # and how to use it again.
        (
            "set_xray_source",
            ["Cu"],
            "set_xray_wavelengths",
            [CU_KALPHA1, CU_KALPHA2],
            "set_xray_wavelengths replaced the Cu Kalpha1/Kalpha2 doublet "
            "with the explicit wavelengths. To use the Cu doublet again, "
            "call set_xray_source.",
        ),
        # C4: A user sets their own Kalpha1 and Kalpha2 wavelengths, then
        # copies in code from another script that sets a Cu tube.
        # Expected: A UserWarning says that their wavelengths were replaced
        # by the Cu doublet, and how to use them again.
        (
            "set_xray_wavelengths",
            [CU_KALPHA1, CU_KALPHA2],
            "set_xray_source",
            ["Cu"],
            "set_xray_source replaced the explicit Kalpha1/Kalpha2 doublet "
            "with the Cu doublet. To use the explicit wavelengths again, call "
            "set_xray_wavelengths.",
        ),
    ],
)
def test_setter_warns_doublet_replaced(
    first_setter_name,
    first_args,
    second_setter_name,
    second_args,
    expected_msg,
):
    instrument = ConstantWavelengthInstrument()
    getattr(instrument, first_setter_name)(*first_args)
    with pytest.warns(UserWarning, match=re.escape(expected_msg)):
        getattr(instrument, second_setter_name)(*second_args)


@pytest.mark.parametrize(
    "first_setter_name, first_args, second_setter_name, second_args, "
    "expected_msg",
    [
        # Test that a polarization rate warns when it has no effect, since
        # ObjCryst only applies it to x-rays
        # C1: A user sets up a neutron instrument, then copies in a
        # polarization rate of 0.95 from their x-ray script.
        # Expected: A UserWarning says that the rate does nothing with
        # neutrons.
        (
            "set_radiation_type",
            ["N"],
            "set_polarization",
            [0.95],
            "The polarization rate 0.95 has no effect with the radiation type "
            "'N', because ObjCryst only applies it to x-rays. Please set a "
            "polarization rate only for x-ray data.",
        ),
        # C2: A user sets a polarization rate of 0.95 for their x-ray data,
        # then switches the instrument to neutrons.
        # Expected: A UserWarning says that the rate now does nothing.
        (
            "set_polarization",
            [0.95],
            "set_radiation_type",
            ["N"],
            "The polarization rate 0.95 has no effect with the radiation type "
            "'N', because ObjCryst only applies it to x-rays. Please set a "
            "polarization rate only for x-ray data.",
        ),
        # C3: A user sets a polarization rate of 0.95, then switches the
        # instrument to electrons.
        # Expected: A UserWarning says that the rate now does nothing.
        (
            "set_polarization",
            [0.95],
            "set_radiation_type",
            ["E"],
            "The polarization rate 0.95 has no effect with the radiation type "
            "'E', because ObjCryst only applies it to x-rays. Please set a "
            "polarization rate only for x-ray data.",
        ),
    ],
)
def test_polarization_warns_without_xrays(
    first_setter_name,
    first_args,
    second_setter_name,
    second_args,
    expected_msg,
):
    instrument = ConstantWavelengthInstrument()
    getattr(instrument, first_setter_name)(*first_args)
    with pytest.warns(UserWarning, match=re.escape(expected_msg)):
        getattr(instrument, second_setter_name)(*second_args)


@pytest.mark.parametrize(
    "first_setter_name, first_args, second_setter_name, second_args",
    [
        # Test that the setters stay silent when a user changes their own
        # setup before refining, so the warnings only appear when something
        # the user may still expect is lost
        # C1: A user sets a Cu tube, then switches to a Mo tube before
        # refining.
        # Expected: No warning, because the wavelength replaced is the Cu
        # tube's, not a refined one.
        ("set_xray_source", ["Cu"], "set_xray_source", ["Mo"]),
        # C2: A user sets the synchrotron's wavelength of 0.4592 angstroms,
        # then corrects it to 0.45917 before refining.
        # Expected: No warning, because the wavelength replaced is the one
        # the user set.
        ("set_wavelength", [0.4592], "set_wavelength", [0.45917]),
        # C3: A user sets a single wavelength of 1.5406 angstroms, then
        # switches to a Cu tube.
        # Expected: No warning, because no doublet is replaced.
        ("set_wavelength", [1.5406], "set_xray_source", ["Cu"]),
        # C4: A user sets a single wavelength of 1.5406 angstroms, then
        # switches to their own Kalpha1 and Kalpha2 wavelengths.
        # Expected: No warning, because no doublet is replaced.
        (
            "set_wavelength",
            [1.5406],
            "set_xray_wavelengths",
            [CU_KALPHA1, CU_KALPHA2],
        ),
        # C5: A user sets a Cu tube, then a polarization rate of 0.95.
        # Expected: No warning, because the rate applies to x-rays.
        ("set_xray_source", ["Cu"], "set_polarization", [0.95]),
        # C6: A user sets up a neutron instrument and leaves the
        # polarization at 0.
        # Expected: No warning, because a rate of 0 changes nothing.
        ("set_radiation_type", ["N"], "set_polarization", [0.0]),
    ],
)
def test_setter_without_warning(
    first_setter_name, first_args, second_setter_name, second_args
):
    # Record every warning from the first setter call, which replaces no
    # wavelength, and from the second
    instrument = ConstantWavelengthInstrument()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        getattr(instrument, first_setter_name)(*first_args)
        getattr(instrument, second_setter_name)(*second_args)
    actual_warning_messages = []
    for caught_warning in caught_warnings:
        if issubclass(caught_warning.category, UserWarning):
            actual_warning_messages.append(str(caught_warning.message))
    expected_warning_messages = []
    assert actual_warning_messages == expected_warning_messages


@pytest.mark.parametrize(
    "parameter_name, pyobjcryst_par_name, new_value",
    [
        # Test that update pushes the pattern Parameters into the pattern
        # C1: The zero shift is changed.
        # Expected: The pattern equals one with Zero set directly.
        ("zero", "Zero", 0.05),
        # C2: The displacement is changed.
        # Expected: The pattern equals one with 2ThetaDispl set directly.
        ("displacement", "2ThetaDispl", 0.05),
        # C3: The transparency is changed.
        # Expected: The pattern equals one with 2ThetaTransp set directly.
        ("transparency", "2ThetaTransp", 0.05),
        # C4: The cylinder absorption is set to 1.
        # Expected: The pattern equals one with MuR set directly.
        ("mu_r", "MuR", 1.0),
    ],
)
def test_update_pattern_parameters(
    pbso4_pattern,
    other_pbso4_pattern,
    parameter_name,
    pyobjcryst_par_name,
    new_value,
):
    # Install the instrument at 1.5406 angstroms, change a Parameter and
    # push it with update
    instrument = ConstantWavelengthInstrument()
    instrument.set_wavelength(1.5406)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.get(parameter_name).set_value(new_value)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, already at 1.5406 angstroms, set the same value
    # directly in pyobjcryst
    other_pbso4_pattern.pattern.GetPar(pyobjcryst_par_name).SetHumanValue(
        new_value
    )
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_update_wavelength(pbso4_pattern, other_pbso4_pattern):
    # Test that update pushes the wavelength into the pattern's radiation
    # Expected: After the wavelength is changed from 1.5406 to 1.6
    # angstroms, the pattern equals one with the Radiation's Wavelength set
    # directly.
    # Install the instrument at 1.5406 angstroms, then refine the
    # wavelength to 1.6 angstroms
    instrument = ConstantWavelengthInstrument()
    instrument.set_wavelength(1.5406)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.wavelength.set_value(1.6)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, set the Wavelength directly in pyobjcryst
    radiation = other_pbso4_pattern.pattern.GetRadiation()
    radiation.GetPar("Wavelength").SetHumanValue(1.6)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_install_shared_instrument(pbso4_pattern, other_pbso4_pattern):
    # Test one instrument shared by two patterns, as the generators of the
    # phases of one dataset share it
    # Expected: Both patterns get the instrument's radiation type,
    # wavelength and zero shift.
    # Set up one neutron instrument with a zero shift
    instrument = ConstantWavelengthInstrument()
    instrument.set_radiation_type("N")
    instrument.set_wavelength(1.909)
    instrument.zero.set_value(0.05)
    shared_patterns = [pbso4_pattern, other_pbso4_pattern]
    # Install and update it in both patterns before reading either, so a
    # change to the second pattern that reaches the first is caught
    for shared_pattern in shared_patterns:
        instrument.install(shared_pattern.pattern, shared_pattern.diffraction)
        instrument.update(shared_pattern.pattern, shared_pattern.diffraction)
    actual_radiation_types = []
    actual_wavelengths = []
    actual_zeros = []
    for shared_pattern in shared_patterns:
        radiation = shared_pattern.pattern.GetRadiation()
        actual_radiation_types.append(radiation.GetRadiationType())
        actual_wavelengths.append(radiation.GetWavelength())
        zero = shared_pattern.pattern.GetPar("Zero").GetHumanValue()
        actual_zeros.append(zero)
    expected_radiation_types = [
        RadiationType.RAD_NEUTRON,
        RadiationType.RAD_NEUTRON,
    ]
    expected_wavelengths = [1.909, 1.909]
    expected_zeros = [0.05, 0.05]
    assert actual_radiation_types == expected_radiation_types
    assert actual_wavelengths == pytest.approx(expected_wavelengths)
    assert actual_zeros == pytest.approx(expected_zeros)


def test_install_neutron_radiation(pbso4_pattern, other_pbso4_pattern):
    # Test installing neutron radiation
    # Expected: The pattern equals a pyobjcryst pattern set directly to
    # neutrons at 1.909 angstroms.
    # Install a neutron instrument
    instrument = ConstantWavelengthInstrument()
    instrument.set_radiation_type("N")
    instrument.set_wavelength(1.909)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, set neutrons directly in pyobjcryst
    other_pbso4_pattern.pattern.SetRadiationType(RadiationType.RAD_NEUTRON)
    other_pbso4_pattern.pattern.SetWavelength(1.909)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


@pytest.mark.parametrize(
    "radiation_type, wavelength, expected_radiation_type",
    [
        # Test that install sets the radiation type of the pattern
        # C1: Neutrons at 1.909 angstroms are set.
        # Expected: The pattern's radiation is neutrons.
        ("N", 1.909, RadiationType.RAD_NEUTRON),
        # C2: Electrons at 0.0536 angstroms, those of a 50 kV beam, are
        # set. Faster electrons are below ObjCryst's wavelength limit of
        # 0.05 angstroms.
        # Expected: The pattern's radiation is electrons.
        ("E", 0.0536, RadiationType.RAD_ELECTRON),
    ],
)
def test_install_radiation_type(
    pbso4_pattern, radiation_type, wavelength, expected_radiation_type
):
    instrument = ConstantWavelengthInstrument()
    instrument.set_radiation_type(radiation_type)
    instrument.set_wavelength(wavelength)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    radiation = pbso4_pattern.pattern.GetRadiation()
    actual_radiation_type = radiation.GetRadiationType()
    assert actual_radiation_type == expected_radiation_type


# The doublet-replaced warning is tested in
# test_setter_warns_doublet_replaced
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_install_neutrons_after_tube(pbso4_pattern, other_pbso4_pattern):
    # Test switching an installed x-ray tube instrument to neutrons
    # Expected: The pattern equals a pyobjcryst pattern set directly to
    # neutrons at 1.909 angstroms, with no doublet left from the tube.
    # Install and calculate with a Cu tube
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_source("Cu")
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    pbso4_pattern.calculate()
    # Switch to a single neutron wavelength, then install and calculate
    # again
    instrument.set_wavelength(1.909)
    instrument.set_radiation_type("N")
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, set neutrons directly in pyobjcryst
    other_pbso4_pattern.pattern.SetRadiationType(RadiationType.RAD_NEUTRON)
    other_pbso4_pattern.pattern.SetWavelength(1.909)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_install_polarization_ratio(pbso4_pattern, other_pbso4_pattern):
    # Test the polarization factor that a polarization rate gives
    # Expected: With a Cu tube, a rate f = 0.95 changes the strongest
    # PbSO4 peak, relative to an unpolarized beam (f = 0), by
    # (1 + A cos^2 2theta) / (1 + A) / ((1 + cos^2 2theta) / 2), with
    # A = (1 - f) / (1 + f) and 2theta that peak's position. (The peak is
    # at 29.69 degrees, and the ratio is 1.133.)
    # Calculate the pattern of an unpolarized beam
    unpolarized_instrument = ConstantWavelengthInstrument()
    unpolarized_instrument.set_xray_source("Cu")
    unpolarized_instrument.install(
        pbso4_pattern.pattern, pbso4_pattern.diffraction
    )
    unpolarized_instrument.update(
        pbso4_pattern.pattern, pbso4_pattern.diffraction
    )
    unpolarized_pattern = pbso4_pattern.calculate()
    # Calculate the pattern of a beam with f = 0.95 on a second pattern
    polarized_instrument = ConstantWavelengthInstrument()
    polarized_instrument.set_xray_source("Cu")
    polarized_instrument.set_polarization(0.95)
    polarized_instrument.install(
        other_pbso4_pattern.pattern, other_pbso4_pattern.diffraction
    )
    polarized_instrument.update(
        other_pbso4_pattern.pattern, other_pbso4_pattern.diffraction
    )
    polarized_pattern = other_pbso4_pattern.calculate()
    # Compare the strongest peak with the polarization formula
    peak_index = np.argmax(unpolarized_pattern)
    peak_two_theta = np.degrees(
        pbso4_pattern.pattern.GetPowderPatternX()[peak_index]
    )
    cos_squared = np.cos(np.radians(peak_two_theta)) ** 2
    polarization_a = (1 - 0.95) / (1 + 0.95)
    actual_ratio = polarized_pattern[peak_index] / (
        unpolarized_pattern[peak_index]
    )
    expected_ratio = (
        (1 + polarization_a * cos_squared)
        / (1 + polarization_a)
        / ((1 + cos_squared) / 2)
    )
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-3)


def test_install_after_polarization_change(pbso4_pattern, other_pbso4_pattern):
    # Test installing again after only the polarization rate changed,
    # which ObjCryst doesn't recalculate on its own
    # Expected: The pattern equals a pyobjcryst pattern built with a Cu
    # tube and f = 0.95 from the start.
    # Install and calculate with an unpolarized Cu tube
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_source("Cu")
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    pbso4_pattern.calculate()
    # Change only the polarization rate, then install and calculate again
    instrument.set_polarization(0.95)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, build the Cu tube with f = 0.95 directly
    other_pbso4_pattern.pattern.SetWavelength("Cu", 0.5)
    other_pbso4_pattern.pattern.GetRadiation().SetLinearPolarRate(0.95)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_install_tube_set_after_polarization(
    pbso4_pattern, other_pbso4_pattern
):
    # Test setting the tube after the polarization rate, since ObjCryst
    # resets the rate when a tube is set
    # Expected: The pattern still has the rate, f = 0.95: it equals a
    # pyobjcryst pattern with the Cu tube set first and the rate second.
    # Set the polarization first and the tube second, then install
    instrument = ConstantWavelengthInstrument()
    instrument.set_polarization(0.95)
    instrument.set_xray_source("Cu")
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, set the Cu tube and then the rate directly
    other_pbso4_pattern.pattern.SetWavelength("Cu", 0.5)
    other_pbso4_pattern.pattern.GetRadiation().SetLinearPolarRate(0.95)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


@pytest.mark.parametrize(
    "polarization_rate",
    [
        # Test that the polarization rates at the edges of [-1, 1] are
        # accepted and installed
        # C1: The rate is -1.
        # Expected: The pattern's radiation has a rate of -1.
        -1.0,
        # C2: The rate is 1.
        # Expected: The pattern's radiation has a rate of 1.
        1.0,
    ],
)
def test_install_polarization_rate_at_range_edges(
    pbso4_pattern, polarization_rate
):
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_source("Cu")
    instrument.set_polarization(polarization_rate)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    radiation = pbso4_pattern.pattern.GetRadiation()
    actual_polarization_rate = radiation.GetLinearPolarRate()
    expected_polarization_rate = polarization_rate
    assert actual_polarization_rate == pytest.approx(
        expected_polarization_rate
    )


# The doublet-replaced warning is tested in
# test_setter_warns_doublet_replaced
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_install_single_wavelength_after_tube(
    pbso4_pattern, other_pbso4_pattern
):
    # Test switching from a tube to a single wavelength, since ObjCryst
    # keeps the tube's Kalpha1/Kalpha2 doublet after SetWavelength(float)
    # Expected: The pattern equals a single-wavelength pyobjcryst pattern
    # at 1.5406 angstroms, not a doublet.
    # Install and calculate with a Cu tube
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_source("Cu")
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    pbso4_pattern.calculate()
    # Switch to a single wavelength, then install and calculate again
    instrument.set_wavelength(1.5406)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # The second pattern was built at a single wavelength of 1.5406
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


@pytest.mark.parametrize(
    "kalpha2_ratio",
    [
        # Test installing a Cu tube
        # C1: The Kalpha2/Kalpha1 ratio is the default, 0.5.
        # Expected: The pattern equals a pyobjcryst pattern set directly to
        # the Cu tube with that ratio.
        0.5,
        # C2: The ratio is 0.4, which ObjCryst would clamp to 0.5 if it
        # were set through its RefinablePar.
        # Expected: The pattern equals a pyobjcryst pattern set directly to
        # the Cu tube with a ratio of 0.4.
        0.4,
    ],
)
def test_install_xray_source(
    pbso4_pattern, other_pbso4_pattern, kalpha2_ratio
):
    # Install the Cu tube
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_source("Cu", kalpha2_ratio)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, set the Cu tube directly in pyobjcryst
    other_pbso4_pattern.pattern.SetWavelength("Cu", kalpha2_ratio)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


@pytest.mark.parametrize(
    "kalpha2_ratio",
    [
        # Test setting the Kalpha1 and Kalpha2 wavelengths explicitly
        # C1: ObjCryst's Cu wavelengths are set with the default
        # Kalpha2/Kalpha1 ratio, 0.5.
        # Expected: The pattern equals a pyobjcryst pattern set directly to
        # the Cu tube with that ratio.
        0.5,
        # C2: ObjCryst's Cu wavelengths are set with a ratio of 0.4.
        # Expected: The pattern equals a pyobjcryst pattern set directly to
        # the Cu tube with a ratio of 0.4.
        0.4,
    ],
)
def test_install_xray_wavelengths(
    pbso4_pattern, other_pbso4_pattern, kalpha2_ratio
):
    # Install the explicit wavelengths
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_wavelengths(CU_KALPHA1, CU_KALPHA2, kalpha2_ratio)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, set the Cu tube directly in pyobjcryst
    other_pbso4_pattern.pattern.SetWavelength("Cu", kalpha2_ratio)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


# The doublet-replaced warnings of C3 and C4 are tested in
# test_setter_warns_doublet_replaced
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "first_setter_name, first_args, second_setter_name, second_args, "
    "expected_kalpha2_ratio",
    [
        # Test that the Kalpha2/Kalpha1 ratio is kept when a user changes
        # the doublet without giving a ratio, so it doesn't fall back to 0.5
        # without the user knowing
        # C1: A user sets a Cu tube, then switches to a Mo tube, without
        # giving a ratio either time.
        # Expected: The ratio is 0.5, the starting ratio.
        ("set_xray_source", ["Cu"], "set_xray_source", ["Mo"], 0.5),
        # C2: A user sets a Cu tube with a ratio of 0.45, then switches to a
        # Mo tube.
        # Expected: The ratio is still 0.45.
        ("set_xray_source", ["Cu", 0.45], "set_xray_source", ["Mo"], 0.45),
        # C3: A user sets a Cu tube with a ratio of 0.45, then switches to
        # their own Kalpha1 and Kalpha2 wavelengths.
        # Expected: The ratio is still 0.45.
        (
            "set_xray_source",
            ["Cu", 0.45],
            "set_xray_wavelengths",
            [CU_KALPHA1, CU_KALPHA2],
            0.45,
        ),
        # C4: A user sets their own Kalpha1 and Kalpha2 wavelengths with a
        # ratio of 0.45, then switches to a Cu tube.
        # Expected: The ratio is still 0.45.
        (
            "set_xray_wavelengths",
            [CU_KALPHA1, CU_KALPHA2, 0.45],
            "set_xray_source",
            ["Cu"],
            0.45,
        ),
        # C5: A user sets a Cu tube with a ratio of 0.45, then sets it again
        # with a ratio of 0.4.
        # Expected: The ratio is 0.4, the one given last.
        ("set_xray_source", ["Cu", 0.45], "set_xray_source", ["Cu", 0.4], 0.4),
    ],
)
def test_install_kalpha2_ratio_kept(
    pbso4_pattern,
    first_setter_name,
    first_args,
    second_setter_name,
    second_args,
    expected_kalpha2_ratio,
):
    # Set the first doublet, then change it
    instrument = ConstantWavelengthInstrument()
    getattr(instrument, first_setter_name)(*first_args)
    getattr(instrument, second_setter_name)(*second_args)
    # Install it and read the ratio ObjCryst uses
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    radiation = pbso4_pattern.pattern.GetRadiation()
    ratio_par = radiation.GetPar("XRayTubeAlpha2Alpha1Ratio")
    actual_kalpha2_ratio = ratio_par.GetHumanValue()
    assert actual_kalpha2_ratio == pytest.approx(expected_kalpha2_ratio)


def test_set_xray_source_unknown_element():
    # Test setting a tube whose element ObjCryst has no wavelengths for
    # Expected: A ValueError names the element and how to set the
    # wavelengths instead, rather than ObjCryst's printed warning.
    instrument = ConstantWavelengthInstrument()
    expected_msg = (
        "ObjCryst has no x-ray tube wavelengths for 'Xx'. Please use one of "
        "the anode elements it has, 'Ag', 'Co', 'Cr', 'Cu', 'Fe' or 'Mo', or "
        "set the wavelengths with set_xray_wavelengths."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        instrument.set_xray_source("Xx")


@pytest.mark.parametrize(
    "setter_name, setter_args, expected_msg",
    [
        # Test setting a wavelength outside ObjCryst's Wavelength limits,
        # [0.05, 20] angstroms, which ObjCryst would clamp
        # C1: A single wavelength above 20 angstroms is set.
        # Expected: A ValueError names the wavelength and the limits.
        (
            "set_wavelength",
            [25.0],
            "The wavelength 25.0 angstroms is outside ObjCryst's limits of "
            "0.05 to 20 angstroms. Please give a wavelength within these "
            "limits.",
        ),
        # C2: The 0.0251 angstroms of a 200 kV electron beam, below 0.05
        # angstroms, is set.
        # Expected: A ValueError names the wavelength and the limits.
        (
            "set_wavelength",
            [0.0251],
            "The wavelength 0.0251 angstroms is outside ObjCryst's limits of "
            "0.05 to 20 angstroms. Please give a wavelength within these "
            "limits.",
        ),
        # C3: Kalpha1 and Kalpha2 wavelengths above 20 angstroms are set.
        # Expected: A ValueError names Kalpha1, the first wavelength
        # outside the limits.
        (
            "set_xray_wavelengths",
            [25.0, 25.004],
            "The wavelength 25.0 angstroms is outside ObjCryst's limits of "
            "0.05 to 20 angstroms. Please give a wavelength within these "
            "limits.",
        ),
    ],
)
def test_set_wavelength_outside_limits(setter_name, setter_args, expected_msg):
    instrument = ConstantWavelengthInstrument()
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        getattr(instrument, setter_name)(*setter_args)


def test_set_xray_wavelengths_kalpha2_not_longer():
    # Test setting a Kalpha2 that isn't longer than Kalpha1, which would
    # make ObjCryst's Kalpha2 - Kalpha1 negative
    # Expected: A ValueError says that the wavelengths are swapped.
    instrument = ConstantWavelengthInstrument()
    expected_msg = (
        "Kalpha2, 1.54056 angstroms, is not longer than Kalpha1, 1.54439 "
        "angstroms. Please give the wavelengths in the order kalpha1, "
        "kalpha2."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        instrument.set_xray_wavelengths(1.544390, 1.540560)


@pytest.mark.parametrize(
    "setter_name, setter_args",
    [
        # Test setting a Kalpha1/Kalpha2 doublet while the radiation type
        # is neutrons, since a doublet is x-ray tube radiation
        # C1: A Cu tube is set.
        # Expected: A ValueError says that a doublet needs x-rays.
        ("set_xray_source", ["Cu"]),
        # C2: The Cu Kalpha1 and Kalpha2 wavelengths are set explicitly.
        # Expected: A ValueError says that a doublet needs x-rays.
        ("set_xray_wavelengths", [CU_KALPHA1, CU_KALPHA2]),
    ],
)
def test_set_doublet_with_neutron_radiation(setter_name, setter_args):
    instrument = ConstantWavelengthInstrument()
    instrument.set_radiation_type("N")
    expected_msg = (
        "A Kalpha1/Kalpha2 doublet needs x-rays, but the radiation type is "
        "'N'. Please set the radiation type to 'X' first, or set a single "
        "wavelength with set_wavelength."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        getattr(instrument, setter_name)(*setter_args)


@pytest.mark.parametrize(
    "radiation_type, expected_msg",
    [
        # Test setting a radiation type other than x-rays while a
        # Kalpha1/Kalpha2 doublet is set
        # C1: Neutrons are set after a Cu tube.
        # Expected: A ValueError says to set a single wavelength first.
        (
            "N",
            "The radiation type 'N' can't be used with the Kalpha1/Kalpha2 "
            "doublet that is set, which needs x-rays. Please set a single "
            "wavelength with set_wavelength first.",
        ),
        # C2: Electrons are set after a Cu tube.
        # Expected: A ValueError says to set a single wavelength first.
        (
            "E",
            "The radiation type 'E' can't be used with the Kalpha1/Kalpha2 "
            "doublet that is set, which needs x-rays. Please set a single "
            "wavelength with set_wavelength first.",
        ),
    ],
)
def test_set_radiation_type_with_doublet(radiation_type, expected_msg):
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_source("Cu")
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        instrument.set_radiation_type(radiation_type)


def test_set_radiation_type_unknown_type():
    # Test setting a radiation type that isn't x-rays, neutrons or
    # electrons
    # Expected: A ValueError names the type and lists the supported ones.
    instrument = ConstantWavelengthInstrument()
    expected_msg = (
        "'Q' is not a supported radiation type. Please use 'X' for x-rays, "
        "'N' for neutrons or 'E' for electrons."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        instrument.set_radiation_type("Q")


@pytest.mark.parametrize(
    "polarization_rate, expected_msg",
    [
        # Test setting a polarization rate outside [-1, 1]
        # C1: The rate is above 1.
        # Expected: A ValueError names the rate and how to convert a GSAS
        # polarization.
        (
            2.0,
            "The polarization rate 2.0 is outside [-1, 1]. Please give "
            "ObjCryst's rate f; for a GSAS polarization P, f = 2P - 1.",
        ),
        # C2: The rate is below -1.
        # Expected: A ValueError names the rate and how to convert a GSAS
        # polarization.
        (
            -1.5,
            "The polarization rate -1.5 is outside [-1, 1]. Please give "
            "ObjCryst's rate f; for a GSAS polarization P, f = 2P - 1.",
        ),
    ],
)
def test_set_polarization_outside_range(polarization_rate, expected_msg):
    instrument = ConstantWavelengthInstrument()
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        instrument.set_polarization(polarization_rate)
