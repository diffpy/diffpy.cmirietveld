import numpy as np
import pytest
from pyobjcryst.reflectionprofile import (
    ReflectionProfilePseudoVoigt,
    ReflectionProfilePseudoVoigtTCH,
)

from diffpy.cmirietveld.peak_profiles import PseudoVoigtProfile, TCHProfile


@pytest.mark.parametrize(
    "profile_class, expected_names",
    [
        # Test the name and Parameters of each peak profile
        # C1: A pseudo-Voigt profile is created.
        # Expected: It is named peak_profile and has the Caglioti, mixing
        # and asymmetry Parameters.
        (
            PseudoVoigtProfile,
            ["U", "V", "W", "asym0", "asym1", "asym2", "eta0", "eta1"],
        ),
        # C2: A TCH profile is created.
        # Expected: It is named peak_profile and has the Gaussian,
        # Lorentzian and size broadening Parameters.
        (
            TCHProfile,
            ["LGmix", "P", "U", "V", "W", "X", "Y", "Z"],
        ),
    ],
)
def test_init(profile_class, expected_names):
    profile = profile_class()
    actual_name = profile.name
    actual_names = sorted(profile.get_names())
    expected_name = "peak_profile"
    assert actual_name == expected_name
    assert actual_names == expected_names


@pytest.mark.parametrize(
    "profile_class, initial_pyobjcryst_class, expected_pyobjcryst_class",
    [
        # Test that install replaces the installed pyobjcryst profile
        # C1: A pseudo-Voigt profile is installed where a TCH profile was.
        # Expected: The installed profile is a pseudo-Voigt profile.
        (
            PseudoVoigtProfile,
            ReflectionProfilePseudoVoigtTCH,
            ReflectionProfilePseudoVoigt,
        ),
        # C2: A TCH profile is installed where a pseudo-Voigt profile was.
        # Expected: The installed profile is a TCH profile.
        (
            TCHProfile,
            ReflectionProfilePseudoVoigt,
            ReflectionProfilePseudoVoigtTCH,
        ),
    ],
)
def test_install(
    pbso4_pattern,
    profile_class,
    initial_pyobjcryst_class,
    expected_pyobjcryst_class,
):
    pbso4_pattern.diffraction.SetProfile(initial_pyobjcryst_class())
    profile_class().install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pyobjcryst_class = type(pbso4_pattern.diffraction.GetProfile())
    assert actual_pyobjcryst_class == expected_pyobjcryst_class


@pytest.mark.parametrize(
    "profile_class, parameter_name, new_value",
    [
        # Test that installing a profile again is safe
        # C1: A pseudo-Voigt profile with a changed W is installed and
        # updated twice.
        # Expected: The pattern equals the one installed and updated once.
        (PseudoVoigtProfile, "W", 0.05),
        # C2: A TCH profile with a changed X is installed and updated twice.
        # Expected: The pattern equals the one installed and updated once.
        (TCHProfile, "X", 0.02),
    ],
)
def test_install_twice(
    pbso4_pattern,
    other_pbso4_pattern,
    profile_class,
    parameter_name,
    new_value,
):
    # Change a Parameter from its default, so the check isn't trivial
    profile = profile_class()
    profile.get(parameter_name).set_value(new_value)
    # Install and update the profile twice on the first pattern
    profile.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    profile.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    profile.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    profile.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # Install and update the same profile once on a second, fresh pattern
    profile.install(
        other_pbso4_pattern.pattern, other_pbso4_pattern.diffraction
    )
    profile.update(
        other_pbso4_pattern.pattern, other_pbso4_pattern.diffraction
    )
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


@pytest.mark.parametrize(
    "profile_class, pyobjcryst_class, parameter_name, pyobjcryst_par_name, "
    "new_value",
    [
        # Test that update pushes a changed Parameter into the installed
        # pyobjcryst profile
        # C1: The pseudo-Voigt W is changed.
        # Expected: The pattern equals one where W was set directly on an
        # installed pyobjcryst pseudo-Voigt profile.
        (PseudoVoigtProfile, ReflectionProfilePseudoVoigt, "W", "W", 0.05),
        # C2: The pseudo-Voigt eta0 is changed, whose name differs from its
        # RefinablePar name, Eta0.
        # Expected: The pattern equals one where Eta0 was set directly.
        (
            PseudoVoigtProfile,
            ReflectionProfilePseudoVoigt,
            "eta0",
            "Eta0",
            0.3,
        ),
        # C3: The TCH Lorentzian width X is changed.
        # Expected: The pattern equals one where X was set directly on an
        # installed pyobjcryst TCH profile.
        (TCHProfile, ReflectionProfilePseudoVoigtTCH, "X", "X", 0.02),
    ],
)
def test_update(
    pbso4_pattern,
    other_pbso4_pattern,
    profile_class,
    pyobjcryst_class,
    parameter_name,
    pyobjcryst_par_name,
    new_value,
):
    # Install the profile, change a Parameter and push it with update
    profile = profile_class()
    profile.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    profile.get(parameter_name).set_value(new_value)
    profile.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, set the same value directly in pyobjcryst
    other_pbso4_pattern.diffraction.SetProfile(pyobjcryst_class())
    installed_profile = other_pbso4_pattern.diffraction.GetProfile()
    installed_profile.GetPar(pyobjcryst_par_name).SetHumanValue(new_value)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


@pytest.mark.parametrize(
    "profile_class, new_values, expected_pyobjcryst_values",
    [
        # Test that update pushes every Parameter to its own RefinablePar
        # C1: Every pseudo-Voigt Parameter is set to a distinct value.
        # Expected: Each value is on the RefinablePar with the matching
        # ObjCryst name, so no two names are swapped.
        (
            PseudoVoigtProfile,
            {
                "U": 0.01,
                "V": -0.02,
                "W": 0.03,
                "eta0": 0.4,
                "eta1": 0.05,
                "asym0": 1.6,
                "asym1": 0.07,
                "asym2": 0.08,
            },
            {
                "U": 0.01,
                "V": -0.02,
                "W": 0.03,
                "Eta0": 0.4,
                "Eta1": 0.05,
                "Asym0": 1.6,
                "Asym1": 0.07,
                "Asym2": 0.08,
            },
        ),
        # C2: Every TCH Parameter is set to a distinct value.
        # Expected: Each value is on the RefinablePar with the same name, so
        # no two names are swapped.
        (
            TCHProfile,
            {
                "U": 0.01,
                "V": -0.02,
                "W": 0.03,
                "X": 0.04,
                "Y": -0.05,
                "Z": 0.06,
                "P": 0.07,
                "LGmix": 0.8,
            },
            {
                "U": 0.01,
                "V": -0.02,
                "W": 0.03,
                "X": 0.04,
                "Y": -0.05,
                "Z": 0.06,
                "P": 0.07,
                "LGmix": 0.8,
            },
        ),
    ],
)
def test_update_all_parameters(
    pbso4_pattern, profile_class, new_values, expected_pyobjcryst_values
):
    # Install the profile and give every Parameter its new value
    profile = profile_class()
    profile.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    for parameter_name, new_value in new_values.items():
        profile.get(parameter_name).set_value(new_value)
    profile.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    # Read every value back from the installed pyobjcryst profile
    installed_profile = pbso4_pattern.diffraction.GetProfile()
    actual_pyobjcryst_values = {}
    for pyobjcryst_par_name in expected_pyobjcryst_values:
        parameter = installed_profile.GetPar(pyobjcryst_par_name)
        actual_pyobjcryst_values[pyobjcryst_par_name] = (
            parameter.GetHumanValue()
        )
    assert actual_pyobjcryst_values == pytest.approx(
        expected_pyobjcryst_values
    )


def test_update_shared_profile(pbso4_pattern, other_pbso4_pattern):
    # Test one profile installed and updated on two patterns
    # Expected: A changed W reaches both installed pyobjcryst profiles,
    # because the profile keeps no reference to either pattern.
    # Install one profile on both patterns
    profile = PseudoVoigtProfile()
    profile.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    profile.install(
        other_pbso4_pattern.pattern, other_pbso4_pattern.diffraction
    )
    # Change W once and push it to both patterns with update
    profile.get("W").set_value(0.05)
    profile.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    profile.update(
        other_pbso4_pattern.pattern, other_pbso4_pattern.diffraction
    )
    # Read W back from each pattern's installed pyobjcryst profile
    actual_values = [
        pbso4_pattern.diffraction.GetProfile().GetPar("W").GetHumanValue(),
        other_pbso4_pattern.diffraction.GetProfile()
        .GetPar("W")
        .GetHumanValue(),
    ]
    expected_values = [0.05, 0.05]
    assert actual_values == pytest.approx(expected_values)
