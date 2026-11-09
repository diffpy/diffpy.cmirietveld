import json
from pathlib import Path

import numpy as np
import pytest
from pyobjcryst.crystal import create_crystal_from_cif
from pyobjcryst.powderpattern import PowderPattern
from pyobjcryst.reflectionprofile import (
    ReflectionProfilePseudoVoigt,
    ReflectionProfilePseudoVoigtAnisotropic,
)

from diffpy.cmirietveld.component import Component
from diffpy.cmirietveld.instruments import ConstantWavelengthInstrument
from diffpy.cmirietveld.peak_profiles import PeakProfile, PseudoVoigtProfile
from diffpy.cmirietveld.powderdiffractiongenerator import (
    PowderDiffractionGenerator,
)
from diffpy.srfit.fitbase import FitContribution, Profile

TESTDATA_DIR = Path(__file__).parent / "testdata"
PBSO4_CIF = TESTDATA_DIR / "PbSO4-Wyckoff.cif"
PSEUDO_VOIGT_PARAMETER_NAMES = {
    "U": "U",
    "V": "V",
    "W": "W",
    "eta0": "Eta0",
    "eta1": "Eta1",
    "asym0": "Asym0",
    "asym1": "Asym1",
    "asym2": "Asym2",
}


class PbSO4Pattern:
    """A plain pyobjcryst pattern for PbSO4, on a 16-110 degree grid at
    1.5406 angstroms, with its crystal and diffraction component."""

    def __init__(self):
        self.crystal = create_crystal_from_cif(str(PBSO4_CIF))
        self.tth = np.arange(16.0, 110.0, 0.025)
        self.pattern = PowderPattern()
        self.pattern.SetWavelength(1.5406)
        self.pattern.SetPowderPatternX(np.radians(self.tth))
        self.pattern.SetPowderPatternObs(np.zeros_like(self.tth))
        self.diffraction = self.pattern.AddPowderPatternDiffraction(
            self.crystal
        )

    def calculate(self):
        self.pattern.Prepare()
        calculated_pattern = np.array(self.pattern.GetPowderPatternCalc())
        return calculated_pattern


class ExternalPeakProfile(PeakProfile):
    """A peak profile that stands for one written in another package.

    The package doesn't know this class, so a generator that calculates
    with it shows that a new peak profile needs no change to the
    generator. It wraps pyobjcryst's anisotropic pseudo-Voigt, which the
    package doesn't have yet, and maps only two of its Parameters.
    """

    def __init__(self):
        PeakProfile.__init__(self)
        self.create_parameters_from(
            ReflectionProfilePseudoVoigtAnisotropic(),
            {"W": "W", "G_HH": "G_HH"},
        )

    def install(self, pattern, diffraction):
        diffraction.SetProfile(ReflectionProfilePseudoVoigtAnisotropic())

    def update(self, pattern, diffraction):
        self.push_parameters(diffraction.GetProfile())


class NotificationCounter:
    """An observer that counts the notifications it receives."""

    def __init__(self):
        self.count = 0

    def __call__(self, semaphores):
        self.count += 1


@pytest.fixture
def notification_counter():
    notification_counter = NotificationCounter()
    return notification_counter


@pytest.fixture
def pbso4_pattern():
    pbso4_pattern = PbSO4Pattern()
    return pbso4_pattern


@pytest.fixture
def other_pbso4_pattern():
    other_pbso4_pattern = PbSO4Pattern()
    return other_pbso4_pattern


@pytest.fixture
def pbso4_generator():
    pbso4_generator = PowderDiffractionGenerator()
    pbso4_generator.set_structure(create_crystal_from_cif(str(PBSO4_CIF)))
    instrument = ConstantWavelengthInstrument()
    instrument.set_wavelength(1.5406)
    pbso4_generator.set_instrument(instrument)
    pbso4_generator.set_peak_profile(PseudoVoigtProfile())
    return pbso4_generator


@pytest.fixture
def other_pbso4_generator():
    other_pbso4_generator = PowderDiffractionGenerator()
    other_pbso4_generator.set_structure(
        create_crystal_from_cif(str(PBSO4_CIF))
    )
    instrument = ConstantWavelengthInstrument()
    instrument.set_wavelength(1.5406)
    other_pbso4_generator.set_instrument(instrument)
    other_pbso4_generator.set_peak_profile(PseudoVoigtProfile())
    return other_pbso4_generator


@pytest.fixture
def pbso4_contribution(pbso4_generator, pbso4_pattern):
    profile = Profile()
    profile.set_observed_profile(
        pbso4_pattern.tth, np.zeros_like(pbso4_pattern.tth)
    )
    pbso4_contribution = FitContribution("xray")
    pbso4_contribution.set_profile(profile)
    pbso4_contribution.add_profile_generator(pbso4_generator)
    pbso4_contribution.set_equation("bragg")
    return pbso4_contribution


@pytest.fixture
def pseudo_voigt_component():
    component = Component("peak_profile")
    component.create_parameters_from(
        ReflectionProfilePseudoVoigt(), PSEUDO_VOIGT_PARAMETER_NAMES
    )
    return component


@pytest.fixture
def user_filesystem(tmp_path):
    base_dir = Path(tmp_path)
    home_dir = base_dir / "home_dir"
    home_dir.mkdir(parents=True, exist_ok=True)
    cwd_dir = base_dir / "cwd_dir"
    cwd_dir.mkdir(parents=True, exist_ok=True)

    home_config_data = {"username": "home_username", "email": "home@email.com"}
    with open(home_dir / "diffpyconfig.json", "w") as f:
        json.dump(home_config_data, f)

    yield tmp_path
