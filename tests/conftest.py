import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from numpy.polynomial import chebyshev
from pyobjcryst.crystal import create_crystal_from_cif
from pyobjcryst.powderpattern import PowderPattern
from pyobjcryst.reflectionprofile import (
    ReflectionProfilePseudoVoigt,
    ReflectionProfilePseudoVoigtAnisotropic,
)

from diffpy.cmirietveld.backgrounds import Background, ChebyshevBackground
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
PBSO4_X_RANGE = (16.0, 110.0)
PBSO4_BACKGROUND_COEFFICIENTS = [150.0, -20.0, 8.0, -3.0, 1.0, 0.5]
TOP_EDGE_EXTENSION = 20.0

matplotlib.use("Agg")


class PbSO4Pattern:
    """A plain pyobjcryst pattern for PbSO4, on a 16-110 degree grid at
    1.5406 angstroms, with its crystal and diffraction component.

    ObjCryst leaves out reflections above the top of its grid, so the
    pattern is calculated on the grid extended 20 degrees upwards at the
    same step, wider than any peak in these tests, and only the points of
    ``tth`` are returned. This is the pattern with every reflection that
    reaches the grid, which the generator also calculates.
    """

    def __init__(self):
        self.crystal = create_crystal_from_cif(str(PBSO4_CIF))
        self.pattern = PowderPattern()
        self.pattern.SetWavelength(1.5406)
        self.set_tth(np.arange(16.0, 110.0, 0.025))
        self.diffraction = self.pattern.AddPowderPatternDiffraction(
            self.crystal
        )

    def set_tth(self, tth):
        self.tth = tth
        step = tth[-1] - tth[-2]
        n_extra_points = int(round(TOP_EDGE_EXTENSION / step))
        extra_tth = tth[-1] + step * np.arange(1, n_extra_points + 1)
        extended_tth = np.concatenate((tth, extra_tth))
        self.pattern.SetPowderPatternX(np.radians(extended_tth))
        self.pattern.SetPowderPatternObs(np.zeros_like(extended_tth))

    def calculate(self):
        self.pattern.Prepare()
        extended_pattern = np.array(self.pattern.GetPowderPatternCalc())
        calculated_pattern = extended_pattern[: len(self.tth)]
        return calculated_pattern


class SyntheticPattern:
    """A pattern made of a known Chebyshev background and sharp peaks.

    The background is ``PBSO4_BACKGROUND_COEFFICIENTS`` on
    ``PBSO4_X_RANGE``, and 21 Gaussian peaks of 0.08 degrees standard
    deviation stand between 20 and 106 degrees, so a background
    estimated from the pattern can be compared with the known one.
    """

    def __init__(self, x_min=16.0, x_max=110.0):
        self.x = np.arange(x_min, x_max, 0.025)
        xmin, xmax = PBSO4_X_RANGE
        scaled_x = 2.0 * (self.x - xmin) / (xmax - xmin) - 1.0
        self.background = chebyshev.chebval(
            scaled_x, PBSO4_BACKGROUND_COEFFICIENTS
        )
        peaks = np.zeros_like(self.x)
        for index, center in enumerate(np.arange(20.0, 108.0, 4.3)):
            height = 2000.0 / (1 + index % 3)
            peaks += height * np.exp(-0.5 * ((self.x - center) / 0.08) ** 2)
        self.y = self.background + peaks


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


class ExternalBackground(Background):
    """A background that stands for one written in another package.

    The package doesn't know this class, so a FitContribution that
    calculates with it shows that a new background needs only a subclass
    of ``Background``. It is a flat background at the height of its one
    Parameter, ``level``.
    """

    def __init__(self, name):
        Background.__init__(self, name)
        self._new_parameter("level", 0.0)

    def __call__(self, x):
        background = np.full_like(x, self.level.value)
        return background


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
def synthetic_pattern():
    synthetic_pattern = SyntheticPattern()
    return synthetic_pattern


@pytest.fixture
def pbso4_background():
    pbso4_background = ChebyshevBackground("background", x_range=PBSO4_X_RANGE)
    for coefficient, value in zip(
        pbso4_background.coefficients, PBSO4_BACKGROUND_COEFFICIENTS
    ):
        coefficient.set_value(value)
    return pbso4_background


@pytest.fixture
def pbso4_background_points(pbso4_background):
    x_points = np.linspace(16.0, 110.0, 15)
    pbso4_background_points = np.column_stack(
        (x_points, pbso4_background(x_points))
    )
    return pbso4_background_points


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
