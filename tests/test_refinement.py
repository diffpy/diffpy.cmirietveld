import numpy as np
import pytest
from conftest import PBSO4_CIF, TESTDATA_DIR
from scipy.optimize import least_squares

from diffpy.cmirietveld import (
    ChebyshevBackground,
    ConstantWavelengthInstrument,
    GSASDataParser,
    PowderDiffractionGenerator,
    PseudoVoigtProfile,
)
from diffpy.srfit.fitbase import (
    FitContribution,
    FitRecipe,
    FitResults,
    Profile,
)

RW_MARGIN = 0.01


class PbSO4XrayRefinement:
    """The PbSO4 lab x-ray refinement, set up as in the simple example.

    The data are 16 to 110 degrees of ``PBSO4.XRA`` with a Cu tube and
    ObjCryst's polarization rate 0.4, GSAS-II's polarization of 0.7. The
    background is fitted to the data first. Every variable is fixed and
    tagged by the stage that frees it.
    """

    def __init__(self):
        parser = GSASDataParser()
        parser.parse_file(TESTDATA_DIR / "PBSO4.XRA")
        profile = Profile()
        profile.load_parsed_data(parser)
        profile.set_calculation_range(xmin=16.0, xmax=110.0)
        self.background = ChebyshevBackground(
            "background", x_range=(16.0, 110.0)
        )
        self.background.fit_to_data(profile.x, profile.y)
        instrument = ConstantWavelengthInstrument()
        instrument.set_xray_source("Cu")
        instrument.set_polarization(0.4)
        self.generator = PowderDiffractionGenerator("bragg")
        self.generator.set_structure(PBSO4_CIF, name="PbSO4")
        self.generator.set_instrument(instrument)
        self.generator.set_peak_profile(PseudoVoigtProfile())
        contribution = FitContribution("pbso4")
        contribution.set_profile(profile, xname="tth")
        contribution.add_profile_generator(self.generator)
        contribution.register_calculator(self.background, argnames=["tth"])
        contribution.set_equation("bragg + background")
        self.recipe = FitRecipe()
        self.recipe.add_contribution(contribution)
        self.recipe.add_variable(self.generator.scale, 1e-5, tags=["scale"])
        for coefficient in self.background.coefficients:
            self.recipe.add_variable(coefficient, tags=["background"])
        phase = self.generator.phase
        for parameter in phase.sgpars.latpars:
            self.recipe.add_variable(parameter, tags=["cell"])
        self.recipe.add_variable(
            instrument.displacement, tags=["displacement"]
        )
        for name in ["U", "V", "W", "eta0", "asym0", "asym1"]:
            self.recipe.add_variable(
                self.generator.peak_profile.get(name), tags=["profile"]
            )
        for parameter in phase.sgpars.xyzpars:
            self.recipe.add_variable(parameter, tags=["atoms"])
        for scatterer in phase.get_scatterers():
            self.recipe.add_variable(
                scatterer.Biso,
                0.8,
                name=f"Biso_{scatterer.name}",
                tags=["atoms"],
            )
        self.recipe.fix("all")

    def refine_in_stages(self, stages):
        for tags in stages:
            self.recipe.free(*tags)
            least_squares(
                self.recipe.residual,
                self.recipe.get_values(),
                x_scale="jac",
                bounds=self.recipe.get_bounds_array(),
            )

    def get_rw(self):
        rw = FitResults(self.recipe).rw
        return rw

    def get_cell(self):
        cell = []
        for name in ["a", "b", "c"]:
            cell.append(self.generator.phase.get(name).value)
        return cell


@pytest.mark.parametrize(
    "stages, gsasii_rwp, gsasii_cell, cell_tolerance",
    [
        # Test that a staged PbSO4 x-ray refinement approaches GSAS-II's
        # reference refinement of the same data, stage by stage. Each case
        # refines from the start up to its stage. The peak shapes differ
        # from GSAS-II's, so Rw is only checked not to exceed GSAS-II's Rwp
        # by more than 0.01, and refined values change slightly from run to
        # run (ObjCryst sums |F| in pointer order), so later stages get
        # wider cell tolerances.
        # C1: A user refines the scale and the background.
        # Expected: Rw is at most GSAS-II's 0.458 plus 0.01, and the cell
        # is still the CIF's.
        (
            [["scale", "background"]],
            0.45812,
            [8.48000, 5.39800, 6.95800],
            1e-9,
        ),
        # C2: A user then refines the cell.
        # Expected: Rw is at most GSAS-II's 0.321 plus 0.01, and the cell
        # is within 0.001 angstroms of GSAS-II's.
        (
            [["scale", "background"], ["cell"]],
            0.32066,
            [8.48578, 5.40226, 6.96410],
            0.001,
        ),
        # C3: A user then refines the sample displacement.
        # Expected: Rw is at most GSAS-II's 0.274 plus 0.01, and the cell
        # is within 0.001 angstroms of GSAS-II's.
        (
            [["scale", "background"], ["cell"], ["displacement"]],
            0.27398,
            [8.48144, 5.39943, 6.96057],
            0.001,
        ),
        # C4: A user then refines the peak profile.
        # Expected: Rw is at most GSAS-II's 0.135 plus 0.01, and the cell
        # is within 0.003 angstroms of GSAS-II's.
        (
            [["scale", "background"], ["cell"], ["displacement"], ["profile"]],
            0.13462,
            [8.47997, 5.39854, 6.95982],
            0.003,
        ),
        # C5: A user then refines the atom positions and ADPs.
        # Expected: Rw is at most GSAS-II's 0.101 plus 0.01, and the cell
        # is within 0.003 angstroms of GSAS-II's.
        (
            [
                ["scale", "background"],
                ["cell"],
                ["displacement"],
                ["profile"],
                ["atoms"],
            ],
            0.10070,
            [8.48035, 5.39862, 6.96002],
            0.003,
        ),
    ],
)
def test_refine_in_stages(stages, gsasii_rwp, gsasii_cell, cell_tolerance):
    refinement = PbSO4XrayRefinement()
    refinement.refine_in_stages(stages)
    actual_rw = refinement.get_rw()
    expected_max_rw = gsasii_rwp + RW_MARGIN
    actual_cell = refinement.get_cell()
    expected_cell = gsasii_cell
    assert actual_rw <= expected_max_rw
    assert np.allclose(actual_cell, expected_cell, atol=cell_tolerance)
