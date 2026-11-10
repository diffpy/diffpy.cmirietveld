"""Refine PbSO4 against lab x-ray data in stages.

This is the refinement of ``tests/test_refinement.py`` as a script. Run
it from the repository root:

    python refine_pbso4_xray.py
"""

from pathlib import Path

from scipy.optimize import least_squares

from diffpy.cmirietveld import (
    ChebyshevBackground,
    ConstantWavelengthInstrument,
    GSASDataParser,
    PowderDiffractionGenerator,
    PseudoVoigtProfile,
)
from diffpy.srfit import (
    FitContribution,
    FitRecipe,
    FitResults,
    Profile,
)

TESTDATA_DIR = Path(__file__).parents[2] / "tests" / "testdata"
DATA_FILE = TESTDATA_DIR / "PBSO4.XRA"
CIF_FILE = TESTDATA_DIR / "PbSO4-Wyckoff.cif"
STAGES = [
    ["scale", "background"],
    ["cell"],
    ["displacement"],
    ["profile"],
    ["atoms"],
]


def make_recipe(data_file, cif_file):
    """Make the recipe for a PbSO4 x-ray refinement.

    The data are 16 to 110 degrees of a GSAS data file, with a Cu tube
    and ObjCryst's polarization rate 0.4, GSAS-II's polarization of 0.7.
    The background is fitted to the data first. Every variable is fixed
    and tagged by the stage that frees it.

    Parameters
    ----------
    data_file : pathlib.Path
        The path to the GSAS data file.
    cif_file : pathlib.Path
        The path to the CIF file of the structure.

    Returns
    -------
    FitRecipe
        The recipe, with all its variables fixed.
    """
    parser = GSASDataParser()
    parser.parse_file(data_file)
    profile = Profile()
    profile.load_parsed_data(parser)
    profile.set_calculation_range(xmin=16.0, xmax=110.0)
    background = ChebyshevBackground("background", x_range=(16.0, 110.0))
    background.fit_to_data(profile.x, profile.y)
    background.plot_fit()
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_source("Cu")
    instrument.set_polarization(0.4)
    generator = PowderDiffractionGenerator("bragg")
    generator.set_structure(cif_file, name="PbSO4")
    generator.set_instrument(instrument)
    generator.set_peak_profile(PseudoVoigtProfile())
    contribution = FitContribution("pbso4")
    contribution.set_profile(profile, xname="tth")
    contribution.add_profile_generator(generator)
    contribution.register_calculator(background, argnames=["tth"])
    contribution.set_equation("bragg + background")
    recipe = FitRecipe()
    recipe.add_contribution(contribution)
    recipe.add_variable(generator.scale, 1e-5, tags=["scale"])
    for coefficient in background.coefficients:
        recipe.add_variable(coefficient, tags=["background"])
    phase = generator.phase
    for parameter in phase.sgpars.latpars:
        recipe.add_variable(parameter, tags=["cell"])
    recipe.add_variable(instrument.displacement, tags=["displacement"])
    for name in ["U", "V", "W", "eta0", "asym0", "asym1"]:
        recipe.add_variable(generator.peak_profile.get(name), tags=["profile"])
    for parameter in phase.sgpars.xyzpars:
        recipe.add_variable(parameter, tags=["atoms"])
    for scatterer in phase.get_scatterers():
        recipe.add_variable(
            scatterer.Biso, 0.8, name=f"Biso_{scatterer.name}", tags=["atoms"]
        )
    recipe.fix("all")
    return recipe


def refine_recipe(recipe, stages):
    """Refine the recipe in stages.

    Each stage frees the variables with its tags and refines them with
    the variables freed before.

    Parameters
    ----------
    recipe : FitRecipe
        The recipe, with all its variables fixed.
    stages : list of list of str
        The tags freed at each stage, in order.
    """
    for tags in stages:
        recipe.free(*tags)
        least_squares(
            recipe.residual,
            recipe.get_values(),
            x_scale="jac",
            bounds=recipe.get_bounds_array(),
        )


def main():
    recipe = make_recipe(DATA_FILE, CIF_FILE)
    refine_recipe(recipe, STAGES)
    FitResults(recipe).print_results_table()
    recipe.plot_recipe()


if __name__ == "__main__":
    main()
