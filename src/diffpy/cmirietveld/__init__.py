#!/usr/bin/env python
##############################################################################
#
# (c) 2026 diffpy.cmirietveld contributors.
# All rights reserved.
#
# File coded by: Caden Myers, members of the Billinge group, and
# contributors to the diffpy.cmirietveld project.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/diffpy/diffpy.cmirietveld/graphs/contributors
#
# See LICENSE.rst for license information.
#
##############################################################################
"""A pack for diffpy.cmi that allows it to do Rietveld refinement of
powder diffraction data."""

from diffpy.cmirietveld.backgrounds import Background, ChebyshevBackground
from diffpy.cmirietveld.component import Component
from diffpy.cmirietveld.corrections import (
    IntensityCorrection,
    MarchDollaseCorrection,
)
from diffpy.cmirietveld.instruments import (
    ConstantWavelengthInstrument,
    Instrument,
    TimeOfFlightInstrument,
)
from diffpy.cmirietveld.parsers import (
    GSASDataParser,
    GSASIIInstrumentParser,
    GSASInstrumentParser,
    InstrumentParameterParser,
    PowderDataParser,
)
from diffpy.cmirietveld.peak_profiles import (
    BackToBackExponentialProfile,
    PeakProfile,
    PseudoVoigtProfile,
    TCHProfile,
)
from diffpy.cmirietveld.powderdiffractiongenerator import (
    PowderDiffractionGenerator,
)
from diffpy.cmirietveld.sources import (
    IntensitySource,
    LeBailIntensities,
    StructureIntensities,
)

# package version
from diffpy.cmirietveld.version import __version__  # noqa

__all__ = [
    "PowderDiffractionGenerator",
    "Component",
    "IntensitySource",
    "StructureIntensities",
    "LeBailIntensities",
    "Instrument",
    "ConstantWavelengthInstrument",
    "TimeOfFlightInstrument",
    "IntensityCorrection",
    "MarchDollaseCorrection",
    "PeakProfile",
    "PseudoVoigtProfile",
    "TCHProfile",
    "BackToBackExponentialProfile",
    "Background",
    "ChebyshevBackground",
    "PowderDataParser",
    "GSASDataParser",
    "InstrumentParameterParser",
    "GSASInstrumentParser",
    "GSASIIInstrumentParser",
]

# silence the pyflakes syntax checker
assert __version__ or True

# End of file
