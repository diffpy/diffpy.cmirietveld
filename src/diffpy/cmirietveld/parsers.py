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
"""Parsers of powder diffraction data files and instrument parameter
files.

A data parser loads a measured pattern into an srfit ``Profile`` with
``Profile.load_parsed_data``.

An instrument is calibrated once on a standard and then reused for every
sample measured on it. An instrument parameter parser reads such a
calibration and creates the components that carry it.
"""

__all__ = [
    "PowderDataParser",
    "GSASDataParser",
    "InstrumentParameterParser",
    "GSASInstrumentParser",
    "GSASIIInstrumentParser",
]

from pathlib import Path

import numpy as np

from diffpy.srfit.fitbase import ProfileParser


class PowderDataParser(ProfileParser):
    """Base class of the parsers of powder diffraction data.

    Subclasses read one format by overriding ``_parse_metadata`` and
    ``_parse_data``. The metadata hold the type of the x axis under
    ``"xtype"``, for example ``"tth"`` for 2theta in degrees.

    Uncertainties given in a file are read as they are. srfit gives a
    point whose uncertainty is zero, negative, infinite or not a number
    no weight in a fit.

    When a file has no uncertainties, they come from counting
    statistics: the square root of the counts, divided by the number of
    counters for counts averaged over several counters. They are floored
    at 1, so that a point with zero counts keeps its weight. Negative
    counts, for example from background-subtracted data, get the square
    root of their absolute value, so that a negative point is weighted
    like a positive one of the same size. For data that aren't raw
    counts, such as normalized synchrotron data, these uncertainties are
    only approximate, and known ones can be set with
    ``Profile.set_observed_profile``.
    """

    def _compute_counting_uncertainty(self, counts, n_counters=1):
        """Return the counting uncertainty of counts averaged over
        ``n_counters`` counters, floored at 1."""
        uncertainty = np.maximum(np.sqrt(np.abs(counts) / n_counters), 1.0)
        return uncertainty


class GSASDataParser(PowderDataParser):
    """Parser of GSAS powder data files.

    The first line of the file is its title, and lines before the first
    ``BANK`` line, such as comments, are skipped. Each bank starts with a
    line ``BANK <number> <points> <records> CONST <start> <step> 0 0
    <format>``, with the start and step in centidegrees, and is followed
    by its data in fields of eight characters, in one of two formats:

    - ``STD``: the counts, ten to a line. Each field holds the number of
      counters averaged (two characters, blank for one) and the counts
      averaged over them (six characters), which are the counts read.
      The uncertainties come from counting statistics.
    - ``ESD``: the counts and their uncertainty, five points to a line,
      read as they are.

    The x axis is 2theta in degrees. Only constant-step STD and ESD banks
    are read, and a file can mix the two. Files with CRLF line endings
    are read like files with LF ones.

    Examples
    --------
    Load the PbSO4 x-ray data into a ``Profile``::

        parser = GSASDataParser()
        parser.parse_file(Path("PBSO4.XRA"))
        profile = Profile()
        profile.load_parsed_data(parser)
    """

    _format = "gsas"

    _data_formats = ("STD", "ESD", "FXYE", "FXY", "ALT")

    def _parse_metadata(self, filename):
        """Return the metadata of a GSAS powder data file."""
        metadata = {"xtype": "tth"}
        return metadata

    def _parse_data(self, filename, column_format=None, **kwargs):
        """Append the STD and ESD banks of a GSAS powder data file."""
        file_name = Path(filename).name
        lines = Path(filename).read_text().splitlines()
        bank_line_indices = []
        for line_index, line in enumerate(lines):
            if line.startswith("BANK"):
                bank_line_indices.append(line_index)
        if not bank_line_indices:
            raise ValueError(
                f"No BANK line was found in '{file_name}'. Please check that "
                "the file is a GSAS powder data file."
            )
        bank_line_indices.append(len(lines))
        for bank_index in range(len(bank_line_indices) - 1):
            bank_line_index = bank_line_indices[bank_index]
            next_bank_line_index = bank_line_indices[bank_index + 1]
            self._banks.append(
                self._read_bank(
                    file_name,
                    lines,
                    bank_line_index,
                    next_bank_line_index,
                )
            )

    def _read_bank(
        self, file_name, lines, bank_line_index, next_bank_line_index
    ):
        """Return the x, y, dx and dy of the bank whose BANK line is at
        ``bank_line_index``."""
        bank_fields = lines[bank_line_index].split()
        if len(bank_fields) < 7:
            raise ValueError(
                f"Line {bank_line_index + 1} of '{file_name}' is a BANK line "
                "without its binning, start and step. Please check that the "
                "file is a GSAS powder data file."
            )
        bank_number = bank_fields[1]
        point_count = int(bank_fields[2])
        binning = bank_fields[4]
        data_format = "STD"
        if bank_fields[-1] in self._data_formats:
            data_format = bank_fields[-1]
        if binning != "CONST":
            raise ValueError(
                f"Bank {bank_number} of '{file_name}' has {binning} binning, "
                "which can't be read. Please use a file with constant-step "
                "(CONST) binning."
            )
        if data_format not in ("STD", "ESD"):
            raise ValueError(
                f"Bank {bank_number} of '{file_name}' is in the {data_format} "
                "format, which can't be read. Please use a file in the STD "
                "or ESD format."
            )
        data_lines = []
        for line_index in range(bank_line_index + 1, next_bank_line_index):
            data_lines.append((line_index + 1, lines[line_index]))
        if data_format == "STD":
            counts, dy = self._read_std_points(file_name, data_lines)
        else:
            counts, dy = self._read_esd_points(file_name, data_lines)
        if len(counts) < point_count:
            raise ValueError(
                f"Bank {bank_number} of '{file_name}' has {len(counts)} "
                f"points, but its BANK line says it has {point_count}. "
                "Please check that the file is complete."
            )
        start = float(bank_fields[5]) / 100.0
        step = float(bank_fields[6]) / 100.0
        x = start + step * np.arange(point_count)
        bank = [x, counts[:point_count], None, dy[:point_count]]
        return bank

    def _read_std_points(self, file_name, data_lines):
        """Return the counts of STD lines and their counting
        uncertainties."""
        counts = []
        counter_numbers = []
        for line_number, line in data_lines:
            for field_start in range(0, len(line), 8):
                field = line[field_start : field_start + 8]
                if not field.strip():
                    continue
                counter_text = field[:2]
                counter_number = 1
                if counter_text.strip():
                    counter_number = self._read_number(
                        file_name, line_number, counter_text
                    )
                counter_numbers.append(counter_number)
                counts.append(
                    self._read_number(file_name, line_number, field[2:])
                )
        counts = np.array(counts)
        dy = self._compute_counting_uncertainty(
            counts, np.array(counter_numbers)
        )
        return counts, dy

    def _read_esd_points(self, file_name, data_lines):
        """Return the counts and uncertainties of ESD lines."""
        counts = []
        dy = []
        for line_number, line in data_lines:
            for field_start in range(0, len(line), 16):
                count_text = line[field_start : field_start + 8]
                uncertainty_text = line[field_start + 8 : field_start + 16]
                if not count_text.strip():
                    continue
                if not uncertainty_text.strip():
                    raise ValueError(
                        f"Line {line_number} of '{file_name}' has the count "
                        f"'{count_text.strip()}' without its uncertainty. "
                        "Please check that the file is in the ESD format."
                    )
                counts.append(
                    self._read_number(file_name, line_number, count_text)
                )
                dy.append(
                    self._read_number(file_name, line_number, uncertainty_text)
                )
        return np.array(counts), np.array(dy)

    @staticmethod
    def _read_number(file_name, line_number, text):
        """Return the number in ``text``, a field of the data."""
        try:
            number = float(text)
        except ValueError:
            raise ValueError(
                f"Line {line_number} of '{file_name}' has '{text.strip()}', "
                "which isn't a number. Please check that the file is a GSAS "
                "powder data file."
            ) from None
        return number


class InstrumentParameterParser:
    """Base class of the parsers of instrument parameter files.

    Subclasses read one format and convert its terms to the conventions
    of ObjCryst. A term with no ObjCryst counterpart gives a warning
    that names it.
    """

    def parse_file(self, path):
        """Parse an instrument parameter file.

        Parameters
        ----------
        path : str or Path
            The path to the instrument parameter file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not in the format of the parser.
        """
        raise NotImplementedError

    def get_instrument(self):
        """Return the instrument described by the file.

        Returns
        -------
        Instrument
            The instrument, with the wavelengths, zero and polarization
            of the file. The phases of a dataset share it.
        """
        raise NotImplementedError

    def get_peak_profile(self):
        """Return a new peak profile with the values of the file.

        Returns
        -------
        PeakProfile
            The peak profile. Each call returns a new object, so each
            phase gets its own, starting from the same values.
        """
        raise NotImplementedError


class GSASInstrumentParser(InstrumentParameterParser):
    """Parser of the GSAS instrument parameter file, ``.prm``."""


class GSASIIInstrumentParser(InstrumentParameterParser):
    """Parser of the GSAS-II instrument parameter file, ``.instprm``."""


# End of file
