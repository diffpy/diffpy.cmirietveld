import re
import warnings

import numpy as np
import pytest
from conftest import TESTDATA_DIR

from diffpy.cmirietveld.parsers import GSASDataParser
from diffpy.srfit.fitbase import Profile


@pytest.mark.parametrize(
    "file_name",
    [
        # Test reading the PbSO4 x-ray data, the same measurement in two
        # GSAS formats
        # C1: A user loads the data as a GSAS STD file of raw counts.
        # Expected: One bank of 6001 points from 10 to 160 degrees in steps
        # of 0.025 degrees, the counts in the file, the square roots of the
        # counts as uncertainties, and a 2theta x axis. The zeros that pad
        # the last line of the file aren't read as points.
        "PBSO4.XRA",
        # C2: A user loads the same data as a GSAS ESD file, whose
        # uncertainties are the square roots of the counts, written to
        # three decimals.
        # Expected: The same grid, counts, uncertainties and x axis as the
        # STD file. The spaces that pad the last line of the file aren't
        # read as points.
        "pbso4-esd.xra",
    ],
)
def test_parse_file_with_pbso4_data(file_name):
    parser = GSASDataParser()
    parser.parse_file(TESTDATA_DIR / file_name)
    actual_x, actual_y, actual_dx, actual_dy = parser.get_data()
    actual_metadata = parser.get_metadata()
    actual_first_counts = list(actual_y[:10])
    actual_last_count = actual_y[-1]
    actual_xtype = actual_metadata["xtype"]
    actual_bank_count = actual_metadata["nbanks"]
    expected_x = 10.0 + 0.025 * np.arange(6001)
    expected_first_counts = [179, 147, 165, 172, 150, 165, 150, 158, 134, 146]
    expected_last_count = 368
    expected_dx = None
    expected_dy = np.sqrt(actual_y)
    expected_xtype = "tth"
    expected_bank_count = 1
    assert np.allclose(actual_x, expected_x)
    assert actual_first_counts == expected_first_counts
    assert actual_last_count == expected_last_count
    assert actual_dx == expected_dx
    # The ESD file writes the uncertainties to three decimals
    assert np.allclose(actual_dy, expected_dy, atol=5e-4)
    assert actual_xtype == expected_xtype
    assert actual_bank_count == expected_bank_count


@pytest.mark.parametrize(
    "file_text, expected_y, expected_dy",
    [
        # Test the counts and uncertainties of a file that has no
        # uncertainties, which come from counting statistics
        # C1: A user loads a file of raw counts.
        # Expected: The counts are those in the file, and the uncertainties
        # are their square roots, with no warning.
        (
            "Test pattern\n"
            "BANK 1 3 1 CONST 1000 2.5 0 0 STD\n"
            "     100     400     900\n",
            [100.0, 400.0, 900.0],
            [10.0, 20.0, 30.0],
        ),
        # C2: A user loads a file of counts averaged over four counters.
        # Expected: The counts are the averages in the file, not their
        # totals over the four counters, and the uncertainties are the
        # square roots of the counts over four, such as sqrt(100 / 4) = 5,
        # with no warning.
        (
            "Test pattern\n"
            "BANK 1 3 1 CONST 1000 2.5 0 0 STD\n"
            " 4   100 4   400 4   900\n",
            [100.0, 400.0, 900.0],
            [5.0, 10.0, 15.0],
        ),
        # C3: A user loads a file of normalized counts that aren't whole
        # numbers, such as synchrotron data.
        # Expected: The uncertainties are still the square roots of the
        # counts, with no warning.
        (
            "Test pattern\n"
            "BANK 1 2 1 CONST 1000 2.5 0 0 STD\n"
            "    6.25   30.25\n",
            [6.25, 30.25],
            [2.5, 5.5],
        ),
        # C4: A user loads a file with a point that has zero counts.
        # Expected: The uncertainty of that point is 1, so it keeps its
        # weight in a fit, with no warning.
        (
            "Test pattern\n"
            "BANK 1 3 1 CONST 1000 2.5 0 0 STD\n"
            "     100       0     900\n",
            [100.0, 0.0, 900.0],
            [10.0, 1.0, 30.0],
        ),
        # C5: A user loads a file of background-subtracted counts, some of
        # which are negative.
        # Expected: The uncertainty of a negative point is the square root
        # of its absolute value, as for a positive point of the same size,
        # with no warning.
        (
            "Test pattern\n"
            "BANK 1 3 1 CONST 1000 2.5 0 0 STD\n"
            "     100      -4     900\n",
            [100.0, -4.0, 900.0],
            [10.0, 2.0, 30.0],
        ),
    ],
)
def test_parse_file_without_uncertainties(
    tmp_path, file_text, expected_y, expected_dy
):
    # Write the file
    data_path = tmp_path / "pattern.gsa"
    data_path.write_text(file_text)
    parser = GSASDataParser()
    # Parse the file, recording every warning to check that none is raised
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        parser.parse_file(data_path)
    actual_warning_messages = []
    for caught_warning in caught_warnings:
        if issubclass(caught_warning.category, UserWarning):
            actual_warning_messages.append(str(caught_warning.message))
    actual_x, actual_y, actual_dx, actual_dy = parser.get_data()
    expected_warning_messages = []
    assert np.allclose(actual_y, expected_y)
    assert np.allclose(actual_dy, expected_dy)
    assert actual_warning_messages == expected_warning_messages


@pytest.mark.parametrize(
    "file_text, expected_x, expected_y, expected_dy",
    [
        # Test reading a file in the GSAS ESD format, which holds each
        # point's counts and uncertainty, five points to a line
        # C1: A user loads an ESD file.
        # Expected: The counts and uncertainties are those in the file,
        # with no warning.
        (
            "Test pattern\n"
            "BANK 1 3 1 CONST 1000 2.5 0 0 ESD\n"
            "     100      10     400      20     900      30\n",
            [10.0, 10.025, 10.05],
            [100.0, 400.0, 900.0],
            [10.0, 20.0, 30.0],
        ),
        # C2: A user loads an ESD file with more points than fit on one
        # line.
        # Expected: The points continue on the next line, on the same grid,
        # with no warning.
        (
            "Test pattern\n"
            "BANK 1 6 2 CONST 1000 2.5 0 0 ESD\n"
            "     100      10     400      20     900      30"
            "    1600      40    2500      50\n"
            "    3600      60\n",
            [10.0, 10.025, 10.05, 10.075, 10.1, 10.125],
            [100.0, 400.0, 900.0, 1600.0, 2500.0, 3600.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        ),
        # C3: A user loads an ESD file with an uncertainty of zero.
        # Expected: The uncertainty is read as zero, as it is in the file,
        # with no warning. srfit gives that point no weight in a fit.
        (
            "Test pattern\n"
            "BANK 1 3 1 CONST 1000 2.5 0 0 ESD\n"
            "     100      10     400       0     900      30\n",
            [10.0, 10.025, 10.05],
            [100.0, 400.0, 900.0],
            [10.0, 0.0, 30.0],
        ),
    ],
)
def test_parse_file_with_uncertainties(
    tmp_path, file_text, expected_x, expected_y, expected_dy
):
    # Write the file
    data_path = tmp_path / "pattern.gsa"
    data_path.write_text(file_text)
    parser = GSASDataParser()
    # Parse the file, recording every warning to check that none is raised
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        parser.parse_file(data_path)
    actual_warning_messages = []
    for caught_warning in caught_warnings:
        if issubclass(caught_warning.category, UserWarning):
            actual_warning_messages.append(str(caught_warning.message))
    actual_x, actual_y, actual_dx, actual_dy = parser.get_data()
    expected_warning_messages = []
    assert np.allclose(actual_x, expected_x)
    assert np.allclose(actual_y, expected_y)
    assert np.allclose(actual_dy, expected_dy)
    assert actual_warning_messages == expected_warning_messages


def test_parse_file_with_two_banks(tmp_path):
    # Test reading a file with an STD bank and an ESD bank, each with its
    # own grid
    # Expected: The parser has two banks, and each has the grid, counts and
    # uncertainties of its own BANK line and data: from counting statistics
    # for the STD bank, and from the file for the ESD bank.
    data_path = tmp_path / "pattern.gsa"
    data_path.write_text(
        "Test pattern\n"
        "BANK 1 2 1 CONST 1000 2.5 0 0 STD\n"
        "     100     400\n"
        "BANK 2 3 1 CONST 2000 5 0 0 ESD\n"
        "     900      15    1600      20    2500      25\n"
    )
    parser = GSASDataParser()
    parser.parse_file(data_path)
    actual_bank_count = parser.get_num_banks()
    actual_first_x, actual_first_y, _, actual_first_dy = parser.get_data(0)
    actual_second_x, actual_second_y, _, actual_second_dy = parser.get_data(1)
    expected_bank_count = 2
    expected_first_x = [10.0, 10.025]
    expected_first_y = [100.0, 400.0]
    expected_first_dy = [10.0, 20.0]
    expected_second_x = [20.0, 20.05, 20.1]
    expected_second_y = [900.0, 1600.0, 2500.0]
    expected_second_dy = [15.0, 20.0, 25.0]
    assert actual_bank_count == expected_bank_count
    assert np.allclose(actual_first_x, expected_first_x)
    assert np.allclose(actual_first_y, expected_first_y)
    assert np.allclose(actual_first_dy, expected_first_dy)
    assert np.allclose(actual_second_x, expected_second_x)
    assert np.allclose(actual_second_y, expected_second_y)
    assert np.allclose(actual_second_dy, expected_second_dy)


@pytest.mark.parametrize(
    "file_bytes",
    [
        # Test that the layouts of real files give the same data as a
        # plain file
        # C1: A user loads a file with LF line endings.
        # Expected: The grid and counts of its BANK line and data.
        (
            b"Test pattern\n"
            b"BANK 1 3 1 CONST 1000 2.5 0 0 STD\n"
            b"     100     400     900\n"
        ),
        # C2: A user loads a file with CRLF line endings, as written on
        # Windows.
        # Expected: The same grid and counts as with LF line endings.
        (
            b"Test pattern\r\n"
            b"BANK 1 3 1 CONST 1000 2.5 0 0 STD\r\n"
            b"     100     400     900\r\n"
        ),
        # C3: A user loads a file that mixes CRLF and LF line endings, as
        # NOMAD's files do.
        # Expected: The same grid and counts as with LF line endings.
        (
            b"Test pattern\r\n"
            b"BANK 1 3 1 CONST 1000 2.5 0 0 STD\n"
            b"     100     400     900\r\n"
        ),
        # C4: A user loads a file with more header lines before its BANK
        # line, such as the instrument parameter file and a comment.
        # Expected: The header lines are skipped, and the grid and counts
        # are the same as without them.
        (
            b"Test pattern\n"
            b"Instrument parameter file:INST_XRY.PRM\n"
            b"# Measured on the lab diffractometer\n"
            b"BANK 1 3 1 CONST 1000 2.5 0 0 STD\n"
            b"     100     400     900\n"
        ),
    ],
)
def test_parse_file_with_file_layouts(tmp_path, file_bytes):
    # Write the bytes as they are, so the line endings aren't changed
    data_path = tmp_path / "pattern.gsa"
    data_path.write_bytes(file_bytes)
    parser = GSASDataParser()
    parser.parse_file(data_path)
    actual_x, actual_y, _, actual_dy = parser.get_data()
    expected_x = [10.0, 10.025, 10.05]
    expected_y = [100.0, 400.0, 900.0]
    expected_dy = [10.0, 20.0, 30.0]
    assert np.allclose(actual_x, expected_x)
    assert np.allclose(actual_y, expected_y)
    assert np.allclose(actual_dy, expected_dy)


@pytest.mark.parametrize(
    "file_text, expected_msg",
    [
        # Test reading files that the parser can't read
        # C1: A user loads a file with no BANK line.
        # Expected: A ValueError says no BANK line was found.
        (
            "Test pattern\n" "     100     400     900\n",
            "No BANK line was found in 'pattern.gsa'. Please check that the "
            "file is a GSAS powder data file.",
        ),
        # C2: A user loads a file whose BANK line stops before its
        # binning, start and step.
        # Expected: A ValueError gives the line and says what is missing.
        (
            "Test pattern\n" "BANK 1 3 1\n" "     100     400     900\n",
            "Line 2 of 'pattern.gsa' is a BANK line without its binning, "
            "start and step. Please check that the file is a GSAS powder "
            "data file.",
        ),
        # C3: A user loads a file whose bank has RALF binning, not a
        # constant step.
        # Expected: A ValueError says the binning isn't supported.
        (
            "Test pattern\n"
            "BANK 1 3 1 RALF 1000 2.5 0 0 STD\n"
            "     100     400     900\n",
            "Bank 1 of 'pattern.gsa' has RALF binning, which can't be read. "
            "Please use a file with constant-step (CONST) binning.",
        ),
        # C4: A user loads a file in the GSAS FXYE format, which this
        # parser doesn't read yet.
        # Expected: A ValueError says the format isn't supported and names
        # the formats that are.
        (
            "Test pattern\n"
            "BANK 1 3 3 CONST 1000 2.5 0 0 FXYE\n"
            "  1000.0  100.0  10.0\n"
            "  1002.5  400.0  20.0\n"
            "  1005.0  900.0  30.0\n",
            "Bank 1 of 'pattern.gsa' is in the FXYE format, which can't be "
            "read. Please use a file in the STD or ESD format.",
        ),
        # C5: A user loads a file that was cut short, with fewer counts than
        # its BANK line says.
        # Expected: A ValueError gives both numbers.
        (
            "Test pattern\n"
            "BANK 1 3 1 CONST 1000 2.5 0 0 STD\n"
            "     100     400\n",
            "Bank 1 of 'pattern.gsa' has 2 points, but its BANK line says it "
            "has 3. Please check that the file is complete.",
        ),
        # C6: A user loads a file with a value that isn't a number, 1. a
        # count in an STD file, 2. an uncertainty in an ESD file.
        # Expected: A ValueError gives the line and the value.
        (
            "Test pattern\n"
            "BANK 1 3 1 CONST 1000 2.5 0 0 STD\n"
            "     100     1x0     900\n",
            "Line 3 of 'pattern.gsa' has '1x0', which isn't a number. Please "
            "check that the file is a GSAS powder data file.",
        ),
        (
            "Test pattern\n"
            "BANK 1 2 1 CONST 1000 2.5 0 0 ESD\n"
            "     100      1x     400      20\n",
            "Line 3 of 'pattern.gsa' has '1x', which isn't a number. Please "
            "check that the file is a GSAS powder data file.",
        ),
        # C7: A user loads an ESD file whose last count has no uncertainty
        # next to it.
        # Expected: A ValueError gives the line and the count.
        (
            "Test pattern\n"
            "BANK 1 2 1 CONST 1000 2.5 0 0 ESD\n"
            "     100      10     400\n",
            "Line 3 of 'pattern.gsa' has the count '400' without its "
            "uncertainty. Please check that the file is in the ESD format.",
        ),
    ],
)
def test_parse_file_with_malformed_file(tmp_path, file_text, expected_msg):
    data_path = tmp_path / "pattern.gsa"
    data_path.write_text(file_text)
    parser = GSASDataParser()
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        parser.parse_file(data_path)


def test_load_parsed_data_with_pbso4_data():
    # Test loading the parsed PbSO4 x-ray data into an srfit Profile, as a
    # refinement script does
    # Expected: The Profile holds the parser's grid, counts, uncertainties
    # and metadata.
    parser = GSASDataParser()
    parser.parse_file(TESTDATA_DIR / "PBSO4.XRA")
    profile = Profile()
    profile.load_parsed_data(parser)
    expected_x, expected_y, _, expected_dy = parser.get_data()
    actual_x = profile.x
    actual_y = profile.y
    actual_dy = profile.dy
    actual_xtype = profile.meta["xtype"]
    expected_xtype = "tth"
    assert np.array_equal(actual_x, expected_x)
    assert np.array_equal(actual_y, expected_y)
    assert np.array_equal(actual_dy, expected_dy)
    assert actual_xtype == expected_xtype
