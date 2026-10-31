Test data
=========

``PBSO4.XRA``, ``INST_XRY.PRM`` and ``PbSO4-Wyckoff.cif`` are copied (with trailing whitespace removed)
from the ``tests/testinp`` directory of GSAS-II
(https://github.com/AdvancedPhotonSource/GSAS-II).

GSAS-II is Copyright 2010, UChicago Argonne, LLC, Operator of Argonne National
Laboratory. All rights reserved. This product includes software produced by
UChicago Argonne, LLC under Contract No. DE-AC02-06CH11357 with the Department
of Energy.

``pbso4-esd.xra`` is ``PBSO4.XRA`` written in the GSAS ESD format, with the
square root of each count as its uncertainty, to three decimals. Each line
holds five (count, uncertainty) pairs of eight characters, and the last line
is padded with spaces to 80 characters, as GSAS writes it.
