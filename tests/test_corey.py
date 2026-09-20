"""Corey formulas are checked against the real SGOF/SWOF table values
shipped in the deck (see the comments above each table in the `.DATA`
file for the anchor parameters used here), not just internal consistency.
"""

import numpy

from nagcsu import corey


def test_gas_relative_permeability_matches_shipped_sgof_table() -> None:
    gas_saturation = numpy.array([0.0000, 0.0480, 0.2400, 0.4800, 0.7200])
    expected_krg = numpy.array([0.00000, 0.00000, 0.01959, 0.20860, 0.77000])

    krg = corey.gas_relative_permeability(
        gas_saturation,
        critical_gas_saturation=0.04,
        connate_water_saturation=0.13,
        residual_oil_saturation=0.15,
        max_gas_relative_permeability=0.77,
        gas_corey_exponent=3.0,
    )

    numpy.testing.assert_allclose(krg, expected_krg, atol=1e-4)


def test_oil_relative_permeability_in_gas_matches_shipped_sgof_table() -> None:
    gas_saturation = numpy.array([0.0000, 0.2400, 0.7200])
    expected_krog = numpy.array([0.80000, 0.39862, 0.00000])

    krog = corey.oil_relative_permeability_in_gas(
        gas_saturation,
        critical_gas_saturation=0.04,
        connate_water_saturation=0.13,
        residual_oil_saturation=0.15,
        max_oil_relative_permeability=0.80,
        oil_corey_exponent=2.0,
    )

    numpy.testing.assert_allclose(krog, expected_krog, atol=1e-4)


def test_water_relative_permeability_matches_shipped_swof_table() -> None:
    water_saturation = numpy.array([0.1300, 0.4300, 0.8800])
    expected_krw = numpy.array([0.00000, 0.02398, 0.78000])

    krw = corey.water_relative_permeability(
        water_saturation,
        connate_water_saturation=0.13,
        residual_oil_saturation=0.12,
        max_water_relative_permeability=0.78,
        water_corey_exponent=3.8,
    )

    numpy.testing.assert_allclose(krw, expected_krw, atol=1e-4)


def test_oil_relative_permeability_in_water_matches_shipped_swof_table() -> None:
    water_saturation = numpy.array([0.1300, 0.4300, 0.8800])
    expected_krow = numpy.array([0.80000, 0.10368, 0.00000])

    krow = corey.oil_relative_permeability_in_water(
        water_saturation,
        connate_water_saturation=0.13,
        residual_oil_saturation=0.12,
        max_oil_relative_permeability=0.80,
        oil_corey_exponent=4.0,
    )

    numpy.testing.assert_allclose(krow, expected_krow, atol=1e-4)
