"""Corey-model relative permeability formulas.

The UGH-1 deck's SGOF and SWOF tables were built from these formulas
(see the Dataset Documentation and the comments above each table in the
`.DATA` file), so tuning the endpoints or exponents Stage D.1 names
means regenerating the affected table's Krg/Krow or Krw/Krow columns
from these formulas rather than editing table rows by hand.

Every formula here has been checked against the anchor values printed
in the deck's SGOF/SWOF comments (Sgc=0.04, krg_max=0.77, ng=3.0,
Swc=0.13, Sorw=0.12, krw_max=0.78, nw=3.8, kro_max=0.80, no=4.0,
Sorg=0.15) and reproduces the shipped table to within rounding.
"""

import numpy as np
import numpy.typing as npt


def gas_relative_permeability(
    gas_saturation: npt.NDArray[np.float64],
    *,
    critical_gas_saturation: float,
    connate_water_saturation: float,
    residual_oil_saturation: float,
    max_gas_relative_permeability: float,
    gas_corey_exponent: float,
) -> npt.NDArray[np.float64]:
    """Corey gas relative permeability for an SGOF table.

    :param gas_saturation: Sg values to evaluate at, as fractions.
    :param critical_gas_saturation: Sgc, the gas saturation below which
        gas is immobile.
    :param connate_water_saturation: Swc, held fixed throughout an SGOF
        table (gas-oil relative permeability assumes water at connate).
    :param residual_oil_saturation: Sorg, the oil saturation below which
        oil stops flowing in the presence of gas.
    :param max_gas_relative_permeability: krg at Sg = 1 - Swc - Sorg.
    :param gas_corey_exponent: ng, the Corey exponent for the gas phase.
    :returns: krg at each requested Sg, zero below Sgc.
    """
    span = 1.0 - connate_water_saturation - critical_gas_saturation - residual_oil_saturation
    normalized = (gas_saturation - critical_gas_saturation) / span
    normalized = np.clip(normalized, 0.0, 1.0)
    krg = max_gas_relative_permeability * normalized**gas_corey_exponent
    return np.where(gas_saturation <= critical_gas_saturation, 0.0, krg)


def oil_relative_permeability_in_gas(
    gas_saturation: npt.NDArray[np.float64],
    *,
    critical_gas_saturation: float,
    connate_water_saturation: float,
    residual_oil_saturation: float,
    max_oil_relative_permeability: float,
    oil_corey_exponent: float,
) -> npt.NDArray[np.float64]:
    """Corey oil relative permeability (gas-oil system) for an SGOF table.

    :param gas_saturation: Sg values to evaluate at, as fractions.
    :param critical_gas_saturation: Sgc, used for the normalizing span.
    :param connate_water_saturation: Swc, held fixed throughout the table.
    :param residual_oil_saturation: Sorg, the saturation at which oil
        stops flowing and krog reaches zero.
    :param max_oil_relative_permeability: krog at Sg = 0.
    :param oil_corey_exponent: The oil-in-gas Corey exponent, `no`.
    :returns: krog at each requested Sg, zero once oil is at Sorg.
    """
    span = 1.0 - connate_water_saturation - critical_gas_saturation - residual_oil_saturation
    normalized = (1.0 - gas_saturation - connate_water_saturation - residual_oil_saturation) / span
    normalized = np.clip(normalized, 0.0, 1.0)
    return max_oil_relative_permeability * normalized**oil_corey_exponent


def water_relative_permeability(
    water_saturation: npt.NDArray[np.float64],
    *,
    connate_water_saturation: float,
    residual_oil_saturation: float,
    max_water_relative_permeability: float,
    water_corey_exponent: float,
) -> npt.NDArray[np.float64]:
    """Corey water relative permeability for an SWOF table.

    :param water_saturation: Sw values to evaluate at, as fractions.
    :param connate_water_saturation: Swc, the saturation below which
        water is immobile.
    :param residual_oil_saturation: Sorw, the residual oil saturation in
        the water-oil system.
    :param max_water_relative_permeability: krw at Sw = 1 - Sorw.
    :param water_corey_exponent: nw, the Corey exponent for the water phase.
    :returns: krw at each requested Sw, zero below Swc.
    """
    span = 1.0 - connate_water_saturation - residual_oil_saturation
    normalized = (water_saturation - connate_water_saturation) / span
    normalized = np.clip(normalized, 0.0, 1.0)
    krw = max_water_relative_permeability * normalized**water_corey_exponent
    return np.where(water_saturation <= connate_water_saturation, 0.0, krw)


def oil_relative_permeability_in_water(
    water_saturation: npt.NDArray[np.float64],
    *,
    connate_water_saturation: float,
    residual_oil_saturation: float,
    max_oil_relative_permeability: float,
    oil_corey_exponent: float,
) -> npt.NDArray[np.float64]:
    """Corey oil relative permeability (water-oil system) for an SWOF table.

    :param water_saturation: Sw values to evaluate at, as fractions.
    :param connate_water_saturation: Swc, used for the normalizing span.
    :param residual_oil_saturation: Sorw, the saturation at which krow
        reaches zero.
    :param max_oil_relative_permeability: krow at Sw = Swc.
    :param oil_corey_exponent: The water-oil Corey exponent, `no`.
    :returns: krow at each requested Sw, zero once oil is at Sorw.
    """
    span = 1.0 - connate_water_saturation - residual_oil_saturation
    normalized = (1.0 - water_saturation - residual_oil_saturation) / span
    normalized = np.clip(normalized, 0.0, 1.0)
    return max_oil_relative_permeability * normalized**oil_corey_exponent
