"""Optimalisatie raveling Schinkel — zelfstandig Streamlit-bestand.

Start: python -m streamlit run Dak_opti_simpel.py
Dependencies: streamlit, numpy, pandas, matplotlib, packaging.

Gebaseerd op de door gebruiker aangeleverde Dak_opti.py, 10 november 2026.
Eenheden rekenkern: mm, N, N/mm (= kN/m), N/mm².
Deelcontrole buiging en einddoorbuiging; GEEN volledige Eurocode-goedkeuring.

REKENAFSPRAKEN
- Eén balkenlaag. Vaste doorgaande dwarsbalken direct buiten de vrije openingen.
- Lichtstraatlast per lastgeval: 25% op ieder van de vier randen; per rand
  uniform verdeeld. Geen daklagen/plafond over de vrije lichtstraatopening.
- Elk onderdeel vrij opgelegd; reacties worden van kind naar ouder overgedragen.
- Doorbuiging t.o.v. de eigen opleglijn; verplaatsing van dragende balken en
  verbindingsslip worden niet opgeteld. De BGT-uitkomst is dus een deelcontrole.
- Onderhoudspuntlast: lokale conservatieve envelop van één volledige puntlast
  per los getoetst onderdeel, NIET gelijktijdig op alle balken. Maximaal moment
  en maximale doorbuiging van deze puntlast worden conservatief opgeteld bij
  de afzonderlijke maxima uit permanent. Geen dubbele afdracht van puntlasten.
- De gebruiker levert de factoren uit het referentievoorbeeld. Hun toepassing
  op regen/sneeuw/onderhoud en gevolgklasse/Nationale Bijlage staat nog open.
- Afschuiving, kip, druk loodrecht vezel, verbindingen, plafondcriteria,
  wind/sneeuwophoping en vervormingsafhankelijke wateraccumulatie ontbreken.

Bronnen voor verdere validatie:
https://eurocodes.jrc.ec.europa.eu/sites/default/files/2022-06/EN1995_2_Winter.pdf
https://www.swedishwood.com/siteassets/5-publikationer/pdfer/design-of-timber-structures-2-2016.pdf
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


# ==================== PROJECTUITGANGSPUNTEN ====================
# Bewust vaste uitgangspunten: de interface bevat alleen ontwerpparameters.
WIDTH, DEPTH = 5480.0, 2824.0
EXT_WIDTH, EXT_DEPTH = 2455.0, 1220.0
EXT_X = WIDTH - EXT_WIDTH
OPEN_SIZE, OPEN_Y = 1500.0, 662.0
OPEN_X = (490.0, 3490.0)
ROOF_DEAD, CEILING_DEAD, DENSITY = 0.15, 0.15, 500.0
SKYLIGHT_KG = 120.0  # inclusief constructie lichtstraat; opstand nog verifiëren
AREA_LOADS = {"Sneeuw": 0.56, "Regenwater": 1.0, "Onderhoud vlaklast": 1.0}
POINT_LOAD_N = 1500.0
CASES = ("Permanent", *AREA_LOADS)
SECTIONS = (
    (45, 145), (45, 170), (45, 195), (45, 220),
    (63, 145), (63, 170), (63, 195), (63, 220), (63, 245),
    (75, 195), (75, 220), (75, 245), (75, 270),
    (100, 220), (100, 245), (100, 270),
)
SPACINGS = (300, 350, 400, 450, 500, 550, 600, 650, 700, 750)
MATERIALS = {"C24": {"fm": 24.0, "E": 11000.0},
             "C18": {"fm": 18.0, "E": 9000.0}}
PROFILE = dict(gamma_g_only=1.20, gamma_g=1.10, gamma_q=1.35,
               gamma_m=1.30, kmod_g=0.60, kmod_q=0.90,
               kdef=0.60, psi2=0.0, limit_ratio=0.004)


@dataclass(frozen=True)
class Parameters:
    strength: str = "C24"
    sheet: int = 18
    spacing: int = 600
    joist: tuple[int, int] = (63, 195)
    edge: tuple[int, int] = (75, 245)
    header: tuple[int, int] = (63, 195)
    bearing_left: int = 150
    bearing_right: int = 150


# Belasting: ('q', begin, eind, N/mm) of ('P', positie, positie, N).
Load = tuple[str, float, float, float]


@dataclass
class Beam:
    name: str
    group: str
    section: tuple[int, int]
    length: float
    # Per uiteinde: None voor muur/kolom, anders (naam ontvangende balk, positie).
    receivers: tuple = (None, None)
    loads: dict = field(default_factory=lambda: {case: [] for case in CASES})


@dataclass
class Model:
