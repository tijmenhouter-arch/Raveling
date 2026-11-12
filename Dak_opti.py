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
ROOF_DEAD, CEILING_DEAD, DENSITY = 0.15, 0.15, 500.0
AREA_LOADS = {"Sneeuw": 0.56, "Regenwater": 1.0, "Onderhoud vlaklast": 1.0}
POINT_LOAD_N = 1500.0
CASES = ("Permanent", *AREA_LOADS)
SECTIONS = (
    (45, 145), (45, 170), (45, 195), (45, 220),
    (63, 145), (63, 170), (63, 195), (63, 220), (63, 245), (65, 175), (75, 175),
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
    header: tuple[int, int] = (63, 300)
    bearing_left: int = 120
    bearing_right: int = 120
    extra_column: bool = False
    column_x: float = 3025.0
    snow: float = 0.56
    rain: float = 1.0
    maintenance_area: float = 1.0
    maintenance_point: float = 1.5  # kN
    skylight_kg: float = 80.0
    skylight_inner_mm: int = 1200
    skylight_outer_mm: int = 1400


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
    beams: dict = field(default_factory=dict)
    order: list = field(default_factory=list)
    main_x: list = field(default_factory=list)
    ext_x: list = field(default_factory=list)
    fixed_x: list = field(default_factory=list)
    # Directe lasten op doorlopend ondersteunde muurregels, niet vrij overspannend.
    wall_loads: dict = field(default_factory=lambda: {case: 0.0 for case in CASES})


def section_text(section):
    return f"{section[0]} × {section[1]} mm"


def load_total(loads):
    return sum(v if kind == "P" else (b - a) * v for kind, a, b, v in loads)


def add_q(beam, case, value, a=0.0, b=None):
    b = beam.length if b is None else b
    if b > a and value:
        beam.loads[case].append(("q", float(a), float(b), float(value)))


def add_point(beam, case, value, x):
    if value:
        if not -1e-6 <= x <= beam.length + 1e-6:
            raise ValueError(f"Belasting buiten {beam.name}: {x:g} mm")
        beam.loads[case].append(("P", float(x), float(x), float(value)))


def regular_grid(anchors, maximum):
    """Vul verplichte hartlijnen aan: ieder hart-op-hartveld <= maximum."""
    result = [float(anchors[0])]
    for left, right in zip(anchors, anchors[1:]):
        n = max(1, math.ceil((right - left) / maximum))
        result.extend(left + (right - left) * i / n for i in range(1, n + 1))
    return result


def tributaries(xs, left, right):
    """Dakstroken tot de middellijn tussen twee opeenvolgende balkassen."""
    for i, x in enumerate(xs):
        a = left if i == 0 else (xs[i - 1] + x) / 2
        b = right if i == len(xs) - 1 else (x + xs[i + 1]) / 2
        yield x, a, b


def reactions(length, loads):
    total = load_total(loads)
    moment = sum(v * a if kind == "P" else v * (b-a) * (a+b)/2
                 for kind, a, b, v in loads)
    right = moment / length
    return total - right, right

def variable_loads(p):
    return {
        "Sneeuw": p.snow,
        "Regenwater": p.rain,
        "Onderhoud vlaklast": p.maintenance_area,
    }


def frame_line_weight(p):
    """Eigen gewicht opstandprofiel in N/mm, numeriek gelijk aan kN/m."""
    b, h = p.header
    return b * h * DENSITY * 9.81 / 1e9


def _simple_response(length, loads, sample_points=()):
    """Exacte momenten-extrema, numerieke evaluatie van Euler-Bernoulli u*EI.

    Macaulay-integratie van punt- en gedeeltelijk verdeelde lasten.
    u wordt met een parabool rond het grootste roosterpunt verfijnd.
    Bij goedkeuring wordt bovendien een kleine numerieke marge toegepast.
    """
    left, right = reactions(length, loads)
    knots = sorted({0.0, length, *(a for _, a, _, _ in loads),
                    *(b for _, _, b, _ in loads)})
    extrema = []
    for a, b in zip(knots, knots[1:]):
        middle = (a + b) / 2
        shear = left
        slope = 0.0
        for kind, start, end, value in loads:
            if kind == "P":
                shear -= value if start < middle else 0.0
            else:
                shear -= value * max(0.0, min(middle, end) - start)
                if start < middle < end:
                    slope += value
        if slope:
            root = middle + shear / slope
            if a < root < b:
                extrema.append(root)
    x = np.unique(np.r_[np.linspace(0.0, length, 161),knots, extrema, sample_points]  )

    def values(xx):
        xx = np.asarray(xx)
        moment = left * xx
        integral = left * xx**3 / 6
        integral_L = left * length**3 / 6
        for kind, a, b, value in loads:
            xa, xb = np.maximum(xx-a, 0), np.maximum(xx-b, 0)
            if kind == "P":
                moment = moment - value * xa
                integral = integral - value * xa**3 / 6
                integral_L -= value * (length-a)**3 / 6
            else:
                moment = moment - value * (xa**2-xb**2) / 2
                integral = integral - value * (xa**4-xb**4) / 24
                integral_L -= value * ((length-a)**4-(length-b)**4) / 24
        down = integral_L * xx / length - integral
        return moment, down

    moment, down = values(x)
    i = int(np.argmax(np.abs(down)))
    if 0 < i < len(x)-1:
        # Lokale coördinaten voorkomen slechte conditionering van de fit.
        sign = 1 if down[i] >= 0 else -1
        coeff = np.polyfit(x[i-1:i+2]-x[i], sign *down[i-1:i+2], 2)
        if coeff[0] < 0:
            x_peak = x[i] - coeff[1] / (2*coeff[0])
            if x[i-1] <= x_peak <= x[i+1]:
                u_peak = float(values([x_peak])[1][0])
                if abs(u_peak) > abs(down[i]):
                    x = np.r_[x, x_peak]
                    moment, down = values(x)
    return {"left": left, "right": right, "x": x,
            "M": moment, "uEI": down,
            "maxM": float(np.max(np.abs(moment))),
            "maxuEI": float(np.max(np.abs(down)))}

def displacement_ei(length, loads, x):
    """Doorbuiging × EI van de basisbalk op positie x."""
    left, _ = reactions(length, loads)
    integral_x = left * x**3 / 6
    integral_L = left * length**3 / 6

    for kind, a, b, value in loads:
        if kind == "P":
            integral_x -= value * max(x-a, 0)**3 / 6
            integral_L -= value * (length-a)**3 / 6
        else:
            integral_x -= value * (
                max(x-a, 0)**4 - max(x-b, 0)**4
            ) / 24
            integral_L -= value * (
                (length-a)**4 - (length-b)**4
            ) / 24

    return integral_L * x / length - integral_x


def response(length, loads, support=None):
    """Balk met eventueel één star verticaal tussensteunpunt."""
    if support is None:
        result = _simple_response(length, loads)
        result["middle"] = 0.0
        return result

    if not 0 < support < length:
        raise ValueError(
            "De kolom moet tussen de eindopleggingen van B1 staan."
        )

    flexibility = (
        support**2 * (length-support)**2 / (3*length)
    )
    middle = (
        displacement_ei(length, loads, support)
        / flexibility
    )

    corrected = [
        *loads,
        ("P", support, support, -middle),
    ]
    points = np.r_[
        np.linspace(0, support, 161),
        np.linspace(support, length, 161),
    ]

    result = _simple_response(length, corrected, points)
    result["middle"] = middle
    return result


def column_axis(p):
    # UI meet vanaf binnenzijde; rekenkern vanaf linker oplegas.
    if not p.extra_column:
        return None
    return p.column_x + p.bearing_left / 2


def field_loads(loads, left, right):
    """Variabele belasting alleen in één veld van B1."""
    result = []

    for kind, a, b, value in loads:
        if kind == "P":
            if left <= a <= right:
                result.append((kind, a, b, value))
        elif min(b, right) > max(a, left):
            result.append(
                (kind, max(a, left), min(b, right), value)
            )

    return result


@st.cache_data(show_spinner=False, max_entries=32)
def point_envelope(length, support):
    """Lokale envelop van één verplaatsbare eenheidspuntlast."""
    if support is None:
        return length / 4, length**3 / 48

    locations = np.unique(np.r_[
        np.linspace(0, support, 161),
        np.linspace(support, length, 161),
    ])

    max_moment = 0.0
    max_deflection = 0.0

    for x in locations:
        result = response(
            length,
            [("P", x, x, 1.0)],
            support,
        )
        max_moment = max(max_moment, result["maxM"])
        max_deflection = max(
            max_deflection, result["maxuEI"]
        )

    # Kleine toeslag vanwege bemonstering van de lastpositie.
    return max_moment * 1.001, max_deflection * 1.001

# Middelpunten uit de oorspronkelijke tekening blijven vast.
SKYLIGHT_CENTRES_X = (1240.0, 4240.0)
SKYLIGHT_CENTRE_Y = 1412.0

CURB_CLEARANCE_MM = 10.0
CURB_MIN_ABOVE_ROOF_MM = 150.0

# Aannames uitsluitend voor de hoogtetoets.
ROOF_COVER_MM = 5.0
INSULATION_MM = 100.0


def skylight_geometry(p):
    clear = float(p.skylight_inner_mm) + CURB_CLEARANCE_MM
    outer = float(p.skylight_outer_mm)
    frame_outer = clear + 2 * p.header[0]

    if p.skylight_inner_mm <= 0 or p.skylight_kg < 0:
        raise ValueError(
            "Binnenmaat moet positief zijn; "
            "gewicht mag niet negatief zijn."
        )

    if outer <= clear:
        raise ValueError(
            "De buitenmaat moet groter zijn "
            "dan de vrije opstandopening."
        )

    if frame_outer > outer:
        available = (outer - clear) / 2
        raise ValueError(
            f"Het houten opstandprofiel is te breed. "
            f"Maximaal {available:g} mm per zijde past "
            "binnen de glascontour, vóór isolatie en afwerking."
        )

    xs = tuple(
        cx - clear / 2 for cx in SKYLIGHT_CENTRES_X
    )
    outer_xs = tuple(
        cx - outer / 2 for cx in SKYLIGHT_CENTRES_X
    )
    y = SKYLIGHT_CENTRE_Y - clear / 2
    outer_y = SKYLIGHT_CENTRE_Y - outer / 2

    if (
        min(outer_xs) < 0
        or max(outer_xs) + outer > WIDTH
        or outer_y < 0
        or outer_y + outer > DEPTH
    ):
        raise ValueError(
            "De buitenmaat past niet binnen het hoofddeel."
        )

    if outer_xs[0] + outer > outer_xs[1]:
        raise ValueError(
            "De buitencontouren van de lichtstraten overlappen."
        )

    if (
        min(xs) - p.joist[0] / 2 <= 0
        or max(xs) + clear + p.joist[0] / 2 >= WIDTH
    ):
        raise ValueError(
            "De dwarsbalkprofielen passen niet "
            "naast de gekozen opening."
        )

    roof_build_up = ROOF_COVER_MM + INSULATION_MM + p.sheet
    above_roof = p.header[1] - roof_build_up

    return dict(
        size=clear,
        x=xs,
        y=y,
        outer=outer,
        outer_x=outer_xs,
        outer_y=outer_y,
        frame_outer=frame_outer,
        frame_axis=clear + p.header[0],
        roof_build_up=roof_build_up,
        above_roof=above_roof,
        minimum_height=(
            roof_build_up + CURB_MIN_ABOVE_ROOF_MM
        ),
        height_ok=(
            above_roof >= CURB_MIN_ABOVE_ROOF_MM
        ),
    )


def roof_strip(p, geometry, case, a, b, y0, y1, length):
    """Dakstrook als lijnlast, zonder dubbele glasbelasting.

    Permanent: daklagen buiten de constructieve opening.
    Variabel: dakvlak buiten de buitencontour van het glas.
    Bij korte balken wordt de dakprojectie naar de oplegassen gemapt.
    """
    permanent = case == "Permanent"

    size = (
        geometry["size"] if permanent
        else geometry["outer"]
    )
    xs = (
        geometry["x"] if permanent
        else geometry["outer_x"]
    )
    y = (
        geometry["y"] if permanent
        else geometry["outer_y"]
    )

    surface = (
        ROOF_DEAD
        + {18: 0.10, 22: 0.12}[p.sheet]
        + CEILING_DEAD
        if permanent
        else variable_loads(p)[case]
    )

    cuts = sorted({
        y0,
        y1,
        max(y0, min(y1, y)),
        max(y0, min(y1, y + size)),
    })

    projection = y1 - y0
    loads = []

    for start, end in zip(cuts, cuts[1:]):
        width = b - a

        if y < (start + end) / 2 < y + size:
            width -= sum(
                max(0.0, min(b, x + size) - max(a, x))
                for x in xs
            )

        q = (
            surface
            * max(0.0, width) / 1000
            * projection / length
        )

        if q:
            loads.append((
                "q",
                (start - y0) * length / projection,
                (end - y0) * length / projection,
                q,
            ))

    return loads

def build_model(p):
    m = Model()
    area_loads = variable_loads(p)
    frame_q = frame_line_weight(p)
    bw = p.joist[0]
    hw = p.header[0]
    left_offset, right_offset = p.bearing_left / 2, p.bearing_right / 2
    g = skylight_geometry(p)
    OPEN_SIZE, OPEN_X, OPEN_Y = g["size"], g["x"], g["y"]

    def member(name, group, section, length, receivers=(None, None)):
        beam = Beam(name, group, section, float(length), receivers)
        m.beams[name] = beam
        # Eigen gewicht in N/mm. Halfopleggingen buiten het vrije veld inbegrepen.
        add_q(beam, "Permanent", section[0]*section[1]*DENSITY*9.81/1e9)
        return beam

    B1 = member("B1", "Randbalk", p.edge, WIDTH+left_offset+right_offset)
    B2 = member("B2", "Randbalk", p.edge, EXT_WIDTH+left_offset+right_offset)
    B3 = member("B3", "Randbalk", p.edge, EXT_DEPTH+left_offset,
                (("B1", EXT_X+left_offset), None))

    fixed = sorted(x for start in OPEN_X for x in
                   (start-bw/2, start+OPEN_SIZE+bw/2))
    m.fixed_x = fixed
    m.main_x = regular_grid([0.0, *fixed, WIDTH], p.spacing)
    m.ext_x = regular_grid([EXT_X, WIDTH], p.spacing)
    lower_axis = OPEN_Y-hw/2
    upper_axis = OPEN_Y+OPEN_SIZE+hw/2
    if lower_axis <= 0 or upper_axis >= DEPTH:
        raise ValueError("Het gekozen ravelprofiel past niet rond de openingen.")

    # Eerst alle doorgaande assen; later de korte balkjes en wissels.
    full, short, headers, extension = [], [], [], []
    side_names = {}
    for index, (x, a, b) in enumerate(tributaries(m.main_x, 0, WIDTH)):
        opening = next((j for j, ox in enumerate(OPEN_X)
                        if ox < x < ox+OPEN_SIZE), None)
        if index in (0, len(m.main_x)-1):
            # Randstroken dragen direct naar doorlopend ondersteunde muurregel.
            for case in CASES:
                m.wall_loads[case] += load_total(
                    roof_strip(
                        p, g, case,
                        a, b, 0, DEPTH, DEPTH,
                    )
                )
            continue
        if opening is None:
            name = f"D{index:02}"
            beam = member(name, "Dwarsbalk", p.joist, DEPTH,
                          (None, ("B1", x+left_offset)))
            full.append(name)
            for j, ox in enumerate(OPEN_X):
                if abs(x-(ox-bw/2)) < 1e-5:
                    side_names[j, "L"] = name
                if abs(x-(ox+OPEN_SIZE+bw/2)) < 1e-5:
                    side_names[j, "R"] = name
            # Dakstrook minus de projectie van de twee vrije lichtstraatopeningen.
            for case in CASES:
                beam.loads[case].extend(
                    roof_strip(
                        p, g, case,
                        a, b, 0, DEPTH, DEPTH,
                    )
                )
        else:
            for suffix, length, header, y_transfer in [
                ("o", lower_axis, f"W{opening+1}o", None),
                ("b", DEPTH-upper_axis, f"W{opening+1}b", None),
            ]:
                ox = OPEN_X[opening]
                relative_x = x-(ox-bw/2)
                receivers = ((None, (header, relative_x)) if suffix == "o"
                             else ((header, relative_x), ("B1", x+left_offset)))
                beam = member(f"D{index:02}{suffix}", "Korte dwarsbalk", p.joist,
                              length, receivers)
                short.append(beam.name)
                # Hele dakprojectie (662 mm) aan de korte balk toekennen,
                y0, y1 = (
                    (0, OPEN_Y)
                    if suffix == "o"
                    else (OPEN_Y + OPEN_SIZE, DEPTH)
                )

                for case in CASES:
                    beam.loads[case].extend(
                        roof_strip(
                            p, g, case,
                            a, b, y0, y1, length,
                        )
                    )

    # De 25% per rand wordt als lijnlast op de bijbehorende rand gezet.
    for j, ox in enumerate(OPEN_X):
        for suffix, axis in [("o", lower_axis), ("b", upper_axis)]:
            name = f"W{j+1}{suffix}"
            beam = member(name, "Ravelbalk", p.header, OPEN_SIZE+bw,
                          ((side_names[j, "L"], axis), (side_names[j, "R"], axis)))
            headers.append(name)
            # Horizontale zijde van het opstandframe boven deze wissel.
            # Totale lengte inclusief beide hoeken: opening + 2 profielbreedten.
            # Het totale gewicht wordt geprojecteerd op de vrije opening.
            add_q(
                beam,
                "Permanent",
                frame_q * (OPEN_SIZE + 2 * hw) / OPEN_SIZE,
                bw / 2,
                bw / 2 + OPEN_SIZE,
            )
            for case in CASES:
                total = p.skylight_kg*9.81 if case == "Permanent" else area_loads[case]* (g["outer"] / 1000) ** 2 * 1000
                add_q(beam, case, total/4/OPEN_SIZE, bw/2, bw/2+OPEN_SIZE)
        for side in ("L", "R"):
            beam = m.beams[side_names[j, side]]
            # Verticale framezijde draagt langs haar lengte op de dwarsbalk.
            add_q(
                beam,
                "Permanent",
                frame_q,
                OPEN_Y,
                OPEN_Y + OPEN_SIZE,
            )
            for case in CASES:
                total = p.skylight_kg*9.81 if case == "Permanent" else area_loads[case]* (g["outer"] / 1000) ** 2 * 1000
                add_q(beam, case, total/4/OPEN_SIZE, OPEN_Y, OPEN_Y+OPEN_SIZE)

    for i, (x, a, b) in enumerate(tributaries(m.ext_x, EXT_X, WIDTH)):
        if i == len(m.ext_x)-1:
            for case in CASES:
                surface = ROOF_DEAD+{18: .10, 22: .12}[p.sheet]+CEILING_DEAD if case == "Permanent" else area_loads[case]
                m.wall_loads[case] += surface*(b-a)*EXT_DEPTH/1000
            continue
        if i == 0:
            beam = B3
        else:
            beam = member(f"O{i:02}", "Dwarsbalk opbouw", p.joist, EXT_DEPTH,
                          (("B1", x+left_offset), ("B2", x-EXT_X+left_offset)))
            extension.append(beam.name)
        for case in CASES:
            surface = ROOF_DEAD+{18: .10, 22: .12}[p.sheet]+CEILING_DEAD if case == "Permanent" else area_loads[case]
            add_q(beam, case, surface*(b-a)/1000, 0, EXT_DEPTH)

    # Doorlopend ondersteunde regels: eigen gewicht direct naar de muren.
    m.wall_loads["Permanent"] += p.edge[0]*p.edge[1]*DENSITY*9.81/1e9*(WIDTH+DEPTH+DEPTH+EXT_DEPTH)
    m.order = short+headers+full+extension+["B3", "B2", "B1"]
    if len(m.order) != len(set(m.order)) or set(m.order) != set(m.beams):
        raise ValueError("Interne fout in de volgorde van belastingafdracht.")
    return m


def analyze(p, keep_details=True):
    m = build_model(p)
    material = MATERIALS[p.strength]
    # Invoerlasten vastleggen vóór reactiekrachten worden toegevoegd.
    applied = {c: m.wall_loads[c]+sum(load_total(b.loads[c]) for b in m.beams.values()) for c in CASES}
    terminal = dict(m.wall_loads)
    rows = []
    for name in m.order:
        beam = m.beams[name]
        support = column_axis(p) if name == "B1" else None

        reference_length = (
            min(support, beam.length-support)
            if support is not None
            else beam.length
        )
        limit = PROFILE["limit_ratio"] * reference_length
        b, h = beam.section
        W, EI = b*h*h/6, material["E"]*b*h**3/12
        calculated = {}
        for case in CASES:
            calculated[case] = response(beam.length, beam.loads[case], support)
            terminal[case] += calculated[case]["middle"]
            for receiver, force in zip(beam.receivers,
                                       (calculated[case]["left"], calculated[case]["right"])):
                if receiver is None:
                    terminal[case] += force
                else:
                    add_point(m.beams[receiver[0]], case, force, receiver[1])
        G = calculated["Permanent"]
        records = []
        for case in CASES:
            # De verschillende responseroosters kunnen andere extrema bevatten.
            # Analyse van de gecombineerde lasten houdt de werkelijke posities vast.
            if case == "Permanent":
                M = PROFILE["gamma_g_only"]*G["maxM"]
                u = (1+PROFILE["kdef"])*G["maxuEI"]/EI
                kmod = PROFILE["kmod_g"]
            else:
                design_loads = [(kind,a,z,v*PROFILE["gamma_g"]) for kind,a,z,v in beam.loads["Permanent"]]
                design_loads += [(kind,a,z,v*PROFILE["gamma_q"]) for kind,a,z,v in beam.loads[case]]
                service_loads = [(kind,a,z,v*(1+PROFILE["kdef"])) for kind,a,z,v in beam.loads["Permanent"]]
                service_loads += [(kind,a,z,v*(1+PROFILE["kdef"]*PROFILE["psi2"])) for kind,a,z,v in beam.loads[case]]
                M = response(beam.length, design_loads, support)["maxM"]
                u = response(beam.length, service_loads, support)["maxuEI"]/EI
                kmod = PROFILE["kmod_q"]
            strength = kmod*material["fm"]/PROFILE["gamma_m"]
            records.append(dict(case=case, moment=M/1e6, deflection=u,
                                ugt=M/W/strength,
                                bgt=u/limit))
            if support is not None and case != "Permanent":
                fields = [
                    ("linkerveld", 0, support),
                    ("rechterveld", support, beam.length),
                ]

                for suffix, start, end in fields:
                    partial = field_loads(
                        beam.loads[case], start, end
                    )

                    design = [
                        (k, a, z, v * PROFILE["gamma_g"])
                        for k, a, z, v
                        in beam.loads["Permanent"]
                    ]
                    design += [
                        (k, a, z, v * PROFILE["gamma_q"])
                        for k, a, z, v in partial
                    ]

                    service = [
                        (k, a, z, v * (1 + PROFILE["kdef"]))
                        for k, a, z, v
                        in beam.loads["Permanent"]
                    ]
                    service += [
                        (
                            k, a, z,
                            v * (
                                1 + PROFILE["kdef"]
                                * PROFILE["psi2"]
                            )
                        )
                        for k, a, z, v in partial
                    ]

                    moment = response(
                        beam.length, design, support
                    )["maxM"]
                    deflection = response(
                        beam.length, service, support
                    )["maxuEI"] / EI

                    records.append(dict(
                        case=f"{case} · {suffix}",
                        moment=moment / 1e6,
                        deflection=deflection,
                        ugt=moment / W / strength,
                        bgt=deflection / limit,
                    ))
        # Eén puntlastenvelop per los onderdeel. Dit zijn bovenbegrenzingen:
        # max(G+P) <= max(G)+max(P), zonder te doen alsof pieken samenvallen.
        strength = PROFILE["kmod_q"]*material["fm"]/PROFILE["gamma_m"]
        point_M, point_uEI = point_envelope(
            beam.length, support
        )

        M = (
            PROFILE["gamma_g"] * G["maxM"]
            + PROFILE["gamma_q"] * (p.maintenance_point * 1000) * point_M
        )

        u = (
            (1 + PROFILE["kdef"]) * G["maxuEI"] / EI
            + (p.maintenance_point * 1000) * point_uEI / EI
        )
        records.append(dict(case="Onderhoud puntlast · envelop", moment=M/1e6,
                            deflection=u, ugt=M/W/strength,
                            bgt=u/limit))
        worst_u = max(records, key=lambda r:r["ugt"])
        worst_b = max(records, key=lambda r:r["bgt"])
        rows.append(dict(name=name, group=beam.group, section=section_text(beam.section),
                         length=beam.length, ugt=worst_u["ugt"], bgt=worst_b["bgt"],
                         ugt_case=worst_u["case"], bgt_case=worst_b["case"],
                         deflection=worst_b["deflection"],
                         limit=limit,
                         cases=records if keep_details else None))
    balance = {c:terminal[c]-applied[c] for c in CASES}
    if max(abs(v) for v in balance.values()) > 1e-5:
        raise ArithmeticError("De belastingafdracht is niet in evenwicht.")
    # Kleine marge voor de numerieke evaluatie van de doorbuiging.
    ugt = max(r["ugt"] for r in rows)
    bgt = max(r["bgt"] for r in rows)*(1+1e-4)
    volume = sum(b.section[0]*b.section[1]*b.length/1e9 for b in m.beams.values())
    volume += p.edge[0]*p.edge[1]*(WIDTH+2*DEPTH+EXT_DEPTH)/1e9
    geometry = skylight_geometry(p)

    volume += (len(SKYLIGHT_CENTRES_X) * (4 * geometry["size"] + 4 * p.header[0]) * p.header[0] * p.header[1] / 1e9)

    return dict(ugt=ugt,bgt=bgt, passed=(
            ugt <= 1
            and bgt <= 1
            and geometry["height_ok"]
        ),
        rows=rows, geometry=geometry, volume=volume, balance=balance, applied=applied, terminal=terminal, model=m if keep_details else None,)


@st.cache_data(show_spinner=False, max_entries=12)
def optimize(p):
    """Discrete zoekactie: dwarsbalkprofiel × maximale h.o.h.

    Randbalk- en ravelprofiel blijven zoals in de zijbalk gekozen. Alle in het
    model berekende balken moeten slagen voor de twee geïmplementeerde toetsen.
    """
    approved = []
    candidates = tuple(dict.fromkeys((*SECTIONS, p.joist)))

    for section in candidates:
        for spacing in SPACINGS:
            trial = replace(p, joist=section, spacing=spacing)
            result = analyze(trial, keep_details=False)
            if result["passed"]:
                approved.append(dict(section=section, spacing=spacing,
                                     ugt=result["ugt"], bgt=result["bgt"],
                                     volume=result["volume"]))
    return sorted(approved, key=lambda r:(r["volume"], r["ugt"], r["bgt"]))


# ==================== TEKENING ====================
def draw_plan(p, model):
    geometry = skylight_geometry(p)
    OPEN_SIZE, OPEN_X, OPEN_Y = (geometry["size"],geometry["x"],geometry["y"],)
    """Binnengeometrie leidend; randbalkdiktes schematisch aan de buitenzijde."""
    fig = Figure(figsize=(12, 8), layout="constrained", facecolor="white")
    ax = fig.subplots()
    orange, dark, pale = "#ec920f", "#9c570d", "#ffe2a8"
    ink, blue = "#172d49", "#c7eafa"
    ax.add_patch(Rectangle((0,0), WIDTH, DEPTH, color="#fff5dd", zorder=0))
    ax.add_patch(Rectangle((EXT_X,DEPTH), EXT_WIDTH, EXT_DEPTH, color="#fff5dd", zorder=0))

    def timber(x,y,w,h,color=pale,edge=dark,z=2):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor=edge,lw=.7,zorder=z))

    bw, hw, ew = p.joist[0], p.header[0], p.edge[0]
    for x in model.main_x[1:-1]:
        opening = next((ox for ox in OPEN_X if ox < x < ox+OPEN_SIZE), None)
        if opening is None:
            timber(x-bw/2,0,bw,DEPTH)
        else:
            timber(x-bw/2,0,bw,OPEN_Y-hw)
            timber(x-bw/2,OPEN_Y+OPEN_SIZE+hw,bw,DEPTH-OPEN_Y-OPEN_SIZE-hw)
    for x in model.ext_x[1:-1]:
        timber(x-bw/2,DEPTH,bw,EXT_DEPTH)
    for i,ox in enumerate(OPEN_X,1):
        timber(ox,OPEN_Y-hw,OPEN_SIZE,hw,orange,z=4)
        timber(ox,OPEN_Y+OPEN_SIZE,OPEN_SIZE,hw,orange,z=4)
        margin = (geometry["outer"] - OPEN_SIZE) / 2

        ax.add_patch(Rectangle((ox - margin, OPEN_Y - margin),geometry["outer"],geometry["outer"],facecolor=blue,edgecolor="#287cb0",lw=1.5,zorder=5,))
        glass = ax.patches[-1]
        for shift in (200,500):
            line, = ax.plot([ox-200+shift,ox+OPEN_SIZE+shift],
                            [OPEN_Y,OPEN_Y+OPEN_SIZE+200],color="white",alpha=.4,lw=18,zorder=6)
            line.set_clip_path(glass)
        ax.text(ox+OPEN_SIZE/2,OPEN_Y+OPEN_SIZE/2,
                f"Lichtstraat {i}\nBinnen: {p.skylight_inner_mm} × {p.skylight_inner_mm}\n{p.skylight_kg:g} kg",
                ha="center",va="center",color=ink,fontsize=10,fontweight="bold",zorder=7)
    # Apart opstandframe bovenop de bestaande balklaag.
    for ox in OPEN_X:
        timber(
            ox - hw, OPEN_Y - hw,
            OPEN_SIZE + 2 * hw, hw,
            orange, z=7,
        )
        timber(
            ox - hw, OPEN_Y + OPEN_SIZE,
            OPEN_SIZE + 2 * hw, hw,
            orange, z=7,
        )
        timber(
            ox - hw, OPEN_Y,
            hw, OPEN_SIZE,
            orange, z=7,
        )
        timber(
            ox + OPEN_SIZE, OPEN_Y,
            hw, OPEN_SIZE,
            orange, z=7,
        )

    ax.annotate(
        "Opstandframe\nbovenop balklaag",
        (
            OPEN_X[0] + OPEN_SIZE + hw / 2,
            OPEN_Y + OPEN_SIZE / 2,
        ),
        xytext=(2740, OPEN_Y + OPEN_SIZE / 2),
        ha="center",
        va="center",
        fontsize=9,
        color=ink,
        arrowprops=dict(
            arrowstyle="->",
            color=ink,
            lw=0.8,
        ),
        zorder=10,
    )
    # Randbalken buiten het schematische binnenvlak; alle balken één laag.
    timber(-ew,-ew,WIDTH+2*ew,ew,orange,z=8)
    timber(-ew,0,ew,DEPTH,orange,z=8)
    timber(WIDTH,0,ew,DEPTH+EXT_DEPTH+ew,orange,z=8)
    timber(0,DEPTH,WIDTH,ew,orange,z=8)
    timber(EXT_X-ew,DEPTH+ew,ew,EXT_DEPTH-ew,orange,z=8)
    timber(EXT_X,DEPTH+EXT_DEPTH,EXT_WIDTH,ew,orange,z=8)
    # Ondersteuningen: muur langs onder/links/rechts, kolom bij opbouw.
    ax.plot([-ew-40,-ew-40],[0,DEPTH],color=ink,lw=2,ls=":")
    ax.plot([0,WIDTH],[-ew-40,-ew-40],color=ink,lw=2,ls=":")
    ax.plot([WIDTH+ew+40,WIDTH+ew+40],[0,DEPTH+EXT_DEPTH],color=ink,lw=2,ls=":")
    ax.scatter([EXT_X-ew/2],[DEPTH+EXT_DEPTH],s=55,color=ink,zorder=9)
    ax.annotate("Kolom",(EXT_X-ew/2,DEPTH+EXT_DEPTH),xytext=(-12,14),
                textcoords="offset points",ha="right",fontsize=9,color=ink)
    ax.text(WIDTH*.38,DEPTH+ew+55,"B1",ha="center",fontsize=10,color=ink)
    ax.text(EXT_X+EXT_WIDTH*.5,DEPTH+EXT_DEPTH+ew+55,"B2",ha="center",fontsize=10,color=ink)
    ax.text(EXT_X-ew-65,DEPTH+EXT_DEPTH*.5,"B3",ha="right",fontsize=10,color=ink)

    def dim(a,b,offset,vertical=False,label=None,fs=10):
        if vertical:
            ax.plot([offset-65,offset+65],[a,a],color=ink,lw=.7)
            ax.plot([offset-65,offset+65],[b,b],color=ink,lw=.7)
            ax.annotate("",(offset,a),(offset,b),arrowprops=dict(arrowstyle="<->",lw=.8,color=ink))
            ax.text(offset-65,(a+b)/2,label or f"{b-a:g}",rotation=90,ha="right",va="center",fontsize=fs,color=ink)
        else:
            ax.plot([a,a],[offset-65,offset+65],color=ink,lw=.7)
            ax.plot([b,b],[offset-65,offset+65],color=ink,lw=.7)
            ax.annotate("",(a,offset),(b,offset),arrowprops=dict(arrowstyle="<->",lw=.8,color=ink))
            ax.text((a+b)/2,offset+55,label or f"{b-a:g}",ha="center",va="bottom",fontsize=fs,color=ink)

    chain = [
        0,
        OPEN_X[0],
        OPEN_X[0] + OPEN_SIZE,
        OPEN_X[1],
        OPEN_X[1] + OPEN_SIZE,
        WIDTH,
    ]
    for a,b in zip(chain,chain[1:]):
        dim(a,b,-290,fs=9)
    dim(0,WIDTH,-580,label="5.480 mm",fs=12)
    for a,b in [(0, OPEN_Y),(OPEN_Y, OPEN_Y + OPEN_SIZE),(OPEN_Y + OPEN_SIZE, DEPTH),]:
        dim(a,b,-300,True,fs=9)
    dim(0,DEPTH,-610,True,label="2.824 mm",fs=12)
    dim(EXT_X,WIDTH,DEPTH+EXT_DEPTH+360,label="2.455 mm",fs=12)
    dim(DEPTH,DEPTH+EXT_DEPTH,EXT_X-430,True,label="1.220 mm")
    dim(0,DEPTH+EXT_DEPTH,WIDTH+430,True,label="4.044 mm",fs=12)
    if p.extra_column:
        ax.scatter([p.column_x - ew / 2],[DEPTH + ew/2],s=110,color=ink,zorder=12,)
        ax.annotate("Extra kolom",(p.column_x - ew / 2, DEPTH + ew/2),xytext=(-14, -22),textcoords="offset points",ha="right",fontsize=9,color=ink,)
    ax.set_title("Raveling dak — plattegrond (bovenaanzicht)",loc="left",
                 fontsize=16,fontweight="bold",color=ink,pad=20)
    ax.text(0,-840,"Binnenmaten in mm · balkdiktes schematisch · opstandframes bovenop balklaag",
            fontsize=9,color=ink)
    ax.set(xlim=(-850,WIDTH+750),ylim=(-960,DEPTH+EXT_DEPTH+700),aspect="equal")
    ax.axis("off")
    return fig

def method_case(p, result, row, case):
    """Reconstrueer hetzelfde lastgeval als de controleberekening."""
    beam = result["model"].beams[row["name"]]
    support = column_axis(p) if beam.name == "B1" else None

    b, h = beam.section
    material = MATERIALS[p.strength]
    E, fm = material["E"], material["fm"]
    A, W, I = b * h, b * h**2 / 6, b * h**3 / 12

    G = beam.loads["Permanent"]
    point_case = case["case"].startswith("Onderhoud puntlast")
    only_g = case["case"] == "Permanent"
    Q = []

    if not point_case and not only_g:
        base = case["case"].split(" · ")[0]
        Q = beam.loads[base]

        if "linkerveld" in case["case"]:
            Q = field_loads(Q, 0, support)
        elif "rechterveld" in case["case"]:
            Q = field_loads(Q, support, beam.length)

    gamma_g = (
        PROFILE["gamma_g_only"]
        if only_g else PROFILE["gamma_g"]
    )
    kmod = (
        PROFILE["kmod_g"]
        if only_g else PROFILE["kmod_q"]
    )

    design = [
        (k, a, z, v * gamma_g)
        for k, a, z, v in G
    ]
    design += [
        (k, a, z, v * PROFILE["gamma_q"])
        for k, a, z, v in Q
    ]

    service = [
        (k, a, z, v * (1 + PROFILE["kdef"]))
        for k, a, z, v in G
    ]
    service += [
        (
            k, a, z,
            v * (1 + PROFILE["kdef"] * PROFILE["psi2"])
        )
        for k, a, z, v in Q
    ]

    g_response = response(beam.length, G, support)
    design_response = response(beam.length, design, support)
    service_response = response(beam.length, service, support)

    M = design_response["maxM"]
    u = service_response["maxuEI"] / (E * I)
    unit_M = unit_u = 0.0

    if point_case:
        unit_M, unit_u = point_envelope(
            beam.length, support
        )
        P = p.maintenance_point * 1000

        M = (
            gamma_g * g_response["maxM"]
            + PROFILE["gamma_q"] * P * unit_M
        )
        u = (
            (1 + PROFILE["kdef"])
            * g_response["maxuEI"] / (E * I)
            + P * unit_u / (E * I)
        )

    strength = kmod * fm / PROFILE["gamma_m"]

    return dict(
        beam=beam,
        support=support,
        A=A, W=W, I=I, E=E, fm=fm,
        G=G, Q=Q,
        gamma_g=gamma_g,
        kmod=kmod,
        strength=strength,
        point_case=point_case,
        only_g=only_g,
        M=M,
        u=u,
        g_response=g_response,
        design=design_response,
        service=service_response,
        unit_M=unit_M,
        unit_u=unit_u,
        ugt=M / W / strength,
        bgt=u / row["limit"],
    )

def draw_beam_minimap(p, model, selected):
    """Ligging uit hetzelfde balkenplan als de berekening."""
    geometry = skylight_geometry(p)
    OPEN_SIZE, OPEN_X, OPEN_Y = (geometry["size"],geometry["x"],geometry["y"],)
    def endpoints(name):
        if name == "B1":
            return (0, DEPTH), (WIDTH, DEPTH)
        if name == "B2":
            return (
                (EXT_X, DEPTH + EXT_DEPTH),
                (WIDTH, DEPTH + EXT_DEPTH),
            )
        if name == "B3":
            return (
                (EXT_X, DEPTH),
                (EXT_X, DEPTH + EXT_DEPTH),
            )
        if name.startswith("O"):
            x = model.ext_x[int(name[1:])]
            return (x, DEPTH), (x, DEPTH + EXT_DEPTH)

        if name.startswith("W"):
            ox = OPEN_X[int(name[1:-1]) - 1]
            y = (
                OPEN_Y - p.header[0] / 2
                if name.endswith("o")
                else OPEN_Y + OPEN_SIZE + p.header[0] / 2
            )
            return (
                (ox - p.joist[0] / 2, y),
                (ox + OPEN_SIZE + p.joist[0] / 2, y),
            )

        if name.startswith("D"):
            suffix = (
                name[-1] if name[-1] in ("o", "b") else ""
            )
            index = int(
                name[1:-1] if suffix else name[1:]
            )
            x = model.main_x[index]

            if suffix == "o":
                return (
                    (x, 0),
                    (x, OPEN_Y - p.header[0] / 2),
                )
            if suffix == "b":
                return (
                    (x, OPEN_Y + OPEN_SIZE + p.header[0] / 2),
                    (x, DEPTH),
                )
            return (x, 0), (x, DEPTH)

        raise ValueError(
            f"Geen tekenpositie bekend voor {name}"
        )

    fig = Figure(
        figsize=(5.4, 3.7),
        layout="constrained",
        facecolor="white",
    )
    ax = fig.subplots()

    for ox in OPEN_X:
        margin = (geometry["outer"] - OPEN_SIZE) / 2

        ax.add_patch(Rectangle(
            (ox - margin, OPEN_Y - margin),
            geometry["outer"],
            geometry["outer"],
            facecolor="#d5edf8",
            edgecolor="#86a6ba",
            lw=1,
        ))

    for name in model.beams:
        start, end = endpoints(name)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#b8b2a6",
            lw=1,
            zorder=2,
        )

    ax.plot(
        [0, WIDTH, WIDTH, EXT_X, EXT_X, 0, 0],
        [
            0, 0,
            DEPTH + EXT_DEPTH,
            DEPTH + EXT_DEPTH,
            DEPTH, DEPTH, 0,
        ],
        color="#6b624f",
        lw=1.5,
    )

    start, end = endpoints(selected)
    xs = [start[0], end[0]]
    ys = [start[1], end[1]]

    ax.plot(xs, ys, color="white", lw=7, zorder=5)
    ax.plot(xs, ys, color="#bd342c", lw=4, zorder=6)
    ax.scatter(xs, ys, color="#bd342c", s=16, zorder=7)

    if p.extra_column:
        ax.scatter(
            [p.column_x], [DEPTH],
            color="#172d49", s=24, zorder=8,
        )

    ax.scatter(
        [EXT_X], [DEPTH + EXT_DEPTH],
        color="#172d49", s=24, zorder=8,
    )
    ax.set_title(
        f"Ligging {selected} — rood gemarkeerd",
        loc="left",
        fontsize=11,
    )
    ax.set(
        xlim=(-160, WIDTH + 160),
        ylim=(-160, DEPTH + EXT_DEPTH + 160),
        aspect="equal",
    )
    ax.axis("off")
    return fig


def show_calculation_method(p, result):
    geometry = skylight_geometry(p)
    OPEN_SIZE, OPEN_X, OPEN_Y = (geometry["size"], geometry["x"], geometry["y"],)
    SKYLIGHT_KG = p.skylight_kg
    st.markdown("### 1. Rekenmodel en belastingafdracht")
    st.write(
        "De dakbelasting wordt verdeeld over de werkelijke "
        "balkposities. Naast iedere lichtstraat liggen vaste "
        "doorgaande dwarsbalken. Die behouden het dwarsbalkprofiel. "
        "Daarboven ligt een apart opstandframe van vier "
        "ravelprofielen per lichtstraat."
    )
    st.write(
        "De berekeningsvolgorde is: korte dwarsbalken → "
        "horizontale wissels → doorgaande dwarsbalken → "
        "randbalken. Reacties worden per lastgeval op hun "
        "werkelijke aansluitpositie doorgegeven. B3 draagt "
        "onderaan op B1; staat daar de extra kolom, dan vangt "
        "die deze reactie rechtstreeks op."
    )

    sheet_dead = {18: 0.10, 22: 0.12}[p.sheet]
    roof = ROOF_DEAD + sheet_dead + CEILING_DEAD

    loads_table = [
        {
            "Belasting": "Dakbedekking + isolatie",
            "Waarde": ROOF_DEAD,
            "Eenheid": "kN/m²",
        },
        {
            "Belasting": f"Underlayment {p.sheet} mm",
            "Waarde": sheet_dead,
            "Eenheid": "kN/m²",
        },
        {
            "Belasting": "Plafond inclusief regels",
            "Waarde": CEILING_DEAD,
            "Eenheid": "kN/m²",
        },
    ]
    loads_table += [
        {"Belasting": k, "Waarde": v, "Eenheid": "kN/m²"}
        for k, v in variable_loads(p).items()
    ]
    loads_table += [
        {
            "Belasting": "Onderhoudspuntlast",
            "Waarde": p.maintenance_point,
            "Eenheid": "kN",
        },
        {
            "Belasting": "Massa per lichtstraat",
            "Waarde": SKYLIGHT_KG,
            "Eenheid": "kg",
        },
    ]

    st.dataframe(
        pd.DataFrame(loads_table),
        hide_index=True,
        width="stretch",
    )
    st.latex(
        rf"g_{{dak}}={ROOF_DEAD:.2f}+{sheet_dead:.2f}"
        rf"+{CEILING_DEAD:.2f}={roof:.2f}"
        r"\;\mathrm{kN/m^2}"
    )
    st.latex(
        r"g_{balk}=\frac{b h}{10^6}\rho"
        r"\frac{9.81}{1000}\;[\mathrm{kN/m}]"
    )
    st.latex(
        r"g_{lijn}=g_{dak}s_{toegerekend}+g_{balk}"
    )
    st.caption(
        "De toegerekende dakstrook loopt tot halverwege "
        "de buurassen. Daklagen en plafond vervallen over "
        "de vrije lichtstraatopening. De korte balken krijgen "
        "de volledige bijbehorende dakprojectie toegerekend."
    )

    st.latex(
        rf"G_{{lichtstraat}}={SKYLIGHT_KG:g}\cdot9.81/1000"
        rf"={SKYLIGHT_KG * 9.81 / 1000:.4f}\;\mathrm{{kN}}"
    )
    st.latex(
        rf"G_{{per\ rand}}=G_{{lichtstraat}}/4"
        rf"={SKYLIGHT_KG * 9.81 / 4000:.4f}\;\mathrm{{kN}}"
    )
    st.write(
        "Iedere framezijde ontvangt 25% van de lichtstraatlast, "
        "gelijkmatig langs de openingrand. Ook de uniforme "
        "variabele lichtstraatbelasting wordt zo verdeeld."
    )

    frame_length = 4 * OPEN_SIZE + 4 * p.header[0]
    frame_weight = (
        frame_line_weight(p) * frame_length / 1000
    )
    st.write(
        f"Per opstandframe: twee zijden van {OPEN_SIZE:g} mm "
        f"en twee zijden van "
        f"{OPEN_SIZE + 2 * p.header[0]:g} mm. "
        f"Totale lengte: {frame_length:g} mm. "
        f"Eigen gewicht: **{frame_weight:.4f} kN per frame**."
    )
    st.write(
        "Dit framegewicht komt bovenop het lichtstraatgewicht. "
        "De verticale framezijden dragen op de doorgaande "
        "dwarsbalken; de horizontale zijden op de bestaande "
        "wissels. Het frame wordt rondom ondersteund "
        "verondersteld. Draagvlakken, excentriciteit en "
        "bevestigingen zijn nog niet gecontroleerd."
    )

    st.markdown("### Maatvoering en hoogte opstand")

    st.write(
        f"Binnenmaat raam: {p.skylight_inner_mm} mm. "
        f"Vrije opstandopening: {OPEN_SIZE:g} mm "
        "(10 mm montagetolerantie totaal). "
        f"Buitenmaat glas: {p.skylight_outer_mm} mm. "
        f"Buitenmaat kaal houten frame: "
        f"{geometry['frame_outer']:g} mm. "
        f"H.o.h. frame: {geometry['frame_axis']:g} mm."
    )

    st.write(
        "Permanent worden de daklagen buiten de "
        "constructieve opening gerekend, plus raam- en "
        "framegewicht. Variabele glasbelasting wordt "
        "over de volledige buitenmaat gerekend. "
        "De variabele dakbelasting wordt daar uitgezonderd, "
        "zodat geen oppervlak dubbel wordt belast."
    )

    st.latex(
        rf"h_{{vrij}}={p.header[1]}"
        rf"-{ROOF_COVER_MM:g}"
        rf"-{INSULATION_MM:g}"
        rf"-{p.sheet}"
        rf"={geometry['above_roof']:g}\;\mathrm{{mm}}"
    )

    st.write(
        "Vereist minimaal 150 mm boven afgewerkt dak. "
        f"Benodigde profielhoogte: "
        f"{geometry['minimum_height']:g} mm. "
        "Dakbedekking 5 mm en isolatie 100 mm zijn "
        "rekenaannames voor deze hoogtetoets. "
        "De permanente vlaklast blijft zoals ingevoerd. "
        "Helling blijft op verzoek buiten het model. "
        "Isolatie en afwerking in de opstandbreedte, "
        "steunlatten en bevestigingen zijn nog "
        "niet gedetailleerd."
    )
    st.markdown("### 2. Casus: de maatgevende balk")
    choice = st.selectbox(
        "Casus",
        ["Hoogste UGT", "Hoogste BGT"],
        key="method_choice",
    )
    key = "ugt" if choice == "Hoogste UGT" else "bgt"
    row = max(result["rows"], key=lambda r: r[key])
    case = max(row["cases"], key=lambda r: r[key])
    c = method_case(p, result, row, case)
    st.pyplot(
        draw_beam_minimap(
            p, result["model"], row["name"]
        ),
        width="content",
    )

    beam = c["beam"]
    L = beam.length
    b, h = beam.section

    st.write(
        f"**{beam.name} · {beam.group} · "
        f"{section_text(beam.section)} · {p.strength}**"
    )
    st.write(
        f"Maatgevend scenario: **{case['case']}**. "
        f"Afstand tussen de buitenste oplegassen: **{L:g} mm**."
    )

    if c["support"] is not None:
        st.write(
            f"Extra steunpunt op {c['support']:g} mm vanaf "
            f"de linker oplegas. Veldlengten: "
            f"{c['support']:g} en {L - c['support']:g} mm. "
            "B1 blijft doorgaand boven de kolom."
        )

    st.latex(
        rf"A=b h={b}\cdot{h}={c['A']:.0f}"
        r"\;\mathrm{mm^2}"
    )
    st.latex(
        rf"W=\frac{{b h^2}}{{6}}={c['W']:.2f}"
        rf"\;\mathrm{{mm^3}},\qquad "
        rf"I=\frac{{b h^3}}{{12}}={c['I']:.2f}"
        r"\;\mathrm{mm^4}"
    )
    st.latex(
        rf"E={c['E']:.0f}\;\mathrm{{N/mm^2}},\qquad "
        rf"EI={c['E'] * c['I']:.3e}"
        r"\;\mathrm{N\,mm^2}"
    )

    own = b * h * DENSITY * 9.81 / 1e9
    st.write(
        f"Eigen gewicht van deze balk: **{own:.5f} kN/m**, "
        f"met ρ = {DENSITY:g} kg/m³."
    )

    st.markdown("### 3. Werkelijke belastingen op deze balk")
    items = []
    for label, loads in [
        ("Permanent", c["G"]),
        ("Variabel gekozen scenario", c["Q"]),
    ]:
        for kind, a, z, v in loads:
            items.append({
                "Herkomst": label,
                "Type": "Puntlast" if kind == "P" else "Lijnlast",
                "Begin / positie [mm]": a,
                "Eind [mm]": z,
                "Waarde": v / 1000 if kind == "P" else v,
                "Eenheid": "kN" if kind == "P" else "kN/m",
            })

    st.dataframe(
        pd.DataFrame(items).round(5),
        hide_index=True,
        width="stretch",
    )
    st.write(
        "Permanente puntlasten zijn onder andere reacties "
        "uit andere balken. Hun aandeel in daklagen, "
        "lichtstraten en eigen gewicht zit daar al in; "
        "dit wordt niet opnieuw toegevoegd."
    )
    st.write(
        f"Totaal permanent op deze balk: "
        f"**{load_total(c['G']) / 1000:.4f} kN**."
    )

    if c["point_case"]:
        st.write(
            f"Variabele belasting: één onderhoudspuntlast "
            f"van **{p.maintenance_point:.4f} kN**. "
            "Deze wordt afzonderlijk als lokale envelop "
            "verwerkt, zonder gelijktijdige onderhoudsvlaklast."
        )
    else:
        st.write(
            f"Totaal variabel in dit scenario: "
            f"**{load_total(c['Q']) / 1000:.4f} kN**."
        )

    st.markdown("### 4. Oplegreacties en momenten")
    st.latex(
        r"R_R=\frac{\sum P_i a_i"
        r"+\sum q_j(b_j-a_j)(a_j+b_j)/2}{L},"
        r"\qquad R_L=\sum F-R_R"
    )
    st.caption(
        "Dit zijn de evenwichtsformules van de vrij opgelegde "
        "basisbalk. q is N/mm, P is N en posities zijn mm. "
        "1 kN/m = 1 N/mm."
    )

    if c["support"] is not None:
        st.latex(
            r"R_C=\frac{[EIu_0](c)}{c^2(L-c)^2/(3L)},"
            r"\qquad u(c)=0"
        )
        st.write(
            "De kolomreactie wordt als opwaartse kracht "
            "op de basisbalk gezet. Daarna worden reacties, "
            "momenten en vervormingen opnieuw bepaald. "
            "Ook het negatieve moment boven de kolom "
            "wordt meegenomen."
        )

    st.latex(
        r"M(x)=R_Lx-\sum_i P_i\langle x-a_i\rangle"
        r"-\sum_j\frac{q_j}{2}"
        r"\left(\langle x-a_j\rangle^2"
        r"-\langle x-b_j\rangle^2\right)"
    )
    st.caption(
        "⟨x−a⟩ = max(x−a, 0). De kolomreactie is, indien "
        "aanwezig, opgenomen als opwaartse puntkracht. "
        "Momentextrema worden bepaald bij lastgrenzen "
        "en waar de dwarskracht nul is."
    )

    if not c["point_case"]:
        r = c["design"]
        st.dataframe(
            pd.DataFrame({
                "Oplegging": ["Links", "Extra kolom", "Rechts"],
                "UGT-reactie [kN]": [
                    r["left"] / 1000,
                    r["middle"] / 1000,
                    r["right"] / 1000,
                ],
            }).round(4),
            hide_index=True,
            width="stretch",
        )
        if min(r["left"], r["middle"], r["right"]) < -1e-6:
            st.warning(
                "Dit scenario vraagt een trekreactie bij een "
                "oplegging. Verankering tegen oplichten "
                "is nog niet getoetst."
            )
    else:
        st.write(
            "De onderhoudspuntlast is een lokale envelop. "
            "De hoogste moment- en doorbuigingswaarde "
            "hoeven niet bij dezelfde lastpositie te horen. "
            "Daarom worden hiervoor geen fictieve "
            "gezamenlijke reacties of curves getoond."
        )

    st.markdown("### 5. UGT: buigsterkte")
    gamma_q = 0.0 if c["only_g"] else PROFILE["gamma_q"]
    st.latex(
        rf"E_d={c['gamma_g']:.2f}G_k"
        rf"+{gamma_q:.2f}Q_k"
    )

    if c["point_case"]:
        st.latex(
            r"M_{Ed,envelop}="
            r"\gamma_G\max|M_G|"
            r"+\gamma_QP_k\max|M_{P=1N}|"
        )
        st.write(
            f"Puntlast: {p.maintenance_point:.3f} kN. "
            f"Permanente momentpiek: "
            f"{c['g_response']['maxM'] / 1e6:.4f} kNm. "
            f"Eenheidspuntlast-envelop: "
            f"{c['unit_M']:.4f} Nmm/N."
        )

    st.latex(
        rf"M_{{Ed,max}}={c['M'] / 1e6:.4f}"
        rf"\;\mathrm{{kNm}},\qquad "
        rf"\sigma_{{m,d}}=\frac{{M_{{Ed,max}}}}{{W}}"
        rf"={c['M'] / c['W']:.4f}\;\mathrm{{N/mm^2}}"
    )
    st.latex(
        rf"f_{{m,d}}="
        rf"\frac{{k_{{mod}}f_{{m,k}}}}{{\gamma_M}}"
        rf"=\frac{{{c['kmod']:.2f}\cdot{c['fm']:.0f}}}"
        rf"{{{PROFILE['gamma_m']:.2f}}}"
        rf"={c['strength']:.4f}\;\mathrm{{N/mm^2}}"
    )
    sign = r"\leq" if c["ugt"] <= 1 else ">"
    st.latex(
        rf"\eta_{{UGT}}="
        rf"\frac{{\sigma_{{m,d}}}}{{f_{{m,d}}}}"
        rf"={c['ugt']:.4f}\quad {sign}\quad1.00"
    )

    st.markdown("### 6. BGT: directe vervorming en kruip")
    st.latex(r"EIu''(x)=-M(x)")
    st.write(
        "De momentfunctie wordt tweemaal geïntegreerd. "
        "De integratieconstanten volgen uit nulverplaatsing "
        "bij de eindopleggingen; bij de kolom geldt "
        "aanvullend nulverplaatsing."
    )
    st.write(
        "Bij een volledig verdeelde last of een puntlast "
        "midden op een vrij opgelegde balk geeft dit:"
    )
    st.latex(
        r"u_{max}=\frac{5qL^4}{384EI},"
        r"\qquad u_{P,midden}=\frac{PL^3}{48EI}"
    )
    st.write(
        "De werkelijke balk heeft gedeeltelijke lijnlasten "
        "en/of puntlasten. Daarom gebruikt de app de "
        "geïntegreerde lastfuncties en geen gemiddelde lijnlast."
    )
    st.latex(
        r"u_{fin}(x)=(1+k_{def})u_G(x)"
        r"+(1+k_{def}\psi_2)u_Q(x)"
    )
    st.write(
        f"kdef = {PROFILE['kdef']:.2f}; "
        f"ψ₂ = {PROFILE['psi2']:.2f}. "
        "De UGT-veiligheidsfactoren worden niet gebruikt "
    )

    if c["point_case"]:
        st.latex(
            r"u_{fin,envelop}="
            r"(1+k_{def})\max|u_G|"
            r"+P_k\max|u_{P=1N}|"
        )
    else:
        st.caption(
            "Eerst worden de vervormingslijnen gecombineerd; "
            "daarna wordt het maximum bepaald. De "
            "afzonderlijke maxima worden niet zonder "
            "meer opgeteld."
        )

    st.write(
        f"Einddoorbuiging voor dit scenario: "
        f"**{c['u']:.4f} mm**. "
        f"Grens: **{row['limit']:.4f} mm**."
    )
    st.latex(
        rf"u_{{lim}}=0.004L_{{ref}}"
        rf"={row['limit']:.4f}\;\mathrm{{mm}},"
        rf"\qquad\eta_{{BGT}}="
        rf"\frac{{u_{{fin}}}}{{u_{{lim}}}}"
        rf"={c['bgt']:.4f}"
    )
    st.caption(
        "Voor B1 met extra kolom geldt conservatief "
        "de kortste veldlengte als Lref voor de hele balk. "
        "De doorbuiging is ten opzichte van de eigen opleglijn."
    )

    if not c["point_case"]:
        fig = Figure(figsize=(9, 4.5), layout="constrained")
        axes = fig.subplots(2, 1)

        axes[0].plot(
            c["design"]["x"],
            c["design"]["M"] / 1e6,
            color="#b47016",
        )
        axes[0].set_ylabel("Moment UGT [kNm]")

        axes[1].plot(
            c["service"]["x"],
            c["service"]["uEI"] / (c["E"] * c["I"]),
            color="#167b83",
        )
        axes[1].set(
            ylabel="Einddoorbuiging [mm]",
            xlabel="Positie vanaf linker oplegas [mm]",
        )

        for ax in axes:
            ax.axhline(0, color="gray", lw=0.7)
            if c["support"] is not None:
                ax.axvline(
                    c["support"], color="gray", ls=":"
                )

        st.pyplot(fig, width="stretch")

    st.markdown("### 7. Alle scenario's voor deze balk")
    st.dataframe(
        pd.DataFrame(row["cases"]).rename(columns={
            "case": "Scenario",
            "moment": "Moment [kNm]",
            "deflection": "Einddoorbuiging [mm]",
            "ugt": "UGT",
            "bgt": "BGT",
        }).round(4),
        hide_index=True,
        width="stretch",
    )
    st.write(
        "UGT en BGT kunnen door verschillende scenario's "
        "worden bepaald. De hoofdresultaten nemen ieder "
        "afzonderlijk het maximum over alle balken "
        "en scenario's."
    )

    st.markdown("### 8. Rekenprofiel en beperkingen")
    st.json(PROFILE)
    st.write(
        "Grootste verschil tussen aangebrachte lasten "
        "en eindreacties: "
        f"{max(abs(v) for v in result['balance'].values()):.3e} N. "
        "De hoofd-BGT bevat 0,01% numerieke toeslag; "
        "de kolom-puntlastenvelop 0,1%."
    )
    #st.warning(
        #"Dit is een deelcontrole volgens het voorlopige "
       # "referentieprofiel. Normcombinaties en kmod/psi2 "
        #"per belastingtype zijn nog niet definitief. "
        #"Afschuiving, stabiliteit, verbindingen/verankering, "
        #"draagvlakken van het opstandframe, kolom/fundering, "
        #"aanvullende plafondcriteria en wateraccumulatie "
        #"zijn niet getoetst. Meevervorming van dragende "
        #"balken en verbindingsslip zijn niet opgeteld."
    #)


# ==================== EENVOUDIGE INTERFACE ====================
def choose_section(label, default, key):
    choice = st.selectbox(
        label,
        [*SECTIONS, "Handmatig"],
        index=(
            SECTIONS.index(default)
            if default in SECTIONS
            else len(SECTIONS)
        ),
        key=key,
    )

    if choice != "Handmatig":
        return choice

    # Handmatige maten behouden bij wisselen naar een standaardprofiel.
    previous = st.session_state.get(
        f"{key}_manual_values", default
    )

    left, right = st.columns(2)

    with left:
        width = st.number_input(
            "Breedte b [mm]",
            min_value=1,
            value=int(previous[0]),
            step=1,
            key=f"{key}_width",
        )

    with right:
        height = st.number_input(
            "Hoogte h [mm]",
            min_value=1,
            value=int(previous[1]),
            step=1,
            key=f"{key}_height",
        )

    section = (int(width), int(height))
    st.session_state[f"{key}_manual_values"] = section
    return section


def main():
    st.set_page_config(page_title="Optimalisatie raveling Schinkel",layout="wide")
    with st.sidebar:
        mode = st.select_slider("Rekenmodus",["Controle", "Optimalisatie"],value="Controle",key="mode",)
        st.header("Ontwerpparameters")
        strength=st.selectbox("Sterkteklasse vurenhout",list(MATERIALS),key="strength")
        sheet=st.selectbox("Underlayment",[18,22],format_func=lambda x:f"{x} mm",key="sheet")
        spacing=st.slider("Maximale h.o.h. dwarsbalken [mm]",300,750,600,25,key="spacing")
        joist=choose_section("Dwarsbalkprofiel",(65,175),"joist")
        edge=choose_section("Randbalkprofiel",(75,175),"edge")
        header=choose_section("Lichtstraat ravelprofiel",(63,300),"header")
        with st.expander("Lichtstraten · per stuk",expanded=True,):
            skylight_kg = st.number_input("Gewicht [kg]",min_value=0.0,value=80.0,step=1.0,key="skylight_kg",)
            skylight_inner_mm = st.number_input("Binnenmaat vierkant raam [mm]",min_value=100,value=1200,step=10,key="skylight_inner_mm",)
            skylight_outer_mm = st.number_input("Buitenmaat vierkant raam [mm]",min_value=100,value=1400,step=10, key="skylight_outer_mm",
            )

            st.caption(
                "Twee identieke ramen. Opstandopening = "
                "binnenmaat + 10 mm. Hoogtetoets: 5 mm "
                "dakbedekking + 100 mm isolatie + underlayment."
            )
        with st.expander("Variabele belastingen", expanded=False):
            snow = st.number_input("Sneeuw [kN/m²]", min_value=0.0, value=0.56, step=0.01, key="snow",)
            rain = st.number_input("Regenwater [kN/m²]", min_value=0.0,value=1.0,step=0.05,key="rain",)
            maintenance_area = st.number_input("Onderhoud vlaklast [kN/m²]",min_value=0.0,value=1.0,step=0.05,key="maintenance_area",)
            maintenance_point = st.number_input("Onderhoud puntlast [kN]",min_value=0.0,value=1.5,step=0.1,key="maintenance_point",)
        extra_column = st.toggle(
            "Extra kolom onder aansluiting B1–B3",
            value=False,
            key="extra_column",
        )

    p=Parameters(strength,sheet,spacing,joist,edge,header, extra_column=extra_column, snow=snow,rain=rain,maintenance_area=maintenance_area,maintenance_point=maintenance_point, skylight_kg=skylight_kg, skylight_inner_mm=skylight_inner_mm, skylight_outer_mm=skylight_outer_mm,)
    st.title("Optimalisatie raveling Schinkel")
    try:
        result=analyze(p)
    except (ValueError, ArithmeticError) as exc:
        st.error(str(exc))
        return
    plot_col, info_col = st.columns([3,1])
    with plot_col:
        st.pyplot(draw_plan(p,result["model"]),width="stretch")
    st.caption("Lichtbruin: dwarsbalken · oranje: randbalken en wissels · blauw: lichtstraten")
    a,b,c=st.columns([1,1,1.4])
    a.metric("UGT · buiging",f"{result['ugt']:.2f}")
    b.metric("BGT · einddoorbuiging",f"{result['bgt']:.2f}")
    status="VOLDOET" if result["passed"] else "VOLDOET NIET"
    (c.success if result["passed"] else c.error)(f"{status} — deelcontrole")
    st.caption("Een benutting ≤ 1,00 voldoet voor de berekende toets")
    worst_u=max(result["rows"],key=lambda r:r["ugt"])
    worst_b=max(result["rows"],key=lambda r:r["bgt"])
    st.caption(f"Bepalend UGT: {worst_u['name']} ({worst_u['ugt_case']}) · "
               f"Bepalend BGT: {worst_b['name']} ({worst_b['bgt_case']})")
    #st.info("Deelcontrole van buiging en einddoorbuiging per balk. Verbindingen, afschuiving, "
            #"stabiliteit en doorbuiging van de totale gekoppelde dakconstructie zijn nog niet getoetst.")
    geometry = result["geometry"]

    st.caption(
        "Vrije opstandhoogte boven afgewerkt dak: "
        f"{geometry['above_roof']:g} mm; "
        "vereist minimaal 150 mm."
    )

    if not geometry["height_ok"]:
        st.error(
            "Opstandprofiel te laag. "
            f"Minimaal {geometry['minimum_height']:g} mm "
            "profielhoogte nodig bij de aangenomen dakopbouw."
        )

    if mode=="Optimalisatie":
        st.subheader("Goedgekeurde combinaties · deelcontrole")
        st.caption("Zoekt dwarsbalkprofiel × maximale h.o.h.; randbalk- en ravelprofiel blijven zoals gekozen. "
                   "Alle berekende balken moeten voldoen. Gesorteerd op minimaal totaal houtvolume.")
        with st.spinner("Profielen en h.o.h.-afstanden controleren…"):
            approved=optimize(p)
        if not approved:
            st.warning("Geen combinatie voldoet binnen deze zoeklijst. Controleer ook het gekozen randbalk- en ravelprofiel.")
        else:
            table=pd.DataFrame([{"Dwarsbalkprofiel":section_text(r["section"]),"Max. h.o.h. [mm]":r["spacing"],
                                 "UGT":r["ugt"],"BGT":r["bgt"],"Houtvolume [m³]":r["volume"]}
                                for r in approved])
            st.dataframe(table,hide_index=True,width="stretch",
                         column_config={"UGT":st.column_config.NumberColumn(format="%.3f"),
                                        "BGT":st.column_config.NumberColumn(format="%.3f"),
                                        "Houtvolume [m³]":st.column_config.NumberColumn(format="%.3f")})
            st.caption("De tekening en hoofdresultaten hierboven horen bij de zijbalkinvoer. "
                       "Kies een combinatie uit de tabel in de zijbalk om die te bekijken.")

    with st.expander("Berekening en uitgangspunten"):
        show_calculation_method(p, result)
        details=pd.DataFrame([{"Balk":r["name"],"Type":r["group"],"Profiel":r["section"],
                               "Oplegasafstand [mm]":r["length"],"UGT":r["ugt"],"BGT":r["bgt"],
                               "Einddoorbuiging [mm]":r["deflection"],"Grens [mm]":r["limit"],
                               "Bepalend UGT":r["ugt_case"],"Bepalend BGT":r["bgt_case"]}
                              for r in result["rows"]])
        st.dataframe(details.round(3),hide_index=True,width="stretch")


if __name__=="__main__":
    main()