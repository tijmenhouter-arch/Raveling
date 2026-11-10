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
    extra_column: bool = False
    column_x: float = 3025.0


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

def build_model(p):
    m = Model()
    bw = p.joist[0]
    hw = p.header[0]
    left_offset, right_offset = p.bearing_left / 2, p.bearing_right / 2

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
                area_load = ROOF_DEAD + {18: .10, 22: .12}[p.sheet] + CEILING_DEAD if case == "Permanent" else AREA_LOADS[case]
                m.wall_loads[case] += area_load*(b-a)*DEPTH/1000
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
            removed = sum(max(0.0, min(b, ox+OPEN_SIZE)-max(a, ox)) for ox in OPEN_X)
            for case in CASES:
                surface = ROOF_DEAD+{18: .10, 22: .12}[p.sheet]+CEILING_DEAD if case == "Permanent" else AREA_LOADS[case]
                add_q(beam, case, surface*(b-a)/1000, 0, OPEN_Y)
                add_q(beam, case, surface*(b-a-removed)/1000, OPEN_Y, OPEN_Y+OPEN_SIZE)
                add_q(beam, case, surface*(b-a)/1000, OPEN_Y+OPEN_SIZE, DEPTH)
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
                # ook de halve wisselbreedte: geen verloren/dubbele dakstrook.
                projection = OPEN_Y if suffix == "o" else DEPTH-OPEN_Y-OPEN_SIZE
                for case in CASES:
                    surface = ROOF_DEAD+{18: .10, 22: .12}[p.sheet]+CEILING_DEAD if case == "Permanent" else AREA_LOADS[case]
                    add_q(beam, case, surface*(b-a)/1000*projection/length)

    # De 25% per rand wordt als lijnlast op de bijbehorende rand gezet.
    for j, ox in enumerate(OPEN_X):
        for suffix, axis in [("o", lower_axis), ("b", upper_axis)]:
            name = f"W{j+1}{suffix}"
            beam = member(name, "Ravelbalk", p.header, OPEN_SIZE+bw,
                          ((side_names[j, "L"], axis), (side_names[j, "R"], axis)))
            headers.append(name)
            for case in CASES:
                total = SKYLIGHT_KG*9.81 if case == "Permanent" else AREA_LOADS[case]*(OPEN_SIZE/1000)**2*1000
                add_q(beam, case, total/4/OPEN_SIZE, bw/2, bw/2+OPEN_SIZE)
        for side in ("L", "R"):
            beam = m.beams[side_names[j, side]]
            for case in CASES:
                total = SKYLIGHT_KG*9.81 if case == "Permanent" else AREA_LOADS[case]*(OPEN_SIZE/1000)**2*1000
                add_q(beam, case, total/4/OPEN_SIZE, OPEN_Y, OPEN_Y+OPEN_SIZE)

    for i, (x, a, b) in enumerate(tributaries(m.ext_x, EXT_X, WIDTH)):
        if i == len(m.ext_x)-1:
            for case in CASES:
                surface = ROOF_DEAD+{18: .10, 22: .12}[p.sheet]+CEILING_DEAD if case == "Permanent" else AREA_LOADS[case]
                m.wall_loads[case] += surface*(b-a)*EXT_DEPTH/1000
            continue
        if i == 0:
            beam = B3
        else:
            beam = member(f"O{i:02}", "Dwarsbalk opbouw", p.joist, EXT_DEPTH,
                          (("B1", x+left_offset), ("B2", x-EXT_X+left_offset)))
            extension.append(beam.name)
        for case in CASES:
            surface = ROOF_DEAD+{18: .10, 22: .12}[p.sheet]+CEILING_DEAD if case == "Permanent" else AREA_LOADS[case]
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
            + PROFILE["gamma_q"] * POINT_LOAD_N * point_M
        )

        u = (
            (1 + PROFILE["kdef"]) * G["maxuEI"] / EI
            + POINT_LOAD_N * point_uEI / EI
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
    return dict(ugt=ugt, bgt=bgt, passed=ugt<=1 and bgt<=1, rows=rows,
                volume=volume, balance=balance, applied=applied, terminal=terminal,
                model=m if keep_details else None)


@st.cache_data(show_spinner=False, max_entries=12)
def optimize(p):
    """Discrete zoekactie: dwarsbalkprofiel × maximale h.o.h.

    Randbalk- en ravelprofiel blijven zoals in de zijbalk gekozen. Alle in het
    model berekende balken moeten slagen voor de twee geïmplementeerde toetsen.
    """
    approved = []
    for section in SECTIONS:
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
        ax.add_patch(Rectangle((ox,OPEN_Y),OPEN_SIZE,OPEN_SIZE,
                               facecolor=blue,edgecolor="#287cb0",lw=1.5,zorder=5))
        # Rustige glasreflectie, geknipt op de lichtstraat.
        glass = ax.patches[-1]
        for shift in (200,500):
            line, = ax.plot([ox-200+shift,ox+OPEN_SIZE+shift],
                            [OPEN_Y,OPEN_Y+OPEN_SIZE+200],color="white",alpha=.4,lw=18,zorder=6)
            line.set_clip_path(glass)
        ax.text(ox+OPEN_SIZE/2,OPEN_Y+OPEN_SIZE/2,
                f"Lichtstraat {i}\n1.500 × 1.500 mm\n120 kg",
                ha="center",va="center",color=ink,fontsize=10,fontweight="bold",zorder=7)
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

    chain=[0,490,1990,3490,4990,5480]
    for a,b in zip(chain,chain[1:]):
        dim(a,b,-290,fs=9)
    dim(0,WIDTH,-580,label="5.480 mm",fs=12)
    for a,b in [(0,662),(662,2162),(2162,2824)]:
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
    ax.text(0,-840,"Binnenmaten in mm · balkdiktes schematisch · alle balken in één laag",
            fontsize=9,color=ink)
    ax.set(xlim=(-850,WIDTH+750),ylim=(-960,DEPTH+EXT_DEPTH+700),aspect="equal")
    ax.axis("off")
    return fig


# ==================== EENVOUDIGE INTERFACE ====================
def choose_section(label, default, key):
    return st.selectbox(label, SECTIONS, index=SECTIONS.index(default),
                        format_func=section_text, key=key)


def main():
    st.set_page_config(page_title="Optimalisatie raveling Schinkel",layout="wide")
    with st.sidebar:
        st.header("Ontwerpparameters")
        strength=st.selectbox("Sterkteklasse vurenhout",list(MATERIALS),key="strength")
        sheet=st.selectbox("Underlayment",[18,22],format_func=lambda x:f"{x} mm",key="sheet")
        spacing=st.slider("Maximale h.o.h. dwarsbalken [mm]",300,750,600,25,key="spacing")
        joist=choose_section("Dwarsbalkprofiel",(63,195),"joist")
        edge=choose_section("Randbalkprofiel",(75,245),"edge")
        header=choose_section("Lichtstraat ravelprofiel",(63,195),"header")
        with st.expander("Opleglengtes",expanded=False):
            bearing_left=st.number_input("Muur / kolom links [mm]",min_value=1,max_value=500,value=120,step=10,key="bearing_left")
            bearing_right=st.number_input("Muur rechts [mm]",min_value=1,max_value=500,value=120,step=10,key="bearing_right")
        st.caption("Alleen bij fysieke muur-/kolomopleggingen. Balkschoenen krijgen geen toeslag.")
        extra_column = st.toggle(
            "Extra kolom onder B1",
            value=False,
            key="extra_column",
        )

        column_x = st.number_input(
            "Kolom vanaf linker binnenzijde [mm]",
            min_value=1,
            max_value=5479,
            value=3025,
            step=10,
            disabled=not extra_column,
            key="column_x",
        )

        if extra_column:
            st.caption(
                "Op 3.025 mm ondersteunt deze kolom "
                "de aansluiting B1–B3."
            )
    p=Parameters(strength,sheet,spacing,joist,edge,header,bearing_left,bearing_right, extra_column, column_x)
    st.title("Optimalisatie raveling Schinkel")
    mode=st.select_slider("Rekenmodus",["Controle","Optimalisatie"],value="Controle",key="mode")
    try:
        result=analyze(p)
    except (ValueError, ArithmeticError) as exc:
        st.error(str(exc))
        return
    st.pyplot(draw_plan(p,result["model"]),width="stretch")
    st.caption("Lichtbruin: dwarsbalken · oranje: randbalken en wissels · blauw: lichtstraten")
    a,b,c=st.columns([1,1,1.4])
    a.metric("UGT · buiging",f"{result['ugt']:.2f}")
    b.metric("BGT · einddoorbuiging",f"{result['bgt']:.2f}")
    status="VOLDOET" if result["passed"] else "VOLDOET NIET"
    (c.success if result["passed"] else c.error)(f"{status} — deelcontrole")
    st.caption("Een benutting ≤ 1,00 voldoet voor de berekende toets. Waarden boven 1,00 blijven zichtbaar.")
    worst_u=max(result["rows"],key=lambda r:r["ugt"])
    worst_b=max(result["rows"],key=lambda r:r["bgt"])
    st.caption(f"Bepalend UGT: {worst_u['name']} ({worst_u['ugt_case']}) · "
               f"Bepalend BGT: {worst_b['name']} ({worst_b['bgt_case']})")
    st.info("Deelcontrole van buiging en einddoorbuiging per balk. Verbindingen, afschuiving, "
            "stabiliteit en doorbuiging van de totale gekoppelde dakconstructie zijn nog niet getoetst.")

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
        st.write(f"Vaste permanente daklast: {ROOF_DEAD+{18:.10,22:.12}[sheet]+CEILING_DEAD:.2f} kN/m² "
                 f"inclusief plafond ({CEILING_DEAD:.2f} kN/m²), buiten de openingen. "
                 f"Houtdichtheid: {DENSITY:g} kg/m³. Lichtstraten: elk {SKYLIGHT_KG:g} kg.")
        st.write("Per lichtstraat gaat 25% van het eigen gewicht én 25% van de uniforme variabele "
                 "belasting naar iedere rand. Dit is de afgesproken modelaanname. "
                 "Op de vrije opening zijn geen daklagen of plafond gerekend.")
        st.write("Sneeuw 0,56; regenwater 1,00; onderhoud vlaklast 1,00 kN/m²: afzonderlijke scenario's "
                 "op dak en lichtstraten. Onderhoudspuntlast 1,50 kN: conservatieve lokale envelop "
                 "per onderdeel, zonder gelijktijdig alle balken te belasten.")
        st.write("De waarden uit het referentievoorbeeld zijn nog geen gevalideerde "
                 "normcombinaties. kmod/psi2 per belastingtype, regen op verhoogde lichtstraten, "
                 "opstandgewicht, plafondcriteria en wateraccumulatie moeten nog worden vastgesteld.")
        st.write("BGT is t.o.v. de eigen opleglijn; doorbuiging van bovenliggende/dragende balken "
                 "en verbindingsslip zijn niet samengesteld. Reacties worden wel doorgegeven.")
        st.json(PROFILE)
        details=pd.DataFrame([{"Balk":r["name"],"Type":r["group"],"Profiel":r["section"],
                               "Oplegasafstand [mm]":r["length"],"UGT":r["ugt"],"BGT":r["bgt"],
                               "Einddoorbuiging [mm]":r["deflection"],"Grens [mm]":r["limit"],
                               "Bepalend UGT":r["ugt_case"],"Bepalend BGT":r["bgt_case"]}
                              for r in result["rows"]])
        st.dataframe(details.round(3),hide_index=True,width="stretch")
        st.download_button("Download balkresultaten",details.to_csv(index=False,sep=";").encode("utf-8-sig"),
                           "schinkel-controle.csv","text/csv")
        st.caption("Evenwichtscontrole belastingafdracht: maximaal verschil "
                   f"{max(abs(v) for v in result['balance'].values()):.2e} N. "
                   "Optimalisatielijst: 16 voorlopige handelsmaten × 10 h.o.h.-waarden; "
                   "leverbaarheid en plaatoverspanning worden niet getoetst.")


if __name__=="__main__":
    main()
