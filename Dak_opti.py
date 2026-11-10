"""Projectinvoer en uitbreidpunten. Alle lengtematen in mm, lasten in kN/m².

Dit bestand bevat nog GEEN constructieve toets of optimalisatie.
"""
from dataclasses import asdict, dataclass, field

# Voorlopige handelsmatencatalogus. Stem werkelijke geschaafde maten en
# leverbare combinaties af met de leverancier; breid deze lijst zelf uit.
STANDARD_SECTIONS = [
    (45, 145), (45, 170), (45, 195), (45, 220),
    (63, 145), (63, 170), (63, 195), (63, 220), (63, 245),
    (75, 195), (75, 220), (75, 245), (75, 270),
    (100, 220), (100, 245), (100, 270),
]


@dataclass
class Geometry:
    main_width: int = 5480
    main_depth: int = 2824
    extension_width: int = 2455
    extension_depth: int = 1220
    skylight_size: int = 1500
    skylight_x: tuple[int, int] = (490, 3490)
    skylight_y: int = 662  # vanaf onderste rand hoofddeel

    @property
    def extension_x(self):
        return self.main_width - self.extension_width


@dataclass
class Spans:
    # Voorlopig binnenmaten. Later vervangen door werkelijke hart-op-hartmaten.
    joist_main: int = 2824
    joist_extension: int = 1220
    upper_main: int = 5480
    upper_extension: int = 2455
    left_extension: int = 1220
    # Positie langs B1 vanaf hart linker muuroplegging, geen extra steunpunt.
    shoe_position: int = 3025


@dataclass
class Parameters:
    timber: str = "Vuren"
    strength_class: str = "C24"
    sheet_mm: int = 18
    spacing_mm: int = 600
    joist_section: tuple[int, int] = (63, 195)
    main_section: tuple[int, int] = (75, 245)
    upper_section: tuple[int, int] = (75, 245)
    left_section: tuple[int, int] = (63, 195)
    trimmer_section: tuple[int, int] = (63, 195)
    goal: str = "Minimaal houtvolume"
    roof_dead: float = 0.15
    snow: float = 0.56
    rain: float = 1.0
    maintenance_area: float = 1.0
    maintenance_point: float = 1.5  # kN, geen vlaklast!
    skylight_mass: float = 120.0  # kg per stuk
    spans: Spans = field(default_factory=Spans)

    @property
    def sheet_dead(self):
        return {18: 0.10, 22: 0.12}[self.sheet_mm]

    @property
    def roof_layers_dead(self):
        return self.roof_dead + self.sheet_dead

    def to_dict(self):
        return asdict(self)


def validate(p: Parameters) -> list[str]:
    errors = []
    if not 0 < p.spans.shoe_position < p.spans.upper_main:
        errors.append("De balkschoenpositie moet tussen de oplegassen van B1 liggen.")
    if any(value <= 0 for value in asdict(p.spans).values()):
        errors.append("Oplegasafstanden en balkschoenpositie moeten positief zijn.")
    return errors


def design_joists(p: Parameters, geometry: Geometry) -> dict:
    """TODO: dwarsbalken én raveling; resultaten/reacties per lastgeval.

    Later: eigen gewicht, belastingafdracht lichtstraten, UGT en BGT.
    None betekent niet berekend; gebruik nooit 0 als schijnresultaat.
    """
    return {"status": "Nog niet berekend", "ugt": None, "bgt": None,
            "support_reactions_by_case": None}


def design_edge_beams(p: Parameters, joist_results: dict) -> dict:
    """TODO: B3 -> reactie balkschoen -> B1; B2 draagt op kolom en muur.

    B1 krijgt dwarsbalkreacties van twee kanten bij de opbouw én de
    geconcentreerde reactie uit B3. Doorlopend ondersteunde randbalken
    afzonderlijk behandelen, inclusief muurbevestigingen.
    """
    if joist_results["support_reactions_by_case"] is None:
        return {"status": "Wacht op dwarsbalken en raveling", "ugt": None, "bgt": None}
    return {"status": "Nog niet berekend", "ugt": None, "bgt": None}


def optimize(p: Parameters, geometry: Geometry) -> dict:
    """TODO: doorzoek STANDARD_SECTIONS en h.o.h.-opties na implementatie toetsen."""
    return {"status": "Rekenmodel nog niet geïmplementeerd", "optimum": None}


# ==================== TEKENINGEN ====================

"""Schematische geometrie los van de instelbare rekenoverspanningen."""
import math

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Polygon


WOOD = "#b57a32"
TEAL = "#167b83"
INK = "#273b50"
GLASS = "#d5edf8"


def draw_plan(p: Parameters, g: Geometry, highlight: str = "Dwarsbalken"):
    fig = Figure(figsize=(8, 6.3), layout="constrained")
    ax = fig.subplots()
    x0, w, d, e = g.extension_x, g.main_width, g.main_depth, g.extension_depth
    ax.add_patch(Polygon([(0, 0), (w, 0), (w, d + e), (x0, d + e),
                          (x0, d), (0, d)], facecolor="#faf7f0", edgecolor="none"))
    joist_color = TEAL if highlight == "Dwarsbalken" else "#c7b38f"
    # Gelijke veldverdeling: werkelijk getekende h.o.h. <= beoogde waarde.
    # Vaste ravelbalken liggen daarnaast op de randen van de openingen.
    n = math.ceil(w / p.spacing_mm)
    for i in range(1, n):
        x = i * w / n
        opening = any(s < x < s + g.skylight_size for s in g.skylight_x)
        intervals = [(0, g.skylight_y), (g.skylight_y + g.skylight_size, d)] if opening else [(0, d)]
        for ya, yb in intervals:
            ax.plot([x, x], [ya, yb], color=joist_color, lw=2)
    n = math.ceil(g.extension_width / p.spacing_mm)
    for i in range(1, n):
        x = x0 + i * g.extension_width / n
        ax.plot([x, x], [d, d + e], color=joist_color, lw=2)
    for xa, ya, xb, yb in [(0, 0, w, 0), (0, 0, 0, d), (w, 0, w, d+e),
                           (0, d, w, d), (x0, d+e, w, d+e), (x0, d, x0, d+e)]:
        ax.plot([xa, xb], [ya, yb], color=WOOD, lw=5, solid_capstyle="butt")
    for i, x in enumerate(g.skylight_x, 1):
        for xx in (x, x+g.skylight_size):
            ax.plot([xx, xx], [0, d], color=WOOD, lw=2.5)
        ax.add_patch(Rectangle((x, g.skylight_y), g.skylight_size, g.skylight_size,
                               facecolor=GLASS, edgecolor=WOOD, linewidth=3, zorder=4))
        ax.text(x+g.skylight_size/2, g.skylight_y+g.skylight_size/2,
                f"Lichtstraat {i}\n1.500 × 1.500 mm\n{p.skylight_mass:g} kg · 4 zijden",
                ha="center", va="center", fontsize=9, zorder=5)
    # Doorlopende muursteun langs de drie bevestigde randbalken.
    ax.plot([-85, -85], [0, d], color=INK, lw=3, ls=":")
    ax.plot([0, w], [-85, -85], color=INK, lw=3, ls=":")
    ax.plot([w+85, w+85], [0, d+e], color=INK, lw=3, ls=":")
    ax.scatter([0, w, w], [d, d, d+e], marker="^", s=90, color=INK, zorder=6)
    ax.scatter([x0], [d+e], s=100, color=INK, zorder=6)
    ax.scatter([x0], [d], marker="s", s=75, facecolor="white", edgecolor=INK, zorder=6)
    ax.annotate("Kolom", (x0, d+e), xytext=(-12, 10), textcoords="offset points", ha="right", fontsize=9)
    ax.annotate("B1", (w*.28, d), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10)
    ax.annotate("B2", ((x0+w)/2, d+e), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10)
    ax.annotate("B3", (x0, d+e*.5), xytext=(-10, 0), textcoords="offset points", ha="right", fontsize=10)
    if highlight == "Langsbalken":
        ax.annotate("R uit B3", (x0, d), xytext=(x0-750, d+400), fontsize=9,
                    arrowprops={"arrowstyle": "->", "color": INK})
    ax.set(xlim=(-350, w+350), ylim=(-400, d+e+400), aspect="equal",
           xlabel="Binnengeometrie x [mm]", ylabel="Binnengeometrie y [mm]")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    return fig


def draw_support_schemes(p: Parameters):
    """Werkelijke invoer op horizontale schema's; steunpunttypes schematisch."""
    s = p.spans
    fig = Figure(figsize=(9, 4), layout="constrained")
    axes = fig.subplots(3, 1)
    rows = [("B1 · bovenste lange randbalk", s.upper_main, "Muur", "Muur"),
            ("B2 · kleine bovenste lange randbalk", s.upper_extension, "Kolom", "Muur"),
            ("B3 · kleine linker randbalk", s.left_extension, "Balkschoen op B1", "Kolom")]
    for ax, (title, length, left, right) in zip(axes, rows):
        ax.plot([0, length], [0, 0], color=WOOD, lw=5)
        ax.scatter([0, length], [0, 0], marker="^", s=85, color=INK, zorder=3)
        ax.text(0, -.28, left, ha="center", fontsize=9)
        ax.text(length, -.28, right, ha="center", fontsize=9)
        ax.text(length/2, -.58, f"Hart-op-hart: {length:,.0f} mm".replace(",", "."), ha="center", fontsize=10)
        if title.startswith("B1"):
            ax.annotate("R uit B3 (nog te berekenen)", (s.shoe_position, 0),
                        xytext=(s.shoe_position, .62), ha="center", fontsize=9,
                        arrowprops={"arrowstyle": "->", "color": INK})
            ax.text(s.shoe_position, -.28, f"a = {s.shoe_position:g} mm", ha="center", fontsize=9)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set(xlim=(-length*.18, length*1.18), ylim=(-.8, 1))
        ax.axis("off")
    return fig


# ==================== STREAMLIT INTERFACE ====================

"""Start met: python -m streamlit run app.py

Interface, invoer, tekeningen en toekomstige berekeningen in dit bestand.
Startbestand voor Streamlit Community Cloud: app.py.
"""
import json
from dataclasses import asdict

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Optimalisatie raveling Schinkel", layout="wide")
g = Geometry()


def section_input(label, default, key):
    return st.selectbox(label, STANDARD_SECTIONS, index=STANDARD_SECTIONS.index(default),
                        format_func=lambda x: f"{x[0]} × {x[1]} mm", key=key)


def distance_input(label, value, key):
    return st.number_input(label, min_value=1, value=value, step=10, key=key)


with st.sidebar:
    st.header("Ontwerpparameters")
    st.caption("Houtsoort: vuren")
    strength = st.selectbox("Sterkteklasse", ["C24", "C18"], key="strength")
    sheet = st.selectbox("Underlayment [mm]", [18, 22], key="sheet")
    st.subheader("Dwarsbalken en raveling")
    spacing = st.slider("Beoogde maximale h.o.h. [mm]", 300, 750, 600, 25, key="spacing")
    joist = section_input("Dwarsbalk b × h", (63, 195), "joist")
    trimmer = section_input("Ravelbalken b × h", (63, 195), "trimmer")
    st.subheader("Vrij overspannende randbalken")
    main_section = section_input("B1 · bovenste lange", (75, 245), "main_section")
    upper_section = section_input("B2 · kleine bovenste lange", (75, 245), "upper_section")
    left_section = section_input("B3 · kleine linker", (63, 195), "left_section")
    st.caption("Voorlopige handelsmaten; leverbare combinaties en werkelijke maten afstemmen.")
    with st.expander("Oplegasafstanden [mm]", expanded=True):
        st.caption("Hart ondersteuning → hart ondersteuning. Standaard voorlopig gelijk aan binnenmaat.")
        jm = distance_input("Dwarsbalk hoofddeel", 2824, "span_joist_main")
        je = distance_input("Dwarsbalk opbouw", 1220, "span_joist_extension")
        b1 = distance_input("B1 · muur → muur", 5480, "span_upper_main")
        b2 = distance_input("B2 · kolom → muur", 2455, "span_upper_extension")
        b3 = distance_input("B3 · balkschoen → kolom", 1220, "span_left_extension")
        shoe = distance_input("a · linker oplegas B1 → balkschoen", 3025, "shoe_position")
    goal = st.selectbox("Optimalisatiedoel", ["Minimaal houtvolume", "Minimale balkhoogte"], key="goal")

p = Parameters(strength_class=strength, sheet_mm=sheet, spacing_mm=spacing,
               joist_section=joist, main_section=main_section,
               upper_section=upper_section, left_section=left_section,
               trimmer_section=trimmer, goal=goal,
               spans=Spans(jm, je, b1, b2, b3, shoe),
               **st.session_state.get("project_loads", {}))

st.title("Optimalisatie raveling Schinkel")
st.caption("App-opzet · dwarsbalken en raveling → oplegreacties → randbalken")
st.info("UGT, BGT, eigen gewicht, lastcombinaties en optimalisatie zijn nog niet geïmplementeerd. "
        "Gekozen doorsneden zijn invoer; er is nog geen uitspraak over geschiktheid.")
errors = validate(p)
for error in errors:
    st.error(error)
if errors:
    st.stop()

# Afgesproken ontwerpvolgorde. Voeg de rekenregels toe bij design_joists en design_edge_beams.
joist_results = design_joists(p, g)
edge_results = design_edge_beams(p, joist_results)

view = st.radio("Weergave", ["Dwarsbalken", "Langsbalken", "Belastingen", "Model / code"], horizontal=True)
if view in ("Dwarsbalken", "Langsbalken"):
    plot_col, info_col = st.columns([3, 2])
    with plot_col:
        st.subheader("Raveling · bovenaanzicht")
        st.pyplot(draw_plan(p, g, view), width="stretch")
        st.caption("B1/B2/B3: vrije randbalken · ● kolom · ▲ muuroplegging · "
                   "□ balkschoen · stippellijn: doorlopende muursteun. Balkdiktes schematisch.")
        st.caption("De plattegrond behoudt de binnenmaten. Gewijzigde oplegasafstanden staan "
                   "in de oplegschema's; ze veranderen de dakvorm en lichtstraatposities niet.")
    with info_col:
        if view == "Dwarsbalken":
            st.subheader("Keuze dwarsbalken")
            st.metric("Doorsnede b × h", f"{joist[0]} × {joist[1]} mm")
            st.metric("Beoogde maximale h.o.h.", f"{spacing} mm")
            st.metric("Oplegasafstand hoofddeel", f"{jm} mm")
            st.metric("Oplegasafstand opbouw", f"{je} mm")
            st.caption("Tekening: gelijke veldverdeling met h.o.h. ≤ ingestelde waarde; "
                       "vaste ravelbalken aanvullend. Geen definitief legplan.")
            st.write("**UGT / BGT:**", joist_results["status"])
        else:
            st.subheader("Keuze randbalken")
            for name, section in [("B1", main_section), ("B2", upper_section), ("B3", left_section)]:
                st.metric(name + " · b × h", f"{section[0]} × {section[1]} mm")
            st.write("**UGT / BGT:**", edge_results["status"])
            st.info("B1 ontvangt bij de opbouw dwarsbalkreacties van beide kanten "
                    "én een geconcentreerde reactie uit B3 via de balkschoen.")
        st.caption("Lichtstraten voorlopig op vier zijden opgelegd; lastverdeling nog bepalen.")
    st.subheader("Oplegschema's · actuele invoer")
    st.pyplot(draw_support_schemes(p), width="stretch")
    st.caption("B3: van onder (balkschoen op B1) naar boven (kolom). "
               "De balkschoen op B1 is een belastingpunt, geen extra ondersteuning van B1.")
    st.subheader("Te vergelijken varianten")
    st.dataframe(pd.DataFrame([
        {"Variant": "Huidige keuze", "Dwarsbalk [mm]": f"{joist[0]} × {joist[1]}",
         "H.o.h. [mm]": str(spacing), "UGT": "Nog niet berekend", "BGT": "Nog niet berekend"},
        {"Variant": "Optimum", "Dwarsbalk [mm]": "—", "H.o.h. [mm]": "—",
         "UGT": "Rekenmodel volgt", "BGT": "Rekenmodel volgt"},
    ]), hide_index=True, width="stretch")
    with st.expander("Vastgelegde opleggingen en geometrie"):
        st.write("Onderste lange, rechter en linker randbalk: doorlopend aan muur bevestigd en ondersteund. "
                 "Dit legt geen momentvaste inklemming vast. Muurbevestigingen krijgen later een eigen controle.")
        st.write("B1: muur–muur. B2: kolom–muur. B3: balkschoen op B1–kolom.")
        st.write("Hoofddeel: 5.480 × 2.824 mm. Opbouw: 2.455 × 1.220 mm. "
                 "Lichtstraten: elk 1.500 × 1.500 mm. x-maatketen: 490 / 1.500 / 1.500 / 1.500 / 490 mm; "
                 "y-maatketen: 662 / 1.500 / 662 mm.")
        st.write("Ravelbalken, doorgaande balken naast openingen, verbindingen en doorlopend "
                 "ondersteunde randbalken moeten in de volgende fase afzonderlijk worden uitgewerkt.")
elif view == "Belastingen":
    st.subheader("Belastingen · onbewerkte invoer")
    st.caption("Waarden uit de projectuitgangspunten, nog geen normtoets of lastcombinatie.")
    load_fields = [
        ("roof_dead", "Dakbedekking + isolatie [kN/m²]", 0.01),
        ("snow", "Sneeuw [kN/m²]", 0.01),
        ("rain", "Regenwater [kN/m²]", 0.05),
        ("maintenance_area", "Onderhoud vlaklast [kN/m²]", 0.05),
        ("maintenance_point", "Onderhoud puntlast [kN]", 0.05),
        ("skylight_mass", "Massa per lichtstraat [kg]", 5.0),
    ]
    for field_name, label, increment in load_fields:
        setattr(p, field_name, st.number_input(label, min_value=0.0,
                value=getattr(p, field_name), step=increment, key="load_" + field_name))
    # Bewaar los van widget keys: Streamlit verwijdert verborgen widgets bij navigatie.
    st.session_state["project_loads"] = {name: getattr(p, name) for name, _, _ in load_fields}
    st.metric("Permanente daklagen", f"{p.roof_layers_dead:.2f} kN/m²")
    st.caption("Exclusief eigen gewicht balken en lichtstraten. Underlayment: "
               f"{p.sheet_dead:.2f} kN/m². Houtdichtheid en gewichtsafdracht volgen.")
    st.info("Sneeuw, wateraccumulatie en onderhoud zijn afzonderlijke invoergegevens. "
            "Combineren, gelijktijdigheid en plaatsing van puntlasten werken we later uit.")
    st.session_state["project_simple_foundation"] = st.checkbox(
        "Vereenvoudiging fundering opnemen: 1,5 m dakbreedte per meter zijbalk",
        value=st.session_state.get("project_simple_foundation", False), key="simple_foundation")
    st.caption("Nog geen funderingsreacties; opbouw en raveling apart afdragen. "
               "Dakstroken niet dubbel tellen met berekende balkreacties.")
else:
    st.subheader("Verder coderen")
    st.markdown("- `Geometry`, `Spans`, `Parameters`: projectinvoer.\n- `STANDARD_SECTIONS`: handelsmaten.\n"
                "- `draw_plan`, `draw_support_schemes`: tekeningen.\n"
                "- `design_joists`, `design_edge_beams`, `optimize`: toekomstige berekeningen.")
    st.code("joists = design_joists(p, g)\nedges = design_edge_beams(p, joists)\n"
            "result = optimize(p, g)", language="python")
    st.json({"parameters": p.to_dict(), "geometry": asdict(g)})

st.download_button("Download huidige invoer (JSON)",
                   json.dumps({"parameters": p.to_dict(), "geometry": asdict(g),
                               "simple_foundation": st.session_state.get("project_simple_foundation", False)},
                              ensure_ascii=False, indent=2),
                   file_name="schinkel-invoer.json", mime="application/json")

