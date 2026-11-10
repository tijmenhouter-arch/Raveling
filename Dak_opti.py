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
    x = np.unique(np.r_[np.linspace(0.0, length, 161), knots, extrema])

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
    i = int(np.argmax(down))
    if 0 < i < len(x)-1:
        # Lokale coördinaten voorkomen slechte conditionering van de fit.
        coeff = np.polyfit(x[i-1:i+2]-x[i], down[i-1:i+2], 2)
        if coeff[0] < 0:
            x_peak = x[i] - coeff[1] / (2*coeff[0])
            if x[i-1] <= x_peak <= x[i+1]:
                u_peak = float(values([x_peak])[1][0])
                if u_peak > down[i]:
                    x = np.r_[x, x_peak]
                    moment, down = values(x)
    return {"left": left, "right": right, "x": x,
            "M": moment, "uEI": down,
            "maxM": float(np.max(np.abs(moment))),
            "maxuEI": float(np.max(np.abs(down)))}


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
        b, h = beam.section
        W, EI = b*h*h/6, material["E"]*b*h**3/12
        calculated = {}
        for case in CASES:
            calculated[case] = response(beam.length, beam.loads[case])
            for receiver, force in zip(beam.receivers,
                                       (calculated[case]["left"], calculated[case]["right"])):
                if receiver is None:
                    terminal[case] += force
                else:
                    add_point(m.beams[receiver[0]], case, force, receiver[1])
        G = calculated["Permanent"]
