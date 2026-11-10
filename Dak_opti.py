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
            bearing_left=st.number_input("Muur / kolom links [mm]",min_value=1,max_value=500,value=150,step=10,key="bearing_left")
            bearing_right=st.number_input("Muur rechts [mm]",min_value=1,max_value=500,value=150,step=10,key="bearing_right")
            st.caption("Alleen bij fysieke muur-/kolomopleggingen. Balkschoenen krijgen geen toeslag.")
    p=Parameters(strength,sheet,spacing,joist,edge,header,bearing_left,bearing_right)
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
