"""Calendar elements from true sidereal longitudes (Source Book ch.11; TS 2.24). All angles in degrees."""
NAK = ["Asvini","Bharani","Krttika","Rohini","Mrgasira","Ardra","Punarvasu","Pusya","Aslesa","Magha","Purvaphalguni","Uttaraphalguni","Hasta","Citra","Svati","Visakha","Anuradha","Jyestha","Mula","Purvasadha","Uttarasadha","Sravana","Dhanistha","Satabhisaj","Purvabhadrapada","Uttarabhadrapada","Revati"]
YOGA = ["Viskambha","Priti","Ayusman","Saubhagya","Sobhana","Atiganda","Sukarma","Dhrti","Sula","Ganda","Vrddhi","Dhruva","Vyaghata","Harsana","Vajra","Siddhi","Vyatipata","Variyan","Parigha","Siva","Siddha","Sadhya","Subha","Sukla","Brahma","Indra","Vaidhrti"]
KARANA7 = ["Bava","Balava","Kaulava","Taitila","Gara","Vanija","Visti"]
def tithi(sun, moon):
    x = ((moon - sun) % 360)/12.0; n = int(x)
    return {"number": n+1, "fraction_elapsed": x-n, "paksa": "sukla" if n < 15 else "krsna"}
def nakshatra(moon):
    x = (moon % 360)/(800.0/60.0); n = int(x)
    return {"number": n+1, "name": NAK[n], "pada": int((x-n)*4)+1, "fraction_elapsed": x-n}
def yoga(sun, moon):
    x = ((sun + moon) % 360)/(800.0/60.0); n = int(x)
    return {"number": n+1, "name": YOGA[n], "fraction_elapsed": x-n}
def karana(sun, moon):
    k = int(((moon - sun) % 360)/6.0)      # 0..59, half-tithis
    if k == 0: name = "Kimstughna"
    elif k <= 56: name = KARANA7[(k-1) % 7]
    else: name = ["Sakuni","Catuspada","Naga"][k-57]
    return {"index": k+1, "name": name}
def pata(sun, moon, ayanamsa):
    """Vyatipata / Vaidhrta: the sum of the savana (tropical) longitudes of Sun and Moon equals 180 / 360 degrees.
    Returns the signed distance (degrees) of the sum from each point."""
    s = (sun + moon + 2*ayanamsa) % 360
    d = lambda t: ((s - t + 180) % 360) - 180
    return {"vyatipata_distance": d(180.0), "vaidhrta_distance": d(0.0)}
def panchanga(sun, moon, ayanamsa=None):
    out = {"tithi": tithi(sun, moon), "nakshatra": nakshatra(moon), "yoga": yoga(sun, moon), "karana": karana(sun, moon)}
    if ayanamsa is not None: out["pata"] = pata(sun, moon, ayanamsa)
    return out
