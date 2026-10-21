# FREEZE – indic-ganita-engine v1.2.0

**Frozen on 2026-10-21, before it is applied to Experiment 1 or Experiment 2.** v1.1.0 only added modules and v1.2.0 only changes the Chandrahari ayanāṃśa anchor; the v1.0.0 computation (positions, Rsine, series) is unchanged. After this point, any change to a file listed in the manifest below requires a new version number and a note here. Results from the experiments must cite this version and the combined hash.

## What is frozen
- Primary configuration: `param="TS"` (Tantrasaṅgraha parameters), `scheme="madhava"` (Mādhava Rsine table, R = 3437′44″48‴).
- Alternate configuration, registered but lower confidence: `param="AB"` (Āryabhaṭīya parameters). Reason: it assumes the equal-pāda reading (Kali start = 3/4 yuga, planets at 0°). That reading is an assumption, and it gives clearly larger errors (below).
- Kali epoch JD 588465.5 (fixed by the book's worked example: 14 Jan 2002 = Ahargaṇa 1,863,823). Mean sunrise at Ujjain 06:00 local, 75.7684°E (external constant, not from the texts).
- `indic_positions` returns the sidereal frame implied by the dhruvas. Ayanāṃśa models (below) are used to convert modern tropical data to sidereal; they are separate from that frame.

## Interpretation choices (the book's text allows more than one reading)
1. Half/full order of manda and śīghra steps in TS 2.26 (exterior planets).
2. Saturn manda ratio taken as 39/320 from the verse reading.
3. Āryabhaṭīya Rsine 16th difference: 119 used; the printed edition shows 117 (flagged in rule `AB.jya`).
4. Sine/cosine series: existence confirmed in the texts, but the exact Sanskrit verse page was not located in the scan (flagged `sanskrit_verified: false`).

## Acceptance criteria (set before running the experiments)
- Reproduces the book's worked examples (Ahargaṇa, Moon mean longitude 8r26°29′, Rsine interpolation 1843′41″). **Met** (tests).
- True-longitude standard deviation against the modern ephemeris (Lahiri sidereal) at most 2.5° for every body in 1900–2029. **Met** for TS (see below).
- AB set: no criterion; reported as is.

## Validation (vs own ephemeris `myeph`, validated against the astral MCP server)
Standard deviation of (Indic − modern) in degrees, 2000–2003, TS: Sun 0.17, Moon 1.39, Mars 1.26, Mercury 2.13, Jupiter 0.28, Venus 1.41, Saturn 0.27. Largest single errors: Mercury 6.2°, Mars 3.7°, Venus 4.1°. For AB: Venus sd 10.7°, Saturn mean offset −8°, Mars −4°, Jupiter +6°. Full tables: `validation/validation_vs_modern_2026-10.txt`.

## Added in v1.1.0 (source: Indian Astronomy: A Source Book, plus the Indrasena/Dulakara paper)
1. **Ayanāṃśa models** (`compute/ayanamsa.py`): Tropical (= 0), Lahiri, Raman, KP, Chandrahari, Dulakara/Indrasena (modern: IAU 2006 general precession anchored to a published value or zero year) and three treatise models (Sūryadeva/Āryabhaṭa school, Sūryasiddhānta/Nīlakaṇṭha, Deva/Amaraja).
2. **Tithi, nakṣatra, yoga, karaṇa, vyatīpāta distance** (`compute/panchanga.py`).
3. **Planetary latitude** (`compute/latitude.py`), **conjunction and graha-yuddha** (`compute/conjunction.py`), **heliacal visibility** (`compute/visibility.py`), **star occultation and Rohiṇī-śakaṭa-bheda** (`compute/occultation.py`).
4. **Validation**: `validation/inscription_eclipses.py`, `validation/rohini_bheda_scan.py`, `validation/ayanamsa_comparison.csv`.

## v1.1.0 results
- **Ayanāṃśa**: recovered zero years Lahiri 285.7, Raman 390.0, KP 292.7, Chandrahari 233.3, Dulakara 243.9 CE (published 285, 389, 292, 233, 244). Values on 27 Oct 2006 within 3″ of the Overbeck table. Dulakara reproduces the author's own 1900–2019 table within 4.4″ (sd 0.4″). On 1 Jan 2000: Lahiri 23.857°, Raman 22.411°, KP 23.760°, Chandrahari 24.583°, Dulakara 24.428° (true, with nutation), Āryabhaṭa school 23.191°, Sūryasiddhānta/Nīlakaṇṭha 22.511°, Deva/Amaraja 22.289°.
- **Planetary latitude** vs the astral server (6 dates, 1995–2020): rms Mars 0.69°, Mercury 0.60°, Jupiter 0.25°, Venus 0.35°, Saturn 0.48°, Moon 0.45°. Signs correct; magnitudes about 20% low because the old maximum latitudes are smaller than the true inclinations.
- **Inscription eclipses** (15 syzygies, 580–1231 CE): engine and modern ephemeris agree to a maximum of 0.5 h (mean 0.15 h). Kalachuri 880 (1128 CE): full moon 8 Nov, 02:22 IST (third quarter of the night), Thursday by the sunrise-based weekday, Moon 0.3° from the node: matches the inscription. Śaka 1113 solar eclipse: 23 Jun 1191, Sunday, matches. Śaka 1119 (Sat) lunar eclipse: no matching eclipse in Mārgaśīrṣa; other lunar and solar entries are inconclusive. The Source Book dates Kalachuri 322 and 404 as 580 and 662 CE; with the epoch that fits Kalachuri 880 = 1128 (248/9 CE) they would be 570 and 652 CE, a 10-year inconsistency.
- **Rohiṇī-bheda scan** 1900–2025: 25 episodes (mostly Venus and Mercury; Saturn once, 1942). Not yet validated against an independent latitude source for these dates.

## Interpretation choices added in v1.1.0
5. "Indrasena ayanāṃśa" is taken to be the Dulakara ayanāṃśa (same author). The Scribd page could not be read.
6. "Tropical ayanāṃśa" is taken to be zero (sāyana).
7. (Superseded in v1.2.0) Chandrahari previously rested on Overbeck 2007.
8. Deva/Amaraja sign convention is inferred.
9. Rohiṇī-bheda longitude window (1°) and the visibility elongation (ecliptic only, no dṛkkarma) are our approximations; the Source Book gives neither.
10. Only Bheda has a numerical limit in the source; Ullekha, Aṃśumardana and Apasavya are not quantified.
11. Source Book text was read by OCR; items checked against page images: visibility limits (p. 257), Dulakara paper (PDF text layer). Junction-star table and Venus diameter are OCR-read only.

## Changed in v1.2.0 (source: K. Chandra Hari, Hindu Zodiac and Ancient Astronomy)
- **Chandrahari ayanāṃśa re-anchored to the author's own definition**: Mūla fixed at sidereal 240°, ayanāṃśa 24°35′08.62″ at J2000.0, zero precession 1 Feb 233 10:46 (Appendix V, p. 476). Reproduces his printed tables for 1800–2100 within 0.2″; the model's zero year is 233.1 (12 days from his date). The earlier Overbeck-based anchor differed by about 9″ in 2006. Values changed only for the ChandraHari model; no other model, position, panchanga, latitude or eclipse result changes.
- Registry: `AY.ChandraHari` rewritten (primary source), `AY.ChandraHari.siddhantic` added (his Chapter III.5: epoch of the Āryabhaṭīya/Sūryasiddhānta at Kali 3600 = 21 Mar 499; the Siddhantic mean Sun lags the equinox by 15′, so Siddhantic and tropical Aries coincide in AD 522; original Sūryasiddhānta zero AD 231).
- Not changed: his Chapter IV discusses other ayanāṃśa schools (Citrapakṣa 285, Revatī 576, Fagan-Bradley 220, Raman, Devadatta) and Lahiri's star-by-star zero years. These are commentary and arguments; our models for those schools keep their original anchors.
- Interpretation choice 12: the book's printed "+" signs at J0/J100/J200 in his tables are read as negative (standard convention); with that reading the match above holds.
- OCR coverage: Chapters III.5, IV and parts of V were read; the remaining chapters (stellar identification, Mūla/Yogatārā evidence, appendices) were not read in this version.

## Manifest (SHA-256)
```
cd5e18ea58043985038dece4ad8f94ca42b3acc61dc72f84123224b4a1c48fd3  indic_engine/__init__.py
a50087d855e41315033e900e636253fffcbbabb9af2618578bd61ec318df6dba  indic_engine/compute/__init__.py
1c59f169a02815fa1580646bf90947808d34f33faca545c5f8ce1da8554338e6  indic_engine/compute/ayanamsa.py
e0c206d30cd599eee247c86017b1c97cdd1219b462af529e609ffd3d85a1fa23  indic_engine/compute/conjunction.py
0fad0df9537f6abce03c96a946c48fd054e21189ffe5e24fb08aa696b64f1ce0  indic_engine/compute/jya.py
6d3530b31ef9c7835a1985fb12ccf03c19fa7fd9ccc3a29e6b69c46fb2c2e822  indic_engine/compute/latitude.py
1bf8d2b218474e9926b398b74f318f7dffb97835c7af0db1efb94141b1c6076d  indic_engine/compute/occultation.py
8be360f69544037ddf1541d0627a7b4cbf077ba8b65ae155f58ce139862d5dd8  indic_engine/compute/panchanga.py
6abed8e32eae0a7b0c1d555084a9d5251f6cee2ef1c29810f48d19b6517eff96  indic_engine/compute/params.py
9801a526da11741623d80cf303cca35bad12db50f306f2ccf73e357aae4337a1  indic_engine/compute/positions.py
f376404d17cbc0e314cc9b6e8f4add46e063dba6eacb681a0e2db55a112edcc7  indic_engine/compute/series.py
821e49af9d3a06c3510babc370755c4ebd65cb5c8cd129dda5c3e921a6743265  indic_engine/compute/visibility.py
bf8d643aaf791dd9a45b4d79375caa6dab78cc0e7a27a461b6f7612ec58c0f1e  indic_engine/rules/__init__.py
336c1a7515b75b4745280973951173e4880f0fc2fa8a5a25f4b0ad9ceb5bf36e  indic_engine/rules/data/aryabhatiya.json
9c0fedeafbc5ad17579a3d6ef37fc6836a7b0b54016512907b35fe8a8d09f213  indic_engine/rules/data/concordance.json
c5fc1a0ee7d83a1628bc5bebed0ca1545793676ac0c776ad21ccaf41a4cb356c  indic_engine/rules/data/sourcebook.json
0f897478aa84f86730034c92374c1b8dcf214ad3efaa03a31d52dfe34728d2f2  indic_engine/rules/data/tantrasangraha.json
3a73514af28a28ce92f6b78199c2ba805268cfc551b4bae114e0d21214793f99  indic_engine/rules/data/yuktibhasa.json
dc9ee49ce419d01d8d0a5c17416c1e4978bdd2a7bb0570b08b67dd094179fefe  indic_engine/rules/registry.py
c1c23a909698fafbdd46e91f0e42f4c6b1c8d680c1c052074a6945ba340cfb6e  indic_engine/server.py
```

Combined hash (indic_engine/*.py and *.json): a1b80bdc488bf2ca
