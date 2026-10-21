# FREEZE – indic-ganita-engine v1.0.0

**Frozen on 2026-10-21, before it is applied to Experiment 1 or Experiment 2.** After this point, any change to a file listed in the manifest below requires a new version number and a note here. Results from the experiments must cite this version and the combined hash.

## What is frozen
- Primary configuration: `param="TS"` (Tantrasaṅgraha parameters), `scheme="madhava"` (Mādhava Rsine table, R = 3437′44″48‴).
- Alternate configuration, registered but lower confidence: `param="AB"` (Āryabhaṭīya parameters). Reason: it assumes the equal-pāda reading (Kali start = 3/4 yuga, planets at 0°). That reading is an assumption, and it gives clearly larger errors (below).
- Kali epoch JD 588465.5 (fixed by the book's worked example: 14 Jan 2002 = Ahargaṇa 1,863,823). Mean sunrise at Ujjain 06:00 local, 75.7684°E (external constant, not from the texts).
- Ayanāṃśa is NOT computed from the texts: `indic_positions` returns the sidereal frame implied by the dhruvas. Tantrasaṅgraha 3.15 (trepidation) is registered-only because its zero epoch is not stated.

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

## Manifest (SHA-256)
```
79f8ed84ce58ca7ed906dc0deddfccd67dfaee37b4d11d3e7099c13651b6a7a0  indic_engine/__init__.py
a50087d855e41315033e900e636253fffcbbabb9af2618578bd61ec318df6dba  indic_engine/compute/__init__.py
0fad0df9537f6abce03c96a946c48fd054e21189ffe5e24fb08aa696b64f1ce0  indic_engine/compute/jya.py
6abed8e32eae0a7b0c1d555084a9d5251f6cee2ef1c29810f48d19b6517eff96  indic_engine/compute/params.py
9801a526da11741623d80cf303cca35bad12db50f306f2ccf73e357aae4337a1  indic_engine/compute/positions.py
f376404d17cbc0e314cc9b6e8f4add46e063dba6eacb681a0e2db55a112edcc7  indic_engine/compute/series.py
bf8d643aaf791dd9a45b4d79375caa6dab78cc0e7a27a461b6f7612ec58c0f1e  indic_engine/rules/__init__.py
336c1a7515b75b4745280973951173e4880f0fc2fa8a5a25f4b0ad9ceb5bf36e  indic_engine/rules/data/aryabhatiya.json
9c0fedeafbc5ad17579a3d6ef37fc6836a7b0b54016512907b35fe8a8d09f213  indic_engine/rules/data/concordance.json
e1f93b54a53d3f9301f6069e3ec3e6186739c52f00ce377316d2668ae3db4f29  indic_engine/rules/data/tantrasangraha.json
3a73514af28a28ce92f6b78199c2ba805268cfc551b4bae114e0d21214793f99  indic_engine/rules/data/yuktibhasa.json
dc9ee49ce419d01d8d0a5c17416c1e4978bdd2a7bb0570b08b67dd094179fefe  indic_engine/rules/registry.py
4829790687237adffb549df5836467e805465c060b17dbc7204bfde435657e76  indic_engine/server.py
a2cd1e0e02d3d9b24cad758da33e0df46d27fa30d3702f48919116dc9479dbd1  tests/test_engine.py
628bfe9fc1f575e460fdbbb4ab43e6f6a4fd97d1d5a971a922857c8994b89a7f  tests/test_registry.py
```

Combined hash: 495ed4aaa017a82b
