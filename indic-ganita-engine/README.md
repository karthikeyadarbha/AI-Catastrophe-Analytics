# indic-ganita-engine

A sibling to `astral-mcp-server`. Where the astral server answers "where was the sky, by modern ephemeris?", this one answers "where does the sky come out by the Indian methods, and which verse says so?".

## Two engines
- **Engine 1 – rules (`indic_engine/rules`)**: a verse-referenced registry (51 records). Each record has text, chapter/verse, the page of the scan where it was checked, the formula, the function that implements it, a status (`implemented` / `registered-only`) and a flag saying whether the Sanskrit was verified against the scan.
- **Engine 2 – computation (`indic_engine/compute`)**: Ahargaṇa and mean planets, true Sun/Moon, four-step (exterior) and two-step (interior) planets, Rsine tables (Āryabhaṭa, Tantrasaṅgraha, Mādhava), and the Mādhava sine/cosine/arctan series.

## Three texts, used together
| Text | What it supplies |
|---|---|
| Āryabhaṭīya | base parameters (alternate set `AB`), Rsine differences, epicycle sizes |
| Tantrasaṅgraha | primary parameter set `TS`, dhruvas, planetary algorithm, Mādhava tables |
| Gaṇitayuktibhāṣā | the reasoning and the series behind the tables |
| Indian Astronomy: A Source Book | ayanāṃśa models, calendar elements, latitudes, conjunctions, visibility, occultations, inscription eclipses (v1.1.0) |
| Indrasena 2011 (Dulakara paper) | the Dulakara/Indrasena ayanāṃśa |

None of the three contains earthquake rules. Those are in Bṛhat Saṃhitā ch. 32, Adbhuta Sāgara and Parāśara Tantra; this platform supplies the Indian-method sky for testing them.

## Tools
`list_sources`, `search_rules`, `get_rule`, `indic_positions`, `jya`, `madhava_series`, `compare_treatises`, `ayanamsa`, `compare_ayanamsas`, `panchanga`, `planet_latitudes`, `graha_yuddha`, `visibility`, `rohini_bheda`, `drk_positions`, `ss_positions`.

## Run
```
pip install -e .
indic-ganita-engine        # stdio
python -m tests         # or pytest tests
```
See `FREEZE.md` for the frozen specification and known limits, and `validation/` for the comparison with the modern ephemeris.


## Added source (v1.2.0)
K. Chandra Hari, *Hindu Zodiac and Ancient Astronomy*: used for the primary definition of the Chandrahari ayanāṃśa and the Siddhantic zero-point discussion.


## Added source (v1.3.0)
Parameśvara, *Dṛggaṇita* (1431 CE), ed. K. V. Sarma (Hoshiarpur 1963): Kerala Dṛk constants, apsides and equation tables (`compute/drk.py`, `rules/data/drgganita.json`). Validation: `validation/validate_drk.py`.


## Added sources (v1.4.0 – v1.4.1)
*Sūryasiddhānta*, Hindi Vijñāna-bhāṣya edition: Siddhāntic mean and true planets, apsides and nodes, Rsine table, variable epicycles, retrograde limits (`compute/ss.py`, `rules/data/suryasiddhanta.json`, tool `ss_positions`). Validation: `validation/validate_ss.py`.
Sanskrit *Sūryasiddhānta*, R. C. Pandey / Chaukhamba edition with Raṅganātha's commentary: numeral-by-numeral concordance with the Hindi edition (rule `SS.concordance`).

## Version
1.4.2. See `FREEZE.md` for the manifest and combined hash that experiment results must cite.
