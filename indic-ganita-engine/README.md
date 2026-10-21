# indic-ganita-engine

A sibling to `astral-mcp-server`. Where the astral server answers "where was the sky, by modern ephemeris?", this one answers "where does the sky come out by the Indian methods, and which verse says so?".

## Two engines
- **Engine 1 – rules (`indic_engine/rules`)**: a verse-referenced registry (30 records). Each record has text, chapter/verse, the page of the scan where it was checked, the formula, the function that implements it, a status (`implemented` / `registered-only`) and a flag saying whether the Sanskrit was verified against the scan.
- **Engine 2 – computation (`indic_engine/compute`)**: Ahargaṇa and mean planets, true Sun/Moon, four-step (exterior) and two-step (interior) planets, Rsine tables (Āryabhaṭa, Tantrasaṅgraha, Mādhava), and the Mādhava sine/cosine/arctan series.

## Three texts, used together
| Text | What it supplies |
|---|---|
| Āryabhaṭīya | base parameters (alternate set `AB`), Rsine differences, epicycle sizes |
| Tantrasaṅgraha | primary parameter set `TS`, dhruvas, planetary algorithm, Mādhava tables |
| Gaṇitayuktibhāṣā | the reasoning and the series behind the tables |

None of the three contains earthquake rules. Those are in Bṛhat Saṃhitā ch. 32, Adbhuta Sāgara and Parāśara Tantra; this platform supplies the Indian-method sky for testing them.

## Tools
`list_sources`, `search_rules`, `get_rule`, `indic_positions`, `jya`, `madhava_series`, `compare_treatises`.

## Run
```
pip install -e .
indic-ganita-engine        # stdio
python -m tests         # or pytest tests
```
See `FREEZE.md` for the frozen specification and known limits, and `validation/` for the comparison with the modern ephemeris.
