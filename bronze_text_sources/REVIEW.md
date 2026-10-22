# Bronze layer: code and data review

Scope: `bronze_text_sources/` at commit a62ce8d, checked against the frozen engine
(`indic-ganita-engine` v1.4.2, not modified), the experiments in `analysis/`, and the
preregistrations in `docs/audit/`. Figures come from `manifest/source_register.json`
and from the test run (26 tests passing).

## What the bronze layer holds now

| Item | Count | Status |
|---|---|---|
| Classical texts and editions named in the project | 13 | none has a scan in the repo, so none can be hashed |
| Unresolved source named in the project (Cunningham) | 1 | no edition or file located |
| Engine rule files (transcribed extracts) | 7 files, 85 rules | hashed and recorded in the audit chain |
| Catalogue-derived or curated data files | 11 | hashed and recorded; rows counted |
| Audit entries written this run | 18 | chain verified; second run added 0 |

Catalogues named in the project: ISC-GEM (pre-1990 events) and USGS ComCat (1990 onward).
The 13 texts are Āryabhaṭīya, Tantrasaṅgraha, Gaṇitayuktibhāṣā, Dṛggaṇita, Sūryasiddhānta
(Hindi Vijñāna-bhāṣya edition), Sūryasiddhānta (Pandey edition with Raṅganātha's
commentary), Indian Astronomy: A Source Book, Hindu Zodiac and Ancient Astronomy,
Dulakara Ayanamsha (Indrasena 2011), Karaṇaratna and Khaṇḍakhādyaka (cited through the
Source Book), Bṛhat Saṃhitā, Adbhuta Sāgara, and Parāśara Tantra (as quoted).

## Findings

Ranked by impact on the research claims. "Fixed" means the change is in this commit.

### High

1. **No raw text is in bronze for any classical text.** Nothing in the repo lets us verify a classical text's bytes. The immutability claim in the README therefore holds only for the catalogues and the engine extracts. *Action:* add the scans or PDFs under `bronze_text_sources/raw/` and rerun the builder. These files are external and are not included here.

2. **Engine rule records do not conform to `rule.schema.json`.** 0 of 85 records have the required `rule_id`, `source_reference` or `provenance` fields. Engine IDs (e.g. `TS.1.9.rev`) fail the schema's `^[A-Z]{2,3}-[0-9]{3,4}$` pattern, and the schema's `text` field expects the rule text, while the engine stores the treatise name there. *Decision needed:* either adapt the schema to the engine's record shape or convert the records. The register reports this as `schema_conformant` for each file.

3. **Provenance is thin.** Of 85 rules, 18 carry a page reference (`pdf_pages`), 2 carry Sanskrit text, and 2 are marked Sanskrit-verified. None records OCR confidence. The schema requires `source_page` and `extraction_method` in `provenance`. Citing a rule to a page is therefore not possible for most of the set.

4. **The OCR gate did not measure OCR quality (fixed).** The old check flagged any `l`, `0`, `1`, `5`, `8`, `O`, `S`, `B` in the text and scored each as a fixed penalty. Examples checked against the old logic: `kalpa yuga manvantara 1.10 loka` scored 0.8 and was rejected. `loka līlā kalpa sūkṣma` was flagged for its `l` and passed only because one issue is tolerated. The new gate flags mixed digit-letter tokens and characters outside the IAST/Devanagari set, and accepts citation tokens (`vv.15-18`, `p.47`). Pass/fail decisions change for texts with digits or ordinary `l`. *Note:* this is a screening heuristic; a calibrated OCR confidence needs the OCR engine's own word scores.

5. **Transliteration output was labelled IAST without being converted (fixed).** `transliterate_iast` returns its input, and `ingest()` reported it as `iast_text`. Now `iast_text` is `None` and `transliteration_status` is `not_implemented` until Aksharamukha is wired in. Silver and Gold cannot rely on IAST yet.

6. **The audit log never checked the source files (fixed).** `verify()` checked only the chain. `sha256_file` existed but was never compared to a recorded hash. `verify_source(path)` now does this, and the register builder records every artifact.

### Medium

7. **The audit chain cannot detect removal of its last entry.** Nothing refers to the last entry. Documented in `audit.py` and the README. *Action:* commit the audit file and tag or record its head hash outside the repo.

8. **Audit `extraction_method` values are not in the schema's enum.** The builder uses `registered_by_reference` and `transcribed_extract`; the schema allows `manual`, `ocr_assisted`, `fully_automated`. Decide whether to extend the enum.

9. **Deprecated timestamp call (fixed).** `datetime.utcnow()` is replaced with a timezone-aware UTC timestamp.

10. **Data files have no bronze schema.** The schema covers rules only, so catalogue inputs are registered without a schema. The register's `catalogue` and `role` fields for DAT-02, DAT-03 and DAT-06 are inferred from file names and column schemas and are marked "confirm".

11. **`extract_word_confidence` is a triage heuristic, not a measure of OCR error.** Its penalties are for length and non-letter characters. It does not feed the pass/fail gate.

12. **The README pointed at documents that do not exist** (`docs/silver_layer/`, `docs/gold_layers/`, `docs/design/`) and listed a `pipeline/__init__.py` that was missing (now added). Corrected.

### Low

13. Tests use `unittest` so they run without installing anything. There are 26 tests across the audit chain and the gate.

14. The engine's `sanskrit` field is populated for 2 of 85 records. The gate was therefore tested on constructed inputs, not on real transcriptions. Re-run it once Sanskrit text is added to the extracts.

## What was not changed

- `indic-ganita-engine/` (frozen at v1.4.2) and the analysis code were read and not modified.
- No commit or push was made.

## Open items for the next step

- Supply scans or PDFs for the 13 texts (item 1).
- Decide the schema for engine extracts versus bronze rules (item 2).
- Add page references and Sanskrit text to the engine extracts to raise provenance coverage (item 3).
- Wire in Aksharamukha for IAST (item 5).
- Resolve the Cunningham source.
