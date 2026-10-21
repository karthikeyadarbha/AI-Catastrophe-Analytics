# Astral Ephemeris & Astrological State Machine MCP Server

High-performance Model Context Protocol (MCP) server for Swiss Ephemeris calculations, temporal state transitions, Jaimini Chara Karakas, and composite interval algebra queries.

## Architecture

```
astral-mcp-server/
├── pyproject.toml              # Dependencies & packaging metadata
├── server.py                   # Root wrapper entrypoint
├── test_server.py              # Verification suite
└── astral_engine/
    ├── server.py               # FastMCP server exposing tool endpoints
    ├── constants.py            # Planet codes, signs, nakshatras, combustion limits, ayanamsas
    ├── utils.py                # Julian Day conversions, DMS formatting, angular separation
    ├── core/
    │   ├── models.py           # Planet Enum, MotionState, DateRange, Period, PlanetPosition
    │   └── algebra.py          # Allen's Interval Algebra (intersection, union, inversion)
    ├── calcs/
    │   ├── abstract_calc.py    # Abstract base calculator
    │   ├── sidereal_calc.py    # Swiss Ephemeris wrapper + Topocentric AltAz support
    │   └── chara_karaka.py     # Jaimini 7/8 Chara Karaka rank sorter
    ├── algorithms/
    │   ├── bisection.py        # Continuous bisection and discrete binary search engine
    │   ├── motion_tracker.py   # Root-finding on speed for retrograde/direct stations
    │   ├── combustion_tracker.py # Root-finding on solar distance for combustion windows
    │   ├── transit_tracker.py  # Binary search on Rāśi & Nakṣatra boundary transitions
    │   └── aspect_engine.py    # Stelliums, conjunctions, lunar phase, and syzygy
    └── query/
        └── evaluator.py        # Declarative multi-condition temporal query compiler
```

## Available MCP Tools

1. **`get_ephemeris_snapshot(datetime_str, ayanamsa='lahiri', lat=0.0, lon=0.0, elev_m=0.0)`**
   - Full 9-planet celestial state (degrees, signs, nakshatras, padas, speeds, retrograde status, combustion, and Jaimini Chara Karakas).
   - Treats all celestial bodies uniformly with identical attributes.
   - Local horizon elevation (`altitude_deg`) and visibility when `lat`/`lon` are provided.

2. **`get_planet_timeline(planet, start_date, end_date, ayanamsa='lahiri')`**
   - Chronological stream of all transitions (Rāśi entry, Nakṣatra entry, retrograde stations) resolved down to microseconds via binary search for any body.

3. **`query_celestial_intervals(start_date, end_date, filters, operator='AND', min_duration_hours=0.0, ayanamsa='lahiri')`**
   - Universal declarative query engine. Evaluates composite conditions across arbitrary planet pairs (e.g., *Jupiter-Saturn Conjunction*, *Mars-Sun Opposition*, *Mars Retrograde*, *4-Planet Stellium*) using Allen's Interval Algebra.

4. **`get_angular_relationship(body1, body2='Sun', datetime_str='...', orb_deg=6.0, ayanamsa='lahiri')`**
   - Pure pairwise geometric relationship between **any arbitrary pair of celestial bodies** (e.g. Moon & Sun, Jupiter & Saturn, Mars & Sun, Venus & Sun).
   - Computes angular separation (0°–180°), longitudinal difference (0°–360°), aspect classification (Conjunction, Sextile, Square, Trine, Opposition), collinear alignment / syzygy flag, and illumination percentage.

5. **`batch_enrich_catalog(records, feature_spec=None, preset='ml_cyclical', ayanamsa='lahiri')`**
   - Bulk-enrichment of in-memory JSON records with dynamic user-specified or AI-decided features.

6. **`describe_feature_options()`**
   - Returns the complete catalog of all feature options, attributes, cyclic transforms, pairwise configurations, and domain presets for AI decision-making.

7. **`enrich_csv_dataset(input_csv_path, output_csv_path=None, feature_spec=None, preset=None, calculation_mode='fast', ...)`**
   - High-throughput single-threaded CSV enrichment pipeline. Auto-detects datetime and coordinate columns, populating custom or preset features across tens of thousands of rows.
   - `calculation_mode='fast'`: Adaptive trajectory grid interpolation + SIMD (default).
   - `calculation_mode='exact'`: Exact deduplicated Swiss Ephemeris evaluations.

---

## Client Setup & Configuration

### Prerequisites
- **Python 3.10 or 3.11** (recommended for pre-compiled Swiss Ephemeris wheels on Windows, macOS, and Linux).
- [uv](https://docs.astral.sh/uv/) installed (recommended for zero-setup execution):
  ```bash
  # Windows (PowerShell)
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

---

### 1. Claude Desktop (`claude_desktop_config.json`)
Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "astral-ephemeris": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/ABSOLUTE/PATH/TO/astral-mcp-server",
        "astral-ephemeris"
      ]
    }
  }
}
```

---

### 2. Cursor (`Settings -> MCP`)
Add a new MCP server with type `command`:
- **Name**: `astral-ephemeris`
- **Command**: `uv run --directory /ABSOLUTE/PATH/TO/astral-mcp-server astral-ephemeris`

---

### 3. Google Antigravity IDE (`~/.gemini/config/mcp_config.json`)
```json
{
  "mcpServers": {
    "astral-ephemeris": {
      "command": "uv",
      "args": [
        "run",
        "--no-project",
        "--link-mode=copy",
        "--python", "3.11",
        "--with", "mcp<2",
        "--with", "pyswisseph>=2.10.3.2",
        "--with", "numpy>=1.24.0",
        "--with", "pytz>=2023.3",
        "--with", "python-dateutil>=2.8.2",
        "/ABSOLUTE/PATH/TO/astral-mcp-server/server.py"
      ]
    }
  }
}
```

---

### 4. Running directly from GitHub via `uvx`
Once pushed to a GitHub repository, anyone can run it without cloning:

```json
{
  "mcpServers": {
    "astral-ephemeris": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/<YOUR_USER>/astral-mcp-server.git",
        "astral-ephemeris"
      ]
    }
  }
}
```

---

## Verification & Tests

```bash
uv run --python 3.11 test_server.py
```
