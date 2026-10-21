"""
Astral Ephemeris FastMCP Server Entrypoint.
Delegates to and re-exports all endpoints from the packaged astral_engine.server module.
"""
from astral_engine.server import (
    mcp,
    main,
    get_ephemeris_snapshot,
    get_planet_timeline,
    query_celestial_intervals,
    get_angular_relationship,
    describe_feature_options,
    enrich_csv_dataset,
    batch_enrich_catalog,
)

if __name__ == "__main__":
    main()
