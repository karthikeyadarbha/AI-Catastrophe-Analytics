---
title: Astro Engine MCP
emoji: 🪐
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Vedic astrology MCP server (charts, panchanga, dasha, varga)
---

# Astro Engine MCP — Hugging Face Space

A free **remote MCP server** exposing the
[`astro_engine`](https://github.com/Celestial-Influence-Research/Astral-DataScience-platform)
Vedic-astrology engine over the Model Context Protocol (Streamable HTTP).

> Free CPU Spaces sleep after inactivity, so the first request after an idle
> period can take roughly 30-60 seconds while the container starts.

**MCP endpoint:** `https://<your-username>-<space-name>.hf.space/mcp`

Connect it from Claude (Settings → Connectors), ChatGPT (Settings → Apps →
Developer Mode), or the Gemini app (Settings → Connected Apps → Gemini Spark).

Tools: `vedic_chart`, `planet_positions`, `ascendant`, `panchanga`,
`vimshottari_dasha`, `divisional_chart`, `find_planetary_event`,
`events_in_range`.

> This Space runs the Dockerfile in this repo. It bundles the `de421.bsp`
> ephemeris (dates 1899–2053) at build time, so no runtime network is needed.
> The server is read-only (astronomical computation on public ephemeris data).
> To require a token, set an `ASTRO_MCP_TOKEN` **Secret** on the Space.
