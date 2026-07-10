# Astro Engine MCP server -- a remote (Streamable HTTP) MCP server that any
# MCP-capable AI app can connect to: Claude (Connectors), ChatGPT (Developer
# Mode), the Gemini app (Gemini Spark), Gemini CLI, Cursor, VS Code, etc.
#
# One image runs unchanged on any free host that injects $PORT
# (Hugging Face Spaces=7860, Google Cloud Run=8080, Render=10000).
FROM python:3.12-slim

WORKDIR /app

# --- install the package + MCP extra ---------------------------------------
COPY pyproject.toml README.md ./
COPY astro_engine ./astro_engine
RUN pip install --no-cache-dir ".[mcp]"

# --- bundle a DE ephemeris kernel at build time ----------------------------
# de421 spans 1899-2053 (small, ~17 MB). For 1849-2150 build with
#   --build-arg KERNEL=de440s.bsp   (~32 MB).
ARG KERNEL=de421.bsp
RUN python -c "from skyfield.api import Loader; Loader('/app/kernels')('${KERNEL}')"
ENV ASTRO_KERNEL=/app/kernels/${KERNEL}

# --- runtime config --------------------------------------------------------
# Stateless JSON HTTP is the proxy-friendly mode for free/serverless hosts.
# Set ASTRO_MCP_TOKEN at deploy time to require an Authorization: Bearer token.
ENV MCP_TRANSPORT=streamable-http \
    ASTRO_MCP_STATELESS=1 \
    MCP_HOST=0.0.0.0 \
    PORT=7860

EXPOSE 7860

# --host 0.0.0.0 is required so the host proxy can reach the server; the port
# is picked up from $PORT automatically. The MCP endpoint is served at /mcp.
CMD ["astro-engine-mcp", "--http", "--host", "0.0.0.0"]
