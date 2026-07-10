# Assemble a ready-to-push Hugging Face Space folder for the Astro Engine MCP
# server. Run from the astro-engine/ directory:
#
#   pwsh deploy/huggingface/build_space.ps1 -Out ..\astro-mcp-space
#
# Then create a Docker Space at https://huggingface.co/new-space and push:
#
#   cd ..\astro-mcp-space
#   git init; git add .; git commit -m "Astro Engine MCP"
#   git remote add origin https://huggingface.co/spaces/<user>/<space>
#   git push -u origin main
#
# The Space builds the Dockerfile and serves the MCP endpoint at /mcp.
param(
    [string]$Out = "..\astro-mcp-space"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # -> astro-engine/

New-Item -ItemType Directory -Force -Path $Out | Out-Null
Copy-Item "$root\Dockerfile"       $Out -Force
Copy-Item "$root\.dockerignore"    $Out -Force
Copy-Item "$root\pyproject.toml"   $Out -Force
Copy-Item "$root\astro_engine"     $Out -Recurse -Force
# The Space README (with the required YAML front-matter) becomes README.md:
Copy-Item "$PSScriptRoot\README.md" "$Out\README.md" -Force

Write-Host "Space folder ready at: $Out"
Write-Host "Files:"; Get-ChildItem $Out | ForEach-Object { "  $($_.Name)" }
