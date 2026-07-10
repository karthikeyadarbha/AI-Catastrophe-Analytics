# Using Astro Engine from AI Apps

> Last verified: **2026-07-10**

This guide explains the practical ways to use Astro Engine from AI chat apps,
mobile apps, browsers, messaging apps, and custom websites. It covers both:

- **MCP integrations**, which can use the server already included in this
  repository.
- **Non-MCP integrations**, such as Claude Skills, Poe Server Bots, ChatGPT GPT
  Actions, Telegram bots, and a standalone web app.

AI products change their plans and menus frequently. Check the linked official
documentation before committing to a production deployment.

## What is ready in this repository?

Astro Engine currently exposes eight read-only MCP tools:

| Type | Tools |
|---|---|
| Snapshot | `vedic_chart`, `planet_positions`, `ascendant`, `panchanga`, `vimshottari_dasha`, `divisional_chart` |
| Event search | `find_planetary_event`, `events_in_range` |

The included Docker image, Hugging Face deployment recipe, `/mcp` endpoint,
optional bearer-token authentication, and local stdio server are ready to use.

The following are **integration ideas, not implemented adapters**:

- REST/OpenAPI API for ChatGPT Actions, Copilot Studio, and custom apps
- Claude Skill ZIP
- Poe Server Bot endpoint
- Telegram or WhatsApp bot
- Standalone web chat/PWA

These can all reuse the same `astro_engine` calculation library. The astronomy
must remain in Astro Engine; an LLM should only interpret the user's language,
select a function, and explain the returned JSON.

## Quick recommendation

| Goal | Recommended route | Why |
|---|---|---|
| Use what is already built | **MCP with Le Chat or Claude** | Only deployment and one-time connector setup are required |
| One known user, no server | **Claude Custom Skill** | Can bundle Python code and run it in Claude's code-execution environment |
| Public bot in an existing mobile AI app | **Poe Server Bot** | Works through Poe's web, iOS, and Android clients |
| User already prefers ChatGPT | **Custom GPT with Actions** | Familiar mobile/web UI; free users can use a published GPT within limits |
| User already uses Telegram | **Telegram bot** | The connector can be invisible to the end user |
| Maximum control and simple sharing | **Standalone PWA/web chat** | One URL or home-screen icon; no dependence on an AI app's connector policy |
| Company uses Microsoft 365 | **Copilot Studio** | Good governance and Teams distribution, but paid and enterprise-oriented |

## Common prompts

The assistant can translate Telugu, Hindi, or English into structured tool
arguments. Examples:

- `బుధుడు చివరిసారి ఎప్పుడు అస్తమించాడు?`
  ("When did Budha last become combust?")
- "When is Saturn next retrograde?"
- "Show all Jupiter sign changes from 2024 to 2030."
- "Calculate today's panchanga for Hyderabad."
- "Give me the D9 chart for 1990-01-01 10:30 IST in Delhi."

Event searches such as combustion, retrograde, and geocentric ingress do not
need birth details. Charts, ascendant, houses, and local panchanga require a
datetime and location.

---

## Option 1: Remote MCP server

This is the shortest path because it uses the server already in the repository.

### Deploy once

From `astro-engine/`, build a Hugging Face Space folder:

```powershell
pwsh deploy/huggingface/build_space.ps1 -Out ..\astro-mcp-space
```

Create a Docker Space, push the generated folder, and use:

```text
https://<hugging-face-user>-<space-name>.hf.space/mcp
```

The Docker image bundles `de421.bsp`, covering approximately 1899-2053. It can
also run on Cloud Run, Render, or another Docker host that supplies `$PORT`.

Free Hugging Face hardware sleeps when idle. The first request after sleep can
take roughly 30-60 seconds. That is acceptable for personal use, but choose an
always-on host for clients with short webhook or tool timeouts.

### Mistral Le Chat

1. Open Le Chat on the web or in its mobile app.
2. Open **Connectors** or **Connections**.
3. Add a custom connector.
4. Paste the `/mcp` URL.
5. If supported by the client and configured on the server, provide the bearer
   token; otherwise deploy a read-only endpoint without static-token auth.
6. Enable the connector in the conversation and ask a normal question.

This is the most direct mobile MCP route where custom connectors are available
to the account.

### Claude

1. Open `claude.ai` in a browser.
2. Go to **Settings > Connectors**.
3. Add a custom connector and paste the `/mcp` URL.
4. Save it and enable it in the conversation.
5. Open the same account in the Claude mobile app and confirm that the connector
   is available before handing the phone to the end user.

Claude's connector limits depend on the account plan. The Free plan has had a
one-custom-connector limit during beta.

### Gemini Spark

Google's official custom-app route for the consumer Gemini app currently uses
MCP.

1. On a computer, open `gemini.google.com`.
2. Go to **Settings & help > Connected Apps**.
3. Under custom apps for Spark, add the `/mcp` URL.
4. Complete any authentication steps.
5. The connected app then works in Gemini Spark on web and mobile.

Current restrictions include: age 18+, United States, personal Google account,
English, Gemini Spark access, and Keep Activity enabled. A custom app can only
be added from the web interface.

### ChatGPT custom MCP

ChatGPT's Developer Mode can connect to remote MCP servers on supported paid
plans. This route has historically been web-oriented and should not be selected
for a mobile-only user until the exact account and mobile client are tested.
For broader ChatGPT mobile usage, use the non-MCP **GPT Actions** route below.

### Local desktop clients

Claude Desktop, Cursor, VS Code, and Gemini CLI can spawn the server over stdio:

```json
{
  "mcpServers": {
    "astro-engine": {
      "command": "astro-engine-mcp",
      "env": {
        "ASTRO_KERNEL": "de421.bsp"
      }
    }
  }
}
```

---

## Option 2: Claude Custom Skill (non-MCP)

Claude Skills can contain instructions, files, dependencies, and executable
Python scripts. They are available on Free, Pro, Max, Team, and Enterprise
plans when **Code execution and file creation** is enabled.

This route can avoid a remote server entirely:

```text
astro-engine-skill.zip
└── astro-engine-skill/
    ├── skill.md
    ├── astro_engine/
    └── scripts/
        └── astro_query.py
```

Recommended design:

1. Bundle this repository's `astro_engine` package in the Skill.
2. Declare `pyswisseph` as a dependency.
3. Use the Swiss/Moshier backend, which needs no external ephemeris file for
   modern dates.
4. Add one strict command-line script that accepts JSON and returns JSON.
5. In `skill.md`, tell Claude when to run that script and never to reproduce the
   astronomical calculation itself.
6. Zip the folder.
7. In Claude, enable code execution, then go to
   **Customize > Skills > + Create skill > Upload a skill**.

Important limitations:

- Individual Free/Pro/Max users upload their own private copy. Organization-wide
  distribution requires Team or Enterprise.
- Claude's official documentation confirms use in `claude.ai`; test the exact
  iOS/Android workflow before relying on it for a mobile-only user.
- Do not bundle API keys or user secrets.
- A JPL Skill would need a compatible kernel-download or bundled-kernel strategy.
  Swiss/Moshier is simpler for this route.

Official documentation:

- [Use Skills in Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
- [Create custom Skills](https://support.claude.com/en/articles/12512198-creating-custom-skills)

---

## Option 3: Poe Server Bot (non-MCP)

Poe supports custom bots backed by any publicly reachable server implementing
the Poe Server Bot protocol. Once registered, a public bot can be used from
Poe's web and mobile clients.

A suitable architecture is:

```text
Poe app
  -> Poe Server Bot adapter
     -> Poe model interprets the prompt and returns strict arguments
     -> Astro Engine performs the deterministic calculation
     -> Poe model formats the JSON result for the user
```

Poe server bots can call Poe-hosted models such as GPT or Claude. Those calls
are charged to the points of the user chatting with the bot, rather than to a
separate model API key owned by the bot creator.

Setup outline:

1. Add `fastapi-poe`.
2. Implement `PoeBot.get_response()`.
3. Declare the selected interpretation model in `server_bot_dependencies`.
4. Validate its structured output before calling Astro Engine.
5. Stream a response immediately.
6. Deploy the endpoint.
7. On Poe's bot-creation page, choose **Server bot**, enter the endpoint and
   generated access key, and select public or private visibility.

Poe requires the initial response within **5 seconds**. A sleeping Hugging Face
Space can miss that deadline, so use an always-on service or a host whose cold
start is safely below the limit.

Official documentation:

- [Poe Server Bot quick start](https://creator.poe.com/docs/server-bots/quick-start)
- [Poe protocol and limits](https://creator.poe.com/docs/server-bots/poe-protocol-specification)
- [Calling other Poe bots](https://creator.poe.com/docs/server-bots/server-bots-functional-guides)

---

## Option 4: ChatGPT Custom GPT with Actions (non-MCP)

GPT Actions call an ordinary HTTPS API described by OpenAPI. They do not use
MCP.

This requires a new REST layer:

```text
ChatGPT GPT Action
  -> POST /v1/events/find
  -> POST /v1/events/range
  -> POST /v1/chart
  -> Astro Engine
```

Suggested setup:

1. Add a small FastAPI adapter around the existing eight tool functions.
2. Expose only typed, read-only `POST` endpoints.
3. Publish `openapi.json` over HTTPS.
4. Add rate limiting and either API-key or OAuth authentication.
5. Create a GPT on the ChatGPT website.
6. Under **Actions**, import the OpenAPI schema and configure authentication.
7. Add clear instructions telling the GPT when each operation applies.
8. Test, then share by link or publish if eligible.

Creating or editing a GPT requires a supported paid account and is web-only.
Free users can use public/shared GPTs within their usage limits. GPTs can be
used in the ChatGPT mobile apps. Public GPTs with Actions need a valid privacy
policy URL.

Official documentation:

- [Configure GPT Actions](https://help.openai.com/en/articles/9442513)
- [GPTs in ChatGPT](https://help.openai.com/en/articles/8554407-gpts-in-chatgpt)
- [ChatGPT Free Tier FAQ](https://help.openai.com/en/articles/9275245-using-chatgpt-s-free-tier-faq)

---

## Option 5: Telegram bot (non-MCP)

Telegram is often the simplest experience for a non-technical user: send them a
bot link and let them type or speak normally.

```text
Telegram
  -> webhook or long-polling adapter
  -> intent parser
  -> Astro Engine
  -> formatted reply
```

Setup outline:

1. Create a bot with Telegram's `@BotFather`.
2. Store the bot token only as a deployment secret.
3. Implement webhook or long-polling message handling.
4. For fixed commands, use deterministic parsing.
5. For unrestricted Telugu/Hindi/English prompts, use an LLM with structured
   function calling, then validate every argument.
6. Call Astro Engine and send the result back.

The Telegram Bot API is free. The model used for natural-language
interpretation may have a cost. A webhook needs public HTTPS; long polling can
run from a private machine but must remain online.

Official documentation:

- [Telegram Bot API](https://core.telegram.org/bots/api)

---

## Option 6: WhatsApp bot (non-MCP)

WhatsApp gives the most familiar experience for many users, but it has more
administrative and pricing overhead than Telegram:

1. Create a Meta business portfolio.
2. Configure a WhatsApp Business Account and phone number.
3. Create a Meta app and enable WhatsApp Cloud API.
4. Register a webhook.
5. Add the same intent-parser and Astro Engine adapter used for Telegram.
6. Submit message templates where required.

WhatsApp pricing and template rules vary by message category and region. This
is not the best first prototype, but it can be a good production channel after
the Telegram or web workflow has proven useful.

Official documentation:

- [WhatsApp Business Platform](https://developers.facebook.com/documentation/business-messaging/whatsapp/about-the-platform)
- [WhatsApp pricing](https://developers.facebook.com/documentation/business-messaging/whatsapp/pricing)

---

## Option 7: Standalone web chat/PWA (non-MCP)

A small progressive web app offers the most control:

- one shareable HTTPS link;
- **Add to Home Screen** on Android/iOS;
- voice input and local-language UI;
- no connector setup by the end user;
- your choice of LLM provider or deterministic command parsing.

Recommended architecture:

```text
PWA
  -> authenticated chat API
     -> LLM function calling for intent extraction
     -> REST adapter or direct Astro Engine call
     -> auditable JSON result
```

This is not an installation inside ChatGPT, Gemini, or Claude. It is your own
app, so you own hosting, model costs, privacy, logging, and updates.

For a self-hosted ready-made UI, LibreChat or Open WebUI can provide an
installable web experience. They still require server administration and an
LLM provider or local model.

---

## Option 8: Developer-API function calling (non-MCP)

OpenAI, Anthropic, Gemini, and Mistral developer APIs can all select custom
functions. This does **not** add the function to their normal consumer apps.
You must build or host the surrounding app:

1. Describe Astro Engine operations using the provider's tool schema.
2. Send the user's message and schemas to the model.
3. Validate the model's proposed arguments.
4. Execute Astro Engine on your server.
5. Return the result to the model for explanation.

Use this route when building the PWA, Telegram bot, WhatsApp bot, or another
custom product.

---

## Option 9: Microsoft Copilot Studio (non-MCP)

Copilot Studio can import an OpenAPI **2.0** definition as a custom connector
and expose the operations as tools. It can publish to Teams, Microsoft 365, or
web channels.

This requires the same REST adapter as GPT Actions, plus a paid Copilot Studio
deployment. The trial can be used to experiment but cannot publish. Choose this
only for a Microsoft-oriented organization, not for a free personal setup.

Official documentation:

- [Create a custom connector from OpenAPI](https://learn.microsoft.com/en-us/connectors/custom-connectors/define-openapi-definition)

---

## Conditional or unsupported consumer routes

| Product | Current conclusion |
|---|---|
| **Gemini classic Gems** | Instructions and knowledge only; they cannot directly call an arbitrary Astro Engine REST API |
| **Gemini Labs/Opal Gems** | Useful for AI workflows, but no verified arbitrary REST execution path in the consumer Gem; Gemini developer function calling requires a separate app |
| **Gemini Spark custom apps** | Supported through MCP, covered above |
| **Mistral developer Agents** | Function calling is available through developer APIs, but arbitrary non-MCP tools are not a verified publishing route into the normal Le Chat consumer app |
| **Claude connectors** | External-service connectors use MCP; Claude Custom Skills are the non-MCP exception |
| **DeepSeek / Perplexity** | No verified end-user mechanism for publishing this custom tool inside the official consumer app/site |
| **Coze** | OpenAPI plugins and messaging channels appear promising, but current global availability, pricing, and channel support should be verified in the target account before implementation |
| **Slack / Discord** | Technically straightforward bot channels, but best only when the intended users already use that workspace or community |

## Security and privacy checklist

1. Keep all routes read-only unless a future use case truly needs writes.
2. Never commit API keys, model keys, bot tokens, or bearer tokens.
3. Validate dates, coordinates, planet names, event types, and date-range caps
   server-side; never trust model-generated arguments.
4. Add authentication and rate limits before publishing a permanent endpoint.
5. Prefer OAuth for public multi-user connectors. Static bearer tokens are
   suitable only when the chosen client can store them safely.
6. Log operation names and timing, but avoid logging birth details or complete
   conversations.
7. Explain which AI vendor receives the user's prompt and location data.
8. Treat uploaded Claude Skills like software: audit every script and dependency.
9. Keep astronomical computation deterministic. Do not let the LLM invent
   longitudes, dates, or event boundaries.
10. Do not present astrological correlations as proven earthquake forecasts or
    use them for public-safety decisions. Scientific and seismic claims require
    preregistered tests, appropriate null models, and independent validation.

## Recommended rollout

1. **Now:** deploy and test the existing MCP server with Le Chat or Claude.
2. **Next:** package a Swiss/Moshier-based Claude Skill for a server-free trial.
3. **Then:** add a Poe Server Bot for a public mobile AI experience.
4. **Shared foundation:** build one FastAPI REST/OpenAPI adapter. It unlocks GPT
   Actions, Copilot Studio, the PWA, and messaging bots.
5. **After usage is proven:** add Telegram; add WhatsApp only if users explicitly
   need it.
6. Move from sleeping free hosting to an always-on service when latency or
   reliability matters.
