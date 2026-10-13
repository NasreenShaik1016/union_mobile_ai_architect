# Union Mobile AI Customer Support

A structured, Docker-ready starter for a stateful multi-agent customer-support workflow using Streamlit, LangGraph, LLMs, and Chroma-based policy retrieval.


## Architecture

```text
app.py                         Streamlit presentation layer
support_app/
  config.py                    Environment / Streamlit settings and policy constants
  observability.py             UTC timestamps, logging, sensitive-field redaction
  models.py                    LLM and embedding client construction
  repositories.py              CSV data, policy index, customer memory, local queues
  runtime.py                   Resource composition root
  state.py                     Typed LangGraph state and node projections
  guardrails.py                Input/output safety classifiers
  tools.py                     Verification-gated support tools
  agents.py                    Specialist prompts and ReAct agent construction
  workflow.py                  Workflow nodes, routing, graph and session orchestration
data/
  accounts.csv                 Synthetic demo accounts only
  plans.csv                    Synthetic demo plans only
  policy_kb.pdf                Demo policy knowledge base
  customer_memory.json         Local development file; ignored by Git
tests/                         Unit tests for state projection and redaction
Dockerfile
requirements.txt
```

## Run locally

Use Python 3.12.

### PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in your PowerShell session before starting Streamlit:

```powershell
$env:OPENAI_API_KEY = "your-key"
# Optional:
$env:OPENAI_API_BASE = "https://your-openai-compatible-endpoint/v1"
streamlit run app.py
```

The app reads environment variables directly. `.env` is an example/template only; it is not automatically loaded. For Streamlit Community Cloud, set the same values in **App settings → Secrets**.

## Run tests

```powershell
python -m pytest -q
```

Tests are intentionally independent of the live LLM endpoint.

## Run with Docker

```powershell
docker build -t union-mobile-support .
docker run --rm -p 8501:8501 `
  -e OPENAI_API_KEY="your-key" `
  -e DATA_DIR="/app/data" `
  union-mobile-support
```

Open `http://localhost:8501`.

For a writable persistent data volume, mount a directory and set `MEMORY_FILE`, `TICKETS_FILE`, and `ESCALATIONS_FILE` to paths in that volume. Do not put secrets in the image or source control.

## Configuration

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required model/embedding credential |
| `OPENAI_API_BASE` | Optional OpenAI-compatible base URL |
| `MODEL_NAME` | Chat model; defaults to `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Embedding model; defaults to `text-embedding-3-small` |
| `DATA_DIR` | Directory containing demo CSV/PDF files |
| `MEMORY_FILE` | Local customer-memory JSON path |
| `TICKETS_FILE` | Local ticket queue JSON path |
| `ESCALATIONS_FILE` | Local escalation queue JSON path |
| `LANGCHAIN_TRACING_V2` | Set `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | LangSmith credential when tracing is enabled |

## Important production hardening backlog

1. **Identity and secrets:** replace CSV PINs with an identity provider and salted password/PIN verification; use managed identity/Key Vault for cloud secrets; enforce rate limits and account lockout in a shared store.
2. **Storage and concurrency:** replace CSV/JSON adapters with Azure SQL/Cosmos DB and a durable queue such as Azure Service Bus. Local JSON queues are not safe as distributed multi-instance queues.
3. **Real side effects:** connect `raise_network_ticket` and `escalate_to_human` to an authenticated ticketing/CRM system. The included adapters only write local queue records; they do not notify a human team.
4. **Guardrail policy:** keep security checks fail-closed, add adversarial regression tests, measure false positives/negatives, and avoid relying on an LLM classifier as the only security boundary.
5. **RAG quality:** evaluate retrieval recall, source/version metadata, groundedness, and citation correctness against labeled test cases; define policy-document refresh and re-indexing.
6. **Observability:** export structured logs and metrics to Azure Monitor / Application Insights; redact personal data, establish retention controls, and add correlation IDs.
7. **Reliability:** add timeouts, bounded retries with backoff, rate limits, circuit breakers, health/readiness checks, concurrency/load tests, and model-cost budgets.
8. **Evaluation:** maintain a versioned dataset covering intent routing, authorization, prompt injection, policy grounding, escalation precision/recall, latency, and token cost.
9. **Privacy:** establish customer-memory consent, access controls, retention/deletion workflows, and encryption at rest and in transit.
10. **Deployment:** use CI/CD, vulnerability scanning, dependency lock files, container scanning, private networking, TLS, and separate dev/test/prod configuration.

## Design notes

- Specialist tools enforce verification at the tool boundary; prompt instructions alone are not treated as authorization.
- Wrong-PIN attempts are scoped to the current Streamlit browser session to avoid one visitor locking out another. For real deployments, enforce limits in a shared identity/rate-limit service.
- Input and output LLM classifier errors fail closed in this starter. A production implementation should pair that behavior with explicit timeout handling, fallback responses, alerts, and service-level objectives.
- The workflow and evaluation metrics are not claims of production performance. Run a reproducible test suite before publishing numerical accuracy or security claims.
