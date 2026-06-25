# Security

## Credentials

- Never commit API keys to Python, YAML, Dockerfiles, examples, or notebooks.
- Pass `RAG_LAW_API_KEY` through the process environment or a secret manager.
- The historical project contained a plaintext credential. That credential must be
  revoked by its owner before any remote LLM test is performed.
- `.env` is ignored; `.env.example` must contain names only, never real values.

## Data and tools

- The current retrieval and ingestion tools are read-only by default.
- Historical local indexes and metadata are treated as immutable source artifacts.
- Recovered or aligned data must be written to new versioned paths and must not
  overwrite historical assets.
