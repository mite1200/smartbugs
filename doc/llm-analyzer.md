# Generic LLM Analyzer

SmartBugs includes a native, profile-driven LLM analyzer selected as the `llm` tool. It performs
one inference per SmartBugs task and emits both normal Smartbugs results and an detailed provenance
artifact. Benchmark loading, repetitions, voting, and evaluation metrics are intentionally outside
the analyzer.

## Configuration

Select `llm`, a Solidity file, JSON parsing, and an `llm` mapping in a SmartBugs configuration:

```yaml
tools: [llm]
files: [samples/SimpleDAO.sol]
json: true
processes: 1
runid: llm-example

llm:
  profile: reentrancy
  strategy: zero_shot
  provider:
    provider: openai
    model: gpt-6-luna
    api_key_env: OPENAI_API_KEY
    max_output_tokens: 16384
    request_timeout: 120
    retry_count: 2
    retry_backoff_seconds: 1
    reasoning:
      effort: medium
```

Run with:

```console
./smartbugs -c doc/examples/llm-openai.yaml
```

Secrets are read from the named environment variable and are never placed in configuration files.

## Providers

### Gemini

```yaml
provider:
  provider: gemini
  model: gemini-3.8-flash
  api_key_env: GEMINI_API_KEY
  max_output_tokens: 16384
  reasoning:
    effort: medium
```

The Gemini adapter uses `models.generateContent` over HTTP. For `gemini-3.8-flash`, reasoning
effort `low`, `medium` or `high` maps to
`generationConfig.thinkingConfig.thinkingLevel`. Sampling parameters `temperature`, `top_p` and
`top_k` must not be configured for this model and are rejected instead of silently changed.

### OpenAI

```yaml
provider:
  provider: openai
  model: gpt-6-luna
  api_key_env: OPENAI_API_KEY
  max_output_tokens: 16384
  reasoning:
    effort: medium
```

The native OpenAI adapter uses the Responses API. Its API style cannot be changed to Chat
Completions accidentally. `reasoning.effort` is passed through unchanged. When effort is not
`none`, explicitly configured `temperature` or `top_p` is rejected because those sampling
parameters must not be sent with the reasoning request. Native OpenAI requests explicitly send
`store: false`, so these independent single-turn responses are requested without API-side response
storage.

### OpenAI-compatible provider

```yaml
provider:
  provider: openai_compatible
  model: <provider model ID>
  base_url: <provider-specific endpoint>
  api_key_env: <provider-specific environment variable>
  api_style: chat_completions
```

`chat_completions` is the default for third-party compatible endpoints. Set `api_style: responses`
only when the selected provider implements this API. The endpoint format and model ID are entirely
provider-specific; the analyzer does not assume a `/v1` suffix or hardcode e.g. a qwen provider.

Provider-specific request values can be supplied without changing provider code:

```yaml
provider:
  options:
    request_body:
      frequency_penalty: 0.1
```

For Gemini, generation settings can additionally be placed under `options.generation_config`.
These values must not contain secrets - arbitrary option values are deliberately omitted from
provenance.

`temperature` and `max_output_tokens` are optional. If left out, they are absent from the provider
request. The example configurations use `max_output_tokens: 16384` as a conservative initial smoke
test value because reasoning tokens count toward output limits. The final experimental value should
be frozen after technical smoke tests based on truncation and API behavior, not classification
performance.

## Concurrency

SmartBugs `processes` is the actual bound on concurrent inference tasks. The optional provider
field `recommended_concurrency` is advised metadata only; the LLM package does not enforce e.g. a
cross-process semaphore. Therefore, configure `processes` no higher than the provider/model limit.

## Analysis profiles

Profiles define the task, preprocessing, accepted primary labels, prompts, auxiliary validation
and optional SmartBugs finding mapping. Generic providers, retries, preprocessing machinery and
JSON parsing do not contain vulnerability specific logic.

To add a profile:

1. Implement `AnalysisProfile` under `sb/llm/profiles/`
2. Give the profile and each strategy explicit version numbers
3. Register it in `sb/llm/profiles/__init__.py`
4. Add profile rendering, validation, and parser tests

The thesis reentrancy profile is `sb/llm/profiles/reentrancy.py` and supports:

- `zero_shot`
- `structured_reasoning`
- `few_shot`
- `structured_reasoning_few_shot` (available as the direct composition of the same procedure and
  demonstrations)

The first three are the intended primary experimental conditions. The fixed definition,
structured procedure, demonstrations, output constraints and version numbers all live in that
single profile module.

## Preprocessing

The generic preprocessor supports independent `remove_comments` and `add_line_numbers` options.
The reentrancy profile enables both; comment removal uses a state machine, preserves all newlines
and character positions and does not treat comment markers inside quoted Solidity strings as
comments. No other source transformation happens.

## Output parsing

The parser first attempts direct JSON parsing. It may then strip a surrounding Markdown code 
or extract the first balanced JSON object from surrounding text. It performs no semantic
reinterpretation and never invokes another model.

`classification_valid` and `schema_valid` are separate. A recognized primary label remains usable
when auxiliary fields are malformed. Invalid localization entries are discarded with warnings.
Malformed or schema-invalid model output is an observation and also is never retried.

## Retries

Only transient technical failures are retried, like e.g. HTTP 429, HTTP 5xx, connection errors and request
timeouts. Retries use a bounded exponential backoff with jitter and preserve the same prompt, model,
parameters and inference identity. Permanent HTTP/configuration failures are not retried.

## Results and provenance

Normal Smartbugs files are produced as usual. The task result directory additionally contains:

```text
llm-provenance.json
```

It records source hashes, profile/prompt versions, exact rendered prompt, raw HTTP response body,
raw model text, parsed result, validity flags, token usage, selected non-sensitive provider
metadata, retry history, runtime and optional opaque experiment metadata. Response bodies and model
text are retained in their original form, except that any exact occurrence of the configured API-key value is
replaced with `[REDACTED]` for security reasons if a provider improperly echoes it.

Authentication headers, API keys, cookies, environment values, arbitrary provider option values,
URL user information and URL query strings are not persisted.

## Limitations

- Provider capabilities and parameter names vary by exact model and endpoint. Use `reasoning` and
  provider options according to the selected provider documentation.
- Provider-native JSON-schema enforcement is not used by the reentrancy profil.
- SmartBugs process timeouts do not forcibly terminate an in-process HTTP call; configure
  `provider.request_timeout` explicitly.
- `recommended_concurrency` is not enforced; `processes` is the concurrency control.

## Tests

```console
.venv/bin/pytest tests/unit/llm -o addopts=''
.venv/bin/pytest tests/unit/test_tools.py tests/unit/test_settings.py \
  tests/unit/test_smartbugs.py tests/unit/test_analysis.py tests/unit/test_parsing.py -o addopts=''
```

All provider tests use mocks and do not make any paid API calls.