# llama-server profiler

A small, dependency-free benchmark for a running
[`llama-server`](https://github.com/ggml-org/llama.cpp). It measures server
prefill and decode throughput, client-observed latency, and speculative decoding
acceptance across deterministic short, medium, and long workloads.

The profiler runs requests sequentially against one loaded model. It is intended
for comparing model, quantization, and server configurations under consistent
single-request conditions, not for measuring concurrent serving capacity.

## Requirements

- Python 3.10 or newer
- A reachable `llama-server` with one model loaded
- The native `/health`, `/props`, `/apply-template`, `/tokenize`, and streaming
  `/completion` endpoints

Only the Python standard library is used, so there are no packages to install.

## Quick start

Start `llama-server` separately, then run:

```bash
./profile_llama_server.py
```

The default server URL is `http://127.0.0.1:8080`. To use another server and
save the complete report:

```bash
./profile_llama_server.py \
  --url http://192.168.1.20:8080 \
  --repeats 10 \
  --warmups 2 \
  --json profile.json
```

If the server requires bearer authentication, set the API key in the
environment:

```bash
LLAMA_API_KEY=secret ./profile_llama_server.py --url https://server.example
```

Credentials are not accepted in `--url`.

## Workloads

The built-in tasks use fixed prompts and output lengths:

| Task | Workload | Output tokens |
| --- | --- | ---: |
| `short` | Support-ticket classification | 32 |
| `medium` | Incident-log summary with a longer prompt | 128 |
| `long` | Python implementation and explanation | 512 |

All three run by default. Select one or more with a comma-separated list:

```bash
./profile_llama_server.py --tasks short,long
```

The profiler checks the formatted prompt and requested output against the
server's context size before starting. Requests use temperature `0`, seed `42`,
prompt caching disabled, and EOS ignored so that each successful run produces
the requested number of tokens. Warmups are excluded from the report.

## Reports and summary replay

`--json PATH` writes a schema-v2 report containing sanitized server metadata,
configuration, every successful run, failures, and the aggregate summary. The
prompts and generated text are not saved.

Reprint the same terminal summary later without contacting the server:

```bash
./profile_llama_server.py --summary-from-json profile.json
```

Summary replay accepts schema-v2 reports only. It is mutually exclusive with
`--json`; older schema-v1 reports retain their stored measurements but cannot be
replayed by this command.

## Metrics

- **TTFT** is client-observed time to the first streamed token.
- **E2E latency** is client-observed time until the terminal stream event.
- **Server prefill TPS** and **server decode TPS** come from the timing fields in
  the terminal `/completion` event.
- **Client decode TPS** is calculated from the arrival interval between the
  first and last streamed tokens.
- **p50** is the median across successful runs; **p95** uses the nearest-rank
  percentile.
- **Overall server throughput** is token-weighted: total processed tokens
  divided by total server time, calculated separately for prefill and decode.
- **Draft acceptance** is printed when the server supplies speculative decoding
  token counts.

Failed measured runs are reported separately and are omitted from aggregate
statistics.

## Command-line options

```text
--url URL                 llama-server base URL
--repeats N               measured repetitions per task (default: 3)
--warmups N               excluded warmup runs (default: 1)
--timeout SECONDS         timeout for each request (default: 600)
--ready-timeout SECONDS   maximum model-loading wait (default: 600)
--tasks NAMES             comma-separated short,medium,long selection
--json PATH               save per-run data and summaries
--summary-from-json PATH  print a saved schema-v2 summary and exit
```

Run `./profile_llama_server.py --help` for the authoritative CLI help.

## Exit codes

| Code | Meaning |
| ---: | --- |
| `0` | Benchmark or report replay completed successfully |
| `1` | Benchmark completed with one or more failed measured runs |
| `2` | Invalid arguments, server/protocol error, or invalid report |

## Tests

Run the test suite with:

```bash
python3 -m unittest discover -s tests -v
```

The integration-style tests open a temporary server on the loopback interface.
