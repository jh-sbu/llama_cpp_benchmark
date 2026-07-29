#!/usr/bin/env python3
"""Profile a single-model llama-server using its native streaming API."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 2
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_READY_TIMEOUT_SECONDS = 600.0
READY_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_REPEATS = 3
DEFAULT_WARMUPS = 1
FIXED_SEED = 42


class ProfileError(RuntimeError):
    """Raised when the server cannot provide a valid benchmark measurement."""


class HttpStatusError(ProfileError):
    """An HTTP error retaining its status code for readiness handling."""

    def __init__(self, method: str, url: str, status: int, detail: str) -> None:
        self.method = method
        self.url = url
        self.status = status
        self.detail = detail
        super().__init__(f"{method} {url} returned HTTP {status}: {detail}")


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    description: str
    prompt: str
    completion_tokens: int


@dataclass(frozen=True)
class PreparedTask:
    task: BenchmarkTask
    formatted_prompt: str
    estimated_prompt_tokens: int


@dataclass(frozen=True)
class RunResult:
    task: str
    repeat: int
    expected_completion_tokens: int
    prompt_tokens: int
    processed_prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int
    streamed_tokens: int
    ttft_ms: float
    total_latency_ms: float
    prefill_ms: float
    server_prefill_tps: float
    decode_ms: float
    server_decode_tps: float
    client_decode_tps: float | None
    stop_type: str | None
    draft_tokens: int | None = None
    draft_accepted_tokens: int | None = None
    draft_acceptance_percent: float | None = None


def _incident_log() -> str:
    services = ("api", "worker", "database", "cache")
    regions = ("us-east", "us-west", "eu-central")
    symptoms = (
        "latency exceeded the service objective",
        "the error rate rose above two percent",
        "a health check restarted one replica",
        "the retry queue grew and then recovered",
    )
    actions = (
        "traffic was shifted to healthy replicas",
        "the deployment was paused",
        "connection limits were raised temporarily",
        "the on-call engineer cleared stale work",
    )
    lines = []
    for index in range(24):
        hour = 9 + index // 6
        minute = (index * 7) % 60
        service = services[index % len(services)]
        region = regions[index % len(regions)]
        symptom = symptoms[index % len(symptoms)]
        action = actions[(index + 1) % len(actions)]
        lines.append(
            f"2026-04-18 {hour:02d}:{minute:02d}Z | {region} | {service} | "
            f"{symptom}; {action}."
        )
    return "\n".join(lines)


def default_tasks() -> list[BenchmarkTask]:
    """Return deterministic workloads with short, medium, and long decodes."""
    return [
        BenchmarkTask(
            name="short",
            description="Short support-ticket classification",
            completion_tokens=32,
            prompt=(
                "Classify each support ticket as BILLING, ACCESS, BUG, or FEATURE. "
                "Return only one line per ticket in the form ID=LABEL.\n"
                "A: The invoice contains a duplicate charge.\n"
                "B: My password reset link has expired.\n"
                "C: Exporting a report produces an empty file.\n"
                "D: Please add scheduled report delivery."
            ),
        ),
        BenchmarkTask(
            name="medium",
            description="Summary of a moderately long incident log",
            completion_tokens=128,
            prompt=(
                "Summarize the incident log below for an engineering handoff. "
                "State the timeline, affected services and regions, mitigations, "
                "and three concrete follow-up actions. Be concise but complete.\n\n"
                f"{_incident_log()}"
            ),
        ),
        BenchmarkTask(
            name="long",
            description="Long Python implementation and explanation",
            completion_tokens=512,
            prompt=(
                "Write a complete Python implementation of a bounded, thread-safe "
                "work queue using only the standard library. It must support blocking "
                "put/get operations with optional timeouts, task_done and join "
                "semantics, graceful closure, and clear exceptions for closed-queue "
                "operations. Include type hints, docstrings, a usage example, focused "
                "unit tests, and a short explanation of the synchronization invariants "
                "and edge cases."
            ),
        ),
    ]


def normalize_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("URL must be an absolute http:// or https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise argparse.ArgumentTypeError(
            "credentials are not allowed in --url; use LLAMA_API_KEY"
        )
    if parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("--url must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    normalized = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path, "", "")
    )
    return normalized


def _error_message(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        return "empty response body"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text[:500]
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if error:
        return str(error)
    return text[:500]


class LlamaServerClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "llama-server-profiler/1",
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        accept: str = "application/json",
    ) -> urllib.request.Request:
        data = None
        headers = dict(self.headers)
        headers["Accept"] = accept
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        return urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )

    def _open(
        self,
        request: urllib.request.Request,
        *,
        timeout_seconds: float | None = None,
    ) -> Any:
        effective_timeout = (
            self.timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        try:
            return urllib.request.urlopen(request, timeout=effective_timeout)
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                detail = _error_message(exc.read())
            finally:
                exc.close()
            raise HttpStatusError(
                request.method, request.full_url, status, detail
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise ProfileError(
                f"{request.method} {request.full_url} failed: {reason}"
            ) from exc
        except TimeoutError as exc:
            raise ProfileError(
                f"{request.method} {request.full_url} timed out after "
                f"{effective_timeout:g}s"
            ) from exc

    def json_request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        request = self._request(method, path, payload)
        with self._open(request, timeout_seconds=timeout_seconds) as response:
            body = response.read()
        try:
            parsed = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProfileError(
                f"{method} {path} returned malformed JSON: "
                f"{body[:200].decode('utf-8', errors='replace')}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ProfileError(f"{method} {path} returned a non-object JSON response")
        if parsed.get("error"):
            raise ProfileError(f"{method} {path} failed: {_event_error(parsed)}")
        return parsed

    def health(self) -> None:
        payload = self.json_request("GET", "/health")
        if payload.get("status") != "ok":
            raise ProfileError(f"server is not ready: {payload!r}")

    def wait_until_ready(
        self,
        timeout_seconds: float,
        *,
        poll_interval_seconds: float = READY_POLL_INTERVAL_SECONDS,
        progress: bool = True,
    ) -> None:
        """Poll transient loading responses until the model is ready."""
        started = time.monotonic()
        deadline = started + timeout_seconds
        waiting_announced = False
        last_status = "unknown"

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProfileError(
                    f"llama-server was not ready after {timeout_seconds:g}s "
                    f"(last status: {last_status})"
                )
            try:
                payload = self.json_request(
                    "GET",
                    "/health",
                    timeout_seconds=min(self.timeout_seconds, remaining),
                )
            except HttpStatusError as exc:
                if exc.status != 503:
                    raise
                last_status = exc.detail
            else:
                if payload.get("status") == "ok":
                    if waiting_announced and progress:
                        elapsed = time.monotonic() - started
                        print(
                            f"llama-server is ready after {elapsed:.1f}s.",
                            file=sys.stderr,
                        )
                    return
                last_status = str(payload.get("status") or payload)

            if not waiting_announced:
                waiting_announced = True
                if progress:
                    print(
                        "llama-server is still loading "
                        f"({last_status}); waiting up to {timeout_seconds:g}s...",
                        file=sys.stderr,
                    )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                continue
            time.sleep(min(poll_interval_seconds, remaining))

    def properties(self) -> dict[str, Any]:
        return self.json_request("GET", "/props")

    def apply_template(self, prompt: str) -> str:
        payload = self.json_request(
            "POST",
            "/apply-template",
            {"messages": [{"role": "user", "content": prompt}]},
        )
        formatted = payload.get("prompt")
        if not isinstance(formatted, str) or not formatted:
            raise ProfileError("/apply-template did not return a non-empty prompt")
        return formatted

    def tokenize(self, content: str) -> int:
        payload = self.json_request(
            "POST",
            "/tokenize",
            {
                "content": content,
                "add_special": False,
                "parse_special": True,
            },
        )
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            raise ProfileError("/tokenize did not return a token list")
        return len(tokens)

    def run_completion(self, task: PreparedTask, repeat: int) -> RunResult:
        payload = {
            "prompt": task.formatted_prompt,
            "n_predict": task.task.completion_tokens,
            "stream": True,
            "cache_prompt": False,
            "ignore_eos": True,
            "return_tokens": True,
            "return_progress": False,
            "timings_per_token": False,
            "temperature": 0.0,
            "seed": FIXED_SEED,
            "n_probs": 0,
        }
        request = self._request(
            "POST", "/completion", payload, accept="text/event-stream"
        )
        started_ns = time.perf_counter_ns()
        first_token_ns: int | None = None
        last_token_ns: int | None = None
        streamed_tokens = 0
        final_event: dict[str, Any] | None = None

        with self._open(request) as response:
            for event_name, event_data in iter_completion_sse(
                response, self.timeout_seconds
            ):
                if event_data == "[DONE]":
                    continue
                try:
                    event = json.loads(event_data)
                except json.JSONDecodeError as exc:
                    raise ProfileError(
                        f"/completion returned malformed SSE JSON: {event_data[:200]}"
                    ) from exc
                if not isinstance(event, dict):
                    raise ProfileError("/completion returned a non-object SSE event")
                if event_name == "error" or event.get("error"):
                    raise ProfileError(f"/completion stream failed: {_event_error(event)}")

                arrived_ns = time.perf_counter_ns()
                raw_tokens = event.get("tokens")
                token_count = len(raw_tokens) if isinstance(raw_tokens, list) else 0
                if token_count == 0 and raw_tokens is None:
                    content = event.get("content")
                    token_count = 1 if isinstance(content, str) and content else 0
                if token_count:
                    if first_token_ns is None:
                        first_token_ns = arrived_ns
                    last_token_ns = arrived_ns
                    streamed_tokens += token_count
                if event.get("stop") is True:
                    final_event = event
                    break

        finished_ns = time.perf_counter_ns()
        if final_event is None:
            raise ProfileError("/completion stream ended without a terminal stop event")
        if first_token_ns is None or last_token_ns is None:
            raise ProfileError("/completion stream ended without a generated token")

        timings = final_event.get("timings")
        if not isinstance(timings, dict):
            raise ProfileError(
                "/completion terminal event did not contain server timing data"
            )
        required_timing_fields = (
            "prompt_n",
            "prompt_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_second",
        )
        missing = [field for field in required_timing_fields if field not in timings]
        if missing:
            raise ProfileError(
                "/completion timing data is missing: " + ", ".join(missing)
            )

        processed_prompt_tokens = _integer(timings, "prompt_n")
        cached_prompt_tokens = _integer(timings, "cache_n", default=0)
        completion_tokens = _integer(timings, "predicted_n")
        prefill_ms = _number(timings, "prompt_ms")
        prefill_tps = _number(timings, "prompt_per_second")
        decode_ms = _number(timings, "predicted_ms")
        decode_tps = _number(timings, "predicted_per_second")
        draft_tokens, draft_accepted_tokens, draft_acceptance_percent = (
            _draft_acceptance(timings)
        )

        expected = task.task.completion_tokens
        if completion_tokens != expected:
            raise ProfileError(
                f"/completion generated {completion_tokens} tokens for {task.task.name}, "
                f"expected {expected}; stop_type={final_event.get('stop_type')!r}"
            )
        if streamed_tokens != completion_tokens:
            raise ProfileError(
                f"/completion streamed {streamed_tokens} token ids but reported "
                f"{completion_tokens} generated tokens"
            )
        if final_event.get("truncated") is True:
            raise ProfileError("/completion reported context truncation")

        decode_interval_ns = last_token_ns - first_token_ns
        client_decode_tps = None
        if streamed_tokens > 1 and decode_interval_ns > 0:
            client_decode_tps = (streamed_tokens - 1) / (
                decode_interval_ns / 1_000_000_000
            )

        return RunResult(
            task=task.task.name,
            repeat=repeat,
            expected_completion_tokens=expected,
            prompt_tokens=processed_prompt_tokens + cached_prompt_tokens,
            processed_prompt_tokens=processed_prompt_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            completion_tokens=completion_tokens,
            streamed_tokens=streamed_tokens,
            ttft_ms=(first_token_ns - started_ns) / 1_000_000,
            total_latency_ms=(finished_ns - started_ns) / 1_000_000,
            prefill_ms=prefill_ms,
            server_prefill_tps=prefill_tps,
            decode_ms=decode_ms,
            server_decode_tps=decode_tps,
            client_decode_tps=client_decode_tps,
            stop_type=_optional_string(final_event.get("stop_type")),
            draft_tokens=draft_tokens,
            draft_accepted_tokens=draft_accepted_tokens,
            draft_acceptance_percent=draft_acceptance_percent,
        )


def iter_sse_messages(lines: Iterable[bytes]) -> Iterator[tuple[str | None, str]]:
    """Parse an SSE byte stream into ``(event name, data)`` messages."""
    event_name: str | None = None
    data_lines: list[str] = []

    def dispatch() -> tuple[str | None, str] | None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return None
        message = (event_name, "\n".join(data_lines))
        event_name = None
        data_lines = []
        return message

    for raw_line in lines:
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProfileError("/completion SSE stream is not valid UTF-8") from exc
        line = line.rstrip("\r\n")
        if not line:
            message = dispatch()
            if message is not None:
                yield message
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    message = dispatch()
    if message is not None:
        yield message


def iter_completion_sse(
    lines: Iterable[bytes], timeout_seconds: float
) -> Iterator[tuple[str | None, str]]:
    """Read completion events while translating transport failures."""
    try:
        yield from iter_sse_messages(lines)
    except ProfileError:
        raise
    except TimeoutError as exc:
        raise ProfileError(
            f"/completion stream timed out after {timeout_seconds:g}s"
        ) from exc
    except OSError as exc:
        raise ProfileError(f"/completion stream failed while reading: {exc}") from exc


def _event_error(event: Mapping[str, Any]) -> str:
    error = event.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if error:
        return str(error)
    if event.get("message"):
        return str(event["message"])
    return json.dumps(event, ensure_ascii=False)[:500]


def _number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"server timing field {key!r} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ProfileError(f"server timing field {key!r} is invalid: {value!r}")
    return result


def _integer(
    mapping: Mapping[str, Any], key: str, *, default: int | None = None
) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProfileError(f"server timing field {key!r} is not a non-negative integer")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _draft_acceptance(
    timings: Mapping[str, Any],
) -> tuple[int | None, int | None, float | None]:
    has_draft = "draft_n" in timings
    has_accepted = "draft_n_accepted" in timings
    if not has_draft and not has_accepted:
        return None, None, None
    if not has_draft or not has_accepted:
        missing = "draft_n" if not has_draft else "draft_n_accepted"
        raise ProfileError(
            f"server timing data contains incomplete draft metrics; missing {missing}"
        )
    draft_tokens = _integer(timings, "draft_n")
    accepted_tokens = _integer(timings, "draft_n_accepted")
    if accepted_tokens > draft_tokens:
        raise ProfileError(
            "server timing field 'draft_n_accepted' exceeds 'draft_n'"
        )
    acceptance_percent = (
        accepted_tokens / draft_tokens * 100 if draft_tokens > 0 else None
    )
    return draft_tokens, accepted_tokens, acceptance_percent


def context_size(props: Mapping[str, Any]) -> int:
    defaults = props.get("default_generation_settings")
    candidates: list[Any] = []
    if isinstance(defaults, dict):
        candidates.append(defaults.get("n_ctx"))
        params = defaults.get("params")
        if isinstance(params, dict):
            candidates.append(params.get("n_ctx"))
    candidates.append(props.get("n_ctx"))
    for candidate in candidates:
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            return candidate
    raise ProfileError("/props did not report a positive context size")


def sanitized_server_metadata(props: Mapping[str, Any]) -> dict[str, Any]:
    model_path = props.get("model_path")
    model_filename = None
    if isinstance(model_path, str):
        model_filename = model_path.replace("\\", "/").rsplit("/", 1)[-1]
    metadata = {
        "model": props.get("model_alias") or props.get("model"),
        "model_filename": model_filename,
        "build_info": props.get("build_info"),
        "context_size": context_size(props),
        "total_slots": props.get("total_slots"),
        "speculative": None,
    }
    defaults = props.get("default_generation_settings")
    if isinstance(defaults, dict):
        metadata["speculative"] = defaults.get("speculative")
    return metadata


def prepare_tasks(
    client: LlamaServerClient,
    tasks: Sequence[BenchmarkTask],
    n_ctx: int,
) -> list[PreparedTask]:
    prepared: list[PreparedTask] = []
    for task in tasks:
        formatted = client.apply_template(task.prompt)
        prompt_tokens = client.tokenize(formatted)
        # Reserve one token for a possible BOS added by /completion.
        required_context = prompt_tokens + task.completion_tokens + 1
        if required_context > n_ctx:
            raise ProfileError(
                f"task {task.name!r} needs approximately {required_context} context "
                f"tokens ({prompt_tokens} prompt + {task.completion_tokens} completion "
                f"+ 1 BOS), but the server context is {n_ctx}; deselect the task or "
                "restart llama-server with a larger context"
            )
        prepared.append(
            PreparedTask(
                task=task,
                formatted_prompt=formatted,
                estimated_prompt_tokens=prompt_tokens,
            )
        )
    return prepared


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    if not 0 <= fraction <= 1:
        raise ValueError("percentile fraction must be between zero and one")
    ordered = sorted(values)
    if fraction == 0:
        return ordered[0]
    rank = math.ceil(fraction * len(ordered))
    return ordered[max(rank - 1, 0)]


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _draft_summary(results: Sequence[RunResult]) -> dict[str, Any] | None:
    applicable = [
        result
        for result in results
        if result.draft_tokens is not None
        and result.draft_accepted_tokens is not None
    ]
    if not applicable:
        return None
    total_draft = sum(result.draft_tokens or 0 for result in applicable)
    total_accepted = sum(result.draft_accepted_tokens or 0 for result in applicable)
    per_run_rates = [
        result.draft_acceptance_percent
        for result in applicable
        if result.draft_acceptance_percent is not None
    ]
    return {
        "runs": len(applicable),
        "draft_tokens": total_draft,
        "accepted_tokens": total_accepted,
        "acceptance_percent": (
            total_accepted / total_draft * 100 if total_draft > 0 else None
        ),
        "per_run_acceptance_percent": (
            _metric_summary(per_run_rates) if per_run_rates else None
        ),
    }


def summarize_results(results: Sequence[RunResult]) -> dict[str, Any]:
    by_task: dict[str, list[RunResult]] = {}
    for result in results:
        by_task.setdefault(result.task, []).append(result)

    tasks: dict[str, Any] = {}
    for task_name, task_results in by_task.items():
        client_rates = [
            result.client_decode_tps
            for result in task_results
            if result.client_decode_tps is not None
        ]
        tasks[task_name] = {
            "runs": len(task_results),
            "prompt_tokens": _metric_summary(
                [float(result.prompt_tokens) for result in task_results]
            ),
            "completion_tokens": _metric_summary(
                [float(result.completion_tokens) for result in task_results]
            ),
            "ttft_ms": _metric_summary(
                [result.ttft_ms for result in task_results]
            ),
            "total_latency_ms": _metric_summary(
                [result.total_latency_ms for result in task_results]
            ),
            "server_prefill_tps": _metric_summary(
                [result.server_prefill_tps for result in task_results]
            ),
            "server_decode_tps": _metric_summary(
                [result.server_decode_tps for result in task_results]
            ),
            "client_decode_tps": (
                _metric_summary(client_rates) if client_rates else None
            ),
            "draft_acceptance": _draft_summary(task_results),
        }

    total_processed = sum(result.processed_prompt_tokens for result in results)
    total_prefill_seconds = sum(result.prefill_ms for result in results) / 1000
    total_decoded = sum(result.completion_tokens for result in results)
    total_decode_seconds = sum(result.decode_ms for result in results) / 1000
    overall: dict[str, Any] = {
        "runs": len(results),
        "processed_prompt_tokens": total_processed,
        "completion_tokens": total_decoded,
        "ttft_ms": (
            _metric_summary([result.ttft_ms for result in results])
            if results
            else None
        ),
        "total_latency_ms": (
            _metric_summary([result.total_latency_ms for result in results])
            if results
            else None
        ),
        "token_weighted_server_prefill_tps": (
            total_processed / total_prefill_seconds
            if total_prefill_seconds > 0
            else None
        ),
        "token_weighted_server_decode_tps": (
            total_decoded / total_decode_seconds
            if total_decode_seconds > 0
            else None
        ),
        "draft_acceptance": _draft_summary(results),
    }
    return {"tasks": tasks, "overall": overall}


def run_benchmark(
    client: LlamaServerClient,
    tasks: Sequence[PreparedTask],
    repeats: int,
    warmups: int,
    *,
    progress: bool = True,
) -> tuple[list[RunResult], list[dict[str, Any]]]:
    if not tasks:
        raise ProfileError("no benchmark tasks were selected")

    for warmup_index in range(warmups):
        if progress:
            print(
                f"Warmup {warmup_index + 1}/{warmups}: {tasks[0].task.name}",
                file=sys.stderr,
            )
        try:
            client.run_completion(tasks[0], repeat=0)
        except ProfileError as exc:
            raise ProfileError(f"warmup failed: {exc}") from exc

    results: list[RunResult] = []
    failures: list[dict[str, Any]] = []
    total_runs = repeats * len(tasks)
    completed = 0
    for repeat_index in range(repeats):
        start = repeat_index % len(tasks)
        ordered = list(tasks[start:]) + list(tasks[:start])
        for task in ordered:
            completed += 1
            if progress:
                print(
                    f"Run {completed}/{total_runs}: {task.task.name} "
                    f"({task.task.completion_tokens} tokens)",
                    file=sys.stderr,
                )
            try:
                result = client.run_completion(task, repeat=repeat_index + 1)
            except ProfileError as exc:
                failure = {
                    "task": task.task.name,
                    "repeat": repeat_index + 1,
                    "error": str(exc),
                }
                failures.append(failure)
                if progress:
                    print(f"  failed: {exc}", file=sys.stderr)
            else:
                results.append(result)
    return results, failures


def build_report(
    *,
    base_url: str,
    repeats: int,
    warmups: int,
    timeout_seconds: float,
    ready_timeout_seconds: float,
    selected_tasks: Sequence[BenchmarkTask],
    server_metadata: Mapping[str, Any],
    results: Sequence[RunResult],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "server": dict(server_metadata),
        "config": {
            "base_url": base_url,
            "repeats": repeats,
            "warmups": warmups,
            "timeout_seconds": timeout_seconds,
            "ready_timeout_seconds": ready_timeout_seconds,
            "seed": FIXED_SEED,
            "cache_prompt": False,
            "ignore_eos": True,
            "tasks": [
                {
                    "name": task.name,
                    "description": task.description,
                    "completion_tokens": task.completion_tokens,
                }
                for task in selected_tasks
            ],
        },
        "runs": [asdict(result) for result in results],
        "failures": [dict(failure) for failure in failures],
        "summary": summarize_results(results),
    }


def _format_number(value: Any, decimals: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{decimals}f}"


def print_summary(report: Mapping[str, Any]) -> None:
    server = report["server"]
    print()
    print("llama-server profile")
    print(
        f"Model: {server.get('model') or server.get('model_filename') or 'unknown'} | "
        f"Context: {server.get('context_size')} | "
        f"Build: {server.get('build_info') or 'unknown'}"
    )

    task_summaries = report["summary"]["tasks"]
    if not task_summaries:
        print("\nNo successful benchmark runs.")
    else:
        print("\nLatency")
        print(
            f"{'Task':<10} {'Runs':>4} {'Prompt':>8} {'Output':>8} "
            f"{'TTFT p50':>10} {'TTFT p95':>10} {'E2E p50':>10} {'E2E p95':>10}"
        )
        for name, summary in task_summaries.items():
            print(
                f"{name:<10} {summary['runs']:>4} "
                f"{_format_number(summary['prompt_tokens']['median'], 0):>8} "
                f"{_format_number(summary['completion_tokens']['median'], 0):>8} "
                f"{_format_number(summary['ttft_ms']['median']):>10} "
                f"{_format_number(summary['ttft_ms']['p95']):>10} "
                f"{_format_number(summary['total_latency_ms']['median']):>10} "
                f"{_format_number(summary['total_latency_ms']['p95']):>10}"
            )
        print("All latency values are milliseconds.")

        print("\nThroughput")
        print(
            f"{'Task':<10} {'Prefill p50':>12} {'Prefill p95':>12} "
            f"{'Decode p50':>12} {'Decode p95':>12} "
            f"{'Client p50':>12} {'Client p95':>12}"
        )
        for name, summary in task_summaries.items():
            client = summary["client_decode_tps"]
            print(
                f"{name:<10} "
                f"{_format_number(summary['server_prefill_tps']['median']):>12} "
                f"{_format_number(summary['server_prefill_tps']['p95']):>12} "
                f"{_format_number(summary['server_decode_tps']['median']):>12} "
                f"{_format_number(summary['server_decode_tps']['p95']):>12} "
                f"{_format_number(client['median'] if client else None):>12} "
                f"{_format_number(client['p95'] if client else None):>12}"
            )
        print("All throughput values are tokens/second.")

        overall = report["summary"]["overall"]
        print(
            "\nOverall token-weighted server throughput: "
            f"prefill={_format_number(overall['token_weighted_server_prefill_tps'])} "
            f"tokens/s, "
            f"decode={_format_number(overall['token_weighted_server_decode_tps'])} "
            "tokens/s"
        )

        draft_summaries = [
            (name, summary["draft_acceptance"])
            for name, summary in task_summaries.items()
            if summary["draft_acceptance"] is not None
        ]
        if draft_summaries:
            print("\nSpeculative decoding")
            print(
                f"{'Task':<10} {'Runs':>4} {'Drafted':>9} {'Accepted':>9} "
                f"{'Accept %':>10} {'Run p50':>10} {'Run p95':>10}"
            )
            for name, draft in draft_summaries:
                assert draft is not None
                per_run = draft["per_run_acceptance_percent"]
                print(
                    f"{name:<10} {draft['runs']:>4} "
                    f"{draft['draft_tokens']:>9} "
                    f"{draft['accepted_tokens']:>9} "
                    f"{_format_number(draft['acceptance_percent']):>10} "
                    f"{_format_number(per_run['median'] if per_run else None):>10} "
                    f"{_format_number(per_run['p95'] if per_run else None):>10}"
                )
            overall_draft = overall["draft_acceptance"]
            if overall_draft is not None:
                print(
                    "Overall draft acceptance: "
                    f"{_format_number(overall_draft['acceptance_percent'])}% "
                    f"({overall_draft['accepted_tokens']}/"
                    f"{overall_draft['draft_tokens']} tokens)"
                )

    failures = report.get("failures", [])
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for failure in failures:
            print(
                f"- {failure['task']} repeat {failure['repeat']}: {failure['error']}"
            )


def write_json_report(path: Path, report: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def _parse_task_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("select at least one task")
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("task names must not be repeated")
    return names


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile prefill TPS, decode TPS, TTFT, and end-to-end latency for a "
            "running single-model llama-server."
        )
    )
    parser.add_argument(
        "--url",
        type=normalize_base_url,
        default=DEFAULT_BASE_URL,
        help=f"llama-server base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--repeats",
        type=_positive_int,
        default=DEFAULT_REPEATS,
        help=f"measured repetitions per task (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--warmups",
        type=_non_negative_int,
        default=DEFAULT_WARMUPS,
        help=f"excluded short-task warmup runs (default: {DEFAULT_WARMUPS})",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"per-request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    parser.add_argument(
        "--ready-timeout",
        type=_positive_float,
        default=DEFAULT_READY_TIMEOUT_SECONDS,
        help=(
            "maximum seconds to wait for a loading model "
            f"(default: {DEFAULT_READY_TIMEOUT_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--tasks",
        type=_parse_task_names,
        default=["short", "medium", "long"],
        metavar="NAMES",
        help="comma-separated tasks selected from short,medium,long (default: all)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="write full per-run data and summaries to PATH",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    available = {task.name: task for task in default_tasks()}
    unknown = [name for name in args.tasks if name not in available]
    if unknown:
        parser.error(
            "unknown task name(s): "
            + ", ".join(unknown)
            + "; choose from short, medium, long"
        )
    selected_tasks = [available[name] for name in args.tasks]
    client = LlamaServerClient(
        base_url=args.url,
        timeout_seconds=args.timeout,
        api_key=os.environ.get("LLAMA_API_KEY"),
    )

    try:
        client.wait_until_ready(args.ready_timeout)
        props = client.properties()
        metadata = sanitized_server_metadata(props)
        prepared = prepare_tasks(
            client, selected_tasks, int(metadata["context_size"])
        )
        results, failures = run_benchmark(
            client,
            prepared,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        report = build_report(
            base_url=args.url,
            repeats=args.repeats,
            warmups=args.warmups,
            timeout_seconds=args.timeout,
            ready_timeout_seconds=args.ready_timeout,
            selected_tasks=selected_tasks,
            server_metadata=metadata,
            results=results,
            failures=failures,
        )
        print_summary(report)
        if args.json:
            write_json_report(args.json, report)
            print(f"\nWrote JSON report to {args.json}")
    except ProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
