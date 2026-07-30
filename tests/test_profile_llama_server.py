from __future__ import annotations

import json
import threading
import time
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import profile_llama_server as profiler


def _send_json(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


@contextmanager
def fake_server(
    *,
    scenario: str = "success",
    expected_api_key: str | None = None,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []
    state = {"health_attempts": 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def handle(self) -> None:
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):
                # Expected when the client-side timeout test closes the socket.
                pass

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _record(self, body: dict[str, Any] | None = None) -> None:
            requests.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "body": body,
                }
            )

        def do_GET(self) -> None:
            self._record()
            if self.path == "/health":
                state["health_attempts"] += 1
                if scenario in {"warming_503", "always_503"} and (
                    scenario == "always_503" or state["health_attempts"] <= 2
                ):
                    _send_json(
                        self,
                        {"error": {"code": 503, "message": "Loading model"}},
                        status=503,
                    )
                    return
                if scenario == "warming_200" and state["health_attempts"] <= 2:
                    _send_json(self, {"status": "loading"})
                    return
                if scenario == "unhealthy":
                    _send_json(self, {"status": "loading"})
                else:
                    _send_json(self, {"status": "ok"})
                return
            if self.path == "/props":
                _send_json(
                    self,
                    {
                        "default_generation_settings": {
                            "n_ctx": 2048,
                            "speculative": False,
                        },
                        "total_slots": 1,
                        "model_path": "/private/models/test.gguf",
                        "model_alias": "test-model",
                        "build_info": "b999-test",
                    },
                )
                return
            _send_json(self, {"error": {"message": "not found"}}, status=404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._record(body)
            if expected_api_key and self.headers.get("Authorization") != (
                f"Bearer {expected_api_key}"
            ):
                _send_json(self, {"error": {"message": "unauthorized"}}, status=401)
                return
            if self.path == "/apply-template":
                content = body["messages"][0]["content"]
                _send_json(self, {"prompt": f"<user>{content}</user><assistant>"})
                return
            if self.path == "/tokenize":
                token_count = max(1, len(body["content"].split()))
                _send_json(self, {"tokens": list(range(token_count))})
                return
            if self.path != "/completion":
                _send_json(self, {"error": {"message": "not found"}}, status=404)
                return
            if scenario == "http_error":
                _send_json(self, {"error": {"message": "request rejected"}}, status=400)
                return
            if scenario == "timeout":
                time.sleep(0.2)

            count = int(body["n_predict"])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            if scenario == "midstream_timeout":
                self.wfile.write(b": stream opened\r\n\r\n")
                self.wfile.flush()
                time.sleep(0.2)
                return

            def event(payload: dict[str, Any], name: str | None = None) -> None:
                if name:
                    self.wfile.write(f"event: {name}\r\n".encode())
                encoded = json.dumps(payload, separators=(",", ":")).encode()
                self.wfile.write(b"data: " + encoded + b"\r\n\r\n")
                self.wfile.flush()

            if scenario == "stream_error":
                event({"error": {"message": "generation failed"}}, "error")
                return
            self.wfile.write(b": keepalive\r\n\r\n")
            self.wfile.flush()
            event({"prompt_progress": {"processed": 4}, "stop": False})

            emitted = count if scenario != "short_stream" else count - 1
            for token_id in range(emitted):
                event(
                    {
                        "content": f"t{token_id}",
                        "tokens": [token_id],
                        "stop": False,
                    }
                )
                time.sleep(0.001)

            final: dict[str, Any] = {
                "content": "",
                "tokens": [],
                "stop": True,
                "stop_type": "limit",
                "truncated": scenario == "truncated",
                "timings": {
                    "cache_n": 0,
                    "prompt_n": 20,
                    "prompt_ms": 10.0,
                    "prompt_per_second": 2000.0,
                    "predicted_n": count,
                    "predicted_ms": 30.0,
                    "predicted_per_second": count / 0.03,
                },
            }
            if scenario == "speculative":
                final["timings"].update(
                    {"draft_n": 10, "draft_n_accepted": 7}
                )
            if scenario == "missing_timings":
                final.pop("timings")
            event(final)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def prepared_task(tokens: int = 3) -> profiler.PreparedTask:
    task = profiler.BenchmarkTask(
        name="test",
        description="test task",
        prompt="Do the test.",
        completion_tokens=tokens,
    )
    return profiler.PreparedTask(
        task=task,
        formatted_prompt="<user>Do the test.</user><assistant>",
        estimated_prompt_tokens=6,
    )


class SseParserTests(unittest.TestCase):
    def test_comments_crlf_multiline_and_eof_dispatch(self) -> None:
        stream = BytesIO(
            b": ping\r\n\r\n"
            b"event: sample\r\n"
            b"data: {\"value\":\r\n"
            b"data: 1}\r\n\r\n"
            b"data: [DONE]\n"
        )
        self.assertEqual(
            list(profiler.iter_sse_messages(stream)),
            [("sample", '{"value":\n1}'), (None, "[DONE]")],
        )

    def test_invalid_utf8_is_rejected(self) -> None:
        with self.assertRaisesRegex(profiler.ProfileError, "valid UTF-8"):
            list(profiler.iter_sse_messages([b"data: \xff\n\n"]))


class StatisticsTests(unittest.TestCase):
    def test_nearest_rank_percentile(self) -> None:
        self.assertEqual(profiler.percentile([4, 1, 3, 2], 0.5), 2)
        self.assertEqual(profiler.percentile([4, 1, 3, 2], 0.95), 4)

    def test_token_weighted_throughput(self) -> None:
        first = profiler.RunResult(
            "a", 1, 2, 10, 10, 0, 2, 2, 5, 20, 10, 1000, 20, 100, 90, "limit"
        )
        second = profiler.RunResult(
            "a", 2, 4, 20, 20, 0, 4, 4, 6, 30, 40, 500, 40, 100, 95, "limit"
        )
        summary = profiler.summarize_results([first, second])["overall"]
        self.assertAlmostEqual(summary["token_weighted_server_prefill_tps"], 600)
        self.assertAlmostEqual(summary["token_weighted_server_decode_tps"], 100)


class ClientTests(unittest.TestCase):
    def test_unhealthy_server_is_rejected(self) -> None:
        with fake_server(scenario="unhealthy") as (url, _):
            client = profiler.LlamaServerClient(url, 2)
            with self.assertRaisesRegex(profiler.ProfileError, "not ready"):
                client.health()

    def test_waits_for_503_loading_response(self) -> None:
        with fake_server(scenario="warming_503") as (url, requests):
            client = profiler.LlamaServerClient(url, 2)
            client.wait_until_ready(
                1, poll_interval_seconds=0.001, progress=False
            )
        health_requests = [item for item in requests if item["path"] == "/health"]
        self.assertEqual(len(health_requests), 3)

    def test_waits_for_200_loading_status(self) -> None:
        with fake_server(scenario="warming_200") as (url, requests):
            client = profiler.LlamaServerClient(url, 2)
            client.wait_until_ready(
                1, poll_interval_seconds=0.001, progress=False
            )
        health_requests = [item for item in requests if item["path"] == "/health"]
        self.assertEqual(len(health_requests), 3)

    def test_readiness_timeout_reports_last_status(self) -> None:
        with fake_server(scenario="always_503") as (url, _):
            client = profiler.LlamaServerClient(url, 2)
            with self.assertRaisesRegex(
                profiler.ProfileError, r"not ready.*Loading model"
            ):
                client.wait_until_ready(
                    0.02, poll_interval_seconds=0.005, progress=False
                )

    def test_health_props_template_tokenize_and_completion(self) -> None:
        with fake_server(expected_api_key="secret") as (url, requests):
            client = profiler.LlamaServerClient(url, 2, api_key="secret")
            client.health()
            props = client.properties()
            self.assertEqual(profiler.context_size(props), 2048)
            prompt = client.apply_template("hello world")
            self.assertEqual(client.tokenize(prompt), 2)
            result = client.run_completion(prepared_task(), repeat=1)

        self.assertEqual(result.completion_tokens, 3)
        self.assertEqual(result.streamed_tokens, 3)
        self.assertGreater(result.ttft_ms, 0)
        self.assertGreater(result.client_decode_tps or 0, 0)
        self.assertEqual(result.server_prefill_tps, 2000)
        self.assertIsNone(result.draft_acceptance_percent)
        protected = [item for item in requests if item["method"] == "POST"]
        self.assertTrue(protected)
        self.assertTrue(
            all(item["authorization"] == "Bearer secret" for item in protected)
        )

    def test_draft_acceptance_is_reported_when_present(self) -> None:
        with fake_server(scenario="speculative") as (url, _):
            client = profiler.LlamaServerClient(url, 2)
            result = client.run_completion(prepared_task(), repeat=1)
        self.assertEqual(result.draft_tokens, 10)
        self.assertEqual(result.draft_accepted_tokens, 7)
        self.assertEqual(result.draft_acceptance_percent, 70)
        draft = profiler.summarize_results([result])["overall"]["draft_acceptance"]
        self.assertEqual(draft["draft_tokens"], 10)
        self.assertEqual(draft["accepted_tokens"], 7)
        self.assertEqual(draft["acceptance_percent"], 70)

    def test_http_error_body_is_reported(self) -> None:
        with fake_server(scenario="http_error") as (url, _):
            client = profiler.LlamaServerClient(url, 2)
            with self.assertRaisesRegex(profiler.ProfileError, "request rejected"):
                client.run_completion(prepared_task(), repeat=1)

    def test_stream_error_is_reported(self) -> None:
        with fake_server(scenario="stream_error") as (url, _):
            client = profiler.LlamaServerClient(url, 2)
            with self.assertRaisesRegex(profiler.ProfileError, "generation failed"):
                client.run_completion(prepared_task(), repeat=1)

    def test_missing_timing_data_is_rejected(self) -> None:
        with fake_server(scenario="missing_timings") as (url, _):
            client = profiler.LlamaServerClient(url, 2)
            with self.assertRaisesRegex(profiler.ProfileError, "timing data"):
                client.run_completion(prepared_task(), repeat=1)

    def test_incomplete_stream_is_rejected(self) -> None:
        with fake_server(scenario="short_stream") as (url, _):
            client = profiler.LlamaServerClient(url, 2)
            with self.assertRaisesRegex(profiler.ProfileError, "streamed 2 token"):
                client.run_completion(prepared_task(), repeat=1)

    def test_context_truncation_is_rejected(self) -> None:
        with fake_server(scenario="truncated") as (url, _):
            client = profiler.LlamaServerClient(url, 2)
            with self.assertRaisesRegex(profiler.ProfileError, "context truncation"):
                client.run_completion(prepared_task(), repeat=1)

    def test_timeout_is_reported(self) -> None:
        with fake_server(scenario="timeout") as (url, _):
            client = profiler.LlamaServerClient(url, 0.02)
            with self.assertRaisesRegex(profiler.ProfileError, "timed out"):
                client.run_completion(prepared_task(), repeat=1)

    def test_midstream_timeout_is_reported(self) -> None:
        with fake_server(scenario="midstream_timeout") as (url, _):
            client = profiler.LlamaServerClient(url, 0.02)
            with self.assertRaisesRegex(profiler.ProfileError, "stream timed out"):
                client.run_completion(prepared_task(), repeat=1)


class WorkflowTests(unittest.TestCase):
    def test_prepare_tasks_rejects_context_overflow(self) -> None:
        with fake_server() as (url, _):
            client = profiler.LlamaServerClient(url, 2)
            task = profiler.BenchmarkTask("large", "large", "one two three", 10)
            with self.assertRaisesRegex(profiler.ProfileError, "server context is 5"):
                profiler.prepare_tasks(client, [task], n_ctx=5)

    def test_report_json_excludes_prompts_and_generated_text(self) -> None:
        result = profiler.RunResult(
            "short", 1, 2, 10, 10, 0, 2, 2, 5, 20, 10, 1000, 20, 100, 90, "limit"
        )
        task = profiler.BenchmarkTask("short", "description", "secret prompt", 2)
        report = profiler.build_report(
            base_url="http://localhost:8080",
            repeats=1,
            warmups=0,
            timeout_seconds=1,
            ready_timeout_seconds=2,
            selected_tasks=[task],
            server_metadata={"context_size": 2048},
            results=[result],
            failures=[],
        )
        encoded = json.dumps(report)
        self.assertNotIn("secret prompt", encoded)
        self.assertNotIn("generated_text", encoded)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            profiler.write_json_report(path, report)
            self.assertEqual(json.loads(path.read_text())["schema_version"], 2)

    def test_summary_from_json_reprints_schema_v2_report(self) -> None:
        result = profiler.RunResult(
            "short", 1, 2, 10, 10, 0, 2, 2, 5, 20, 10, 1000, 20, 100, 90, "limit"
        )
        report = profiler.build_report(
            base_url="http://localhost:8080",
            repeats=1,
            warmups=0,
            timeout_seconds=1,
            ready_timeout_seconds=2,
            selected_tasks=[
                profiler.BenchmarkTask("short", "description", "prompt", 2)
            ],
            server_metadata={
                "model": "test-model",
                "context_size": 2048,
                "build_info": "test-build",
            },
            results=[result],
            failures=[],
        )
        expected = StringIO()
        with redirect_stdout(expected):
            profiler.print_summary(report)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            profiler.write_json_report(path, report)
            actual = StringIO()
            with redirect_stdout(actual):
                status = profiler.main(["--summary-from-json", str(path)])

        self.assertEqual(status, 0)
        self.assertEqual(actual.getvalue(), expected.getvalue())

    def test_summary_from_json_rejects_other_schema_versions(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text('{"schema_version": 1}', encoding="utf-8")
            errors = StringIO()
            with redirect_stderr(errors):
                status = profiler.main(["--summary-from-json", str(path)])

        self.assertEqual(status, 2)
        self.assertIn("uses schema version 1; expected 2", errors.getvalue())

    def test_benchmark_rotates_task_order_and_collects_failures(self) -> None:
        calls: list[tuple[str, int]] = []

        class StubClient:
            def run_completion(
                self, task: profiler.PreparedTask, repeat: int
            ) -> profiler.RunResult:
                calls.append((task.task.name, repeat))
                if task.task.name == "b" and repeat == 2:
                    raise profiler.ProfileError("planned failure")
                count = task.task.completion_tokens
                return profiler.RunResult(
                    task.task.name,
                    repeat,
                    count,
                    10,
                    10,
                    0,
                    count,
                    count,
                    1,
                    2,
                    1,
                    10000,
                    1,
                    count * 1000,
                    count * 900,
                    "limit",
                )

        tasks = [
            profiler.PreparedTask(
                profiler.BenchmarkTask(name, name, name, 2), name, 1
            )
            for name in ("a", "b", "c")
        ]
        results, failures = profiler.run_benchmark(
            StubClient(), tasks, repeats=2, warmups=0, progress=False
        )
        self.assertEqual(
            calls,
            [("a", 1), ("b", 1), ("c", 1), ("b", 2), ("c", 2), ("a", 2)],
        )
        self.assertEqual(len(results), 5)
        self.assertEqual(failures[0]["error"], "planned failure")


if __name__ == "__main__":
    unittest.main()
