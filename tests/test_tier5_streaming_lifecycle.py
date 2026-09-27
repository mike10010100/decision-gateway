"""
Tier 5 White-Box Adversarial Coverage Hardening:
Streaming Lifecycle, Client Disconnects, Lifespan, and Concurrency.

Covers:
1. Multi-byte UTF-8 split across SSE and NDJSON chunk boundaries.
2. Client disconnect timing: before first chunk, between chunks, during keepalive quiet window.
3. Upstream streaming anomalies: incomplete NDJSON lines, huge 4MB chunk sizes, mid-stream errors.
4. Model deletion URL encoding & special characters (colons, plus, dots, spaces, unicode, slashes).
5. Concurrency stress: 50 concurrent mixed requests across all gateway endpoints
   and partial failure resilience.
6. Lifespan task cancellation, background task tracking, and clean shutdown.
"""

import sys
import os
import asyncio
import json
from typing import Any, Callable, Sequence, Union
import httpx
import pytest
import respx
from starlette.requests import Request, ClientDisconnect
from starlette.datastructures import Headers

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app, lifespan, background_tasks, preload_models_background
from app.config import OLLAYA_URL, OLLAYA_MCP_URL
from app.streaming import (
    SafeStreamingResponse,
    sse_stream_bridge,
    ndjson_stream_bridge,
    format_sse_event,
    sanitize_upstream_headers,
    HOP_BY_HOP_HEADERS,
)

# ---------------------------------------------------------------------------
# Test Helpers & Mock Classes
# ---------------------------------------------------------------------------


class MockAsyncByteStream(httpx.AsyncByteStream):
    """Hermetic async byte stream with lifecycle tracking and simulated delays."""

    def __init__(
        self,
        chunks_or_gen: Union[Sequence[bytes], Callable[[], Any]],
        delay: float = 0.0,
    ):
        self.chunks_or_gen = chunks_or_gen
        self.delay = delay
        self.closed = False
        self.yielded_count = 0

    async def __aiter__(self):
        if callable(self.chunks_or_gen):
            async for chunk in self.chunks_or_gen():
                self.yielded_count += 1
                yield chunk
        else:
            for chunk in self.chunks_or_gen:
                if self.delay > 0:
                    await asyncio.sleep(self.delay)
                self.yielded_count += 1
                yield chunk

    async def aclose(self) -> None:
        self.closed = True


class MockDisconnectRequest:
    """Mock Starlette Request that signals http.disconnect after a configurable delay."""

    def __init__(self, disconnect_delay: float = 0.0):
        self.disconnect_delay = disconnect_delay
        self.headers = Headers({"accept": "text/event-stream"})
        self.query_params: dict[str, str] = {}
        self.method = "GET"

    async def receive(self) -> dict[str, str]:
        if self.disconnect_delay > 0:
            await asyncio.sleep(self.disconnect_delay)
        return {"type": "http.disconnect"}

    async def body(self) -> bytes:
        return b""


# ===========================================================================
# 1. Multi-Byte UTF-8 Split Across SSE and NDJSON Chunk Boundaries
# ===========================================================================


class TestUTF8ChunkBoundaryStreaming:
    """Verifies that multi-byte UTF-8 sequences split arbitrarily across chunk boundaries

    are streamed without decoding corruption or gateway crashes.
    """

    @pytest.mark.asyncio
    async def test_sse_utf8_split_emoji_across_chunks(self):
        """Adversarial: 4-byte UTF-8 emoji (🚀 = f0 9f 9a 80) split across two chunks."""
        # Chunk 1 ends with the first two bytes of 🚀
        chunk1 = b'event: message\ndata: {"icon": "test \xf0\x9f'
        # Chunk 2 starts with the remaining two bytes of 🚀
        chunk2 = b'\x9a\x80"}\n\n'

        stream = MockAsyncByteStream([chunk1, chunk2])
        resp = httpx.Response(200, stream=stream)
        req = MockDisconnectRequest(disconnect_delay=10.0)

        bridge = sse_stream_bridge(req, resp)  # type: ignore[arg-type]
        collected = []
        async for chunk in bridge:
            collected.append(chunk)

        combined = b"".join(collected).decode("utf-8")
        assert '{"icon": "test 🚀"}' in combined
        assert stream.closed is True

    @pytest.mark.asyncio
    async def test_sse_utf8_split_cjk_across_chunks(self):
        """Adversarial: 3-byte CJK character (网 = e7 bd 91) split 1 byte / 2 bytes."""
        # "智能网关" in UTF-8
        chunk1 = b'event: message\ndata: {"name": "\xe6\x99\xba\xe8\x83\xbd\xe7'
        chunk2 = b'\xbd\x91\xe5\x85\xb3"}\n\n'

        stream = MockAsyncByteStream([chunk1, chunk2])
        resp = httpx.Response(200, stream=stream)
        req = MockDisconnectRequest(disconnect_delay=10.0)

        bridge = sse_stream_bridge(req, resp)  # type: ignore[arg-type]
        collected = []
        async for chunk in bridge:
            collected.append(chunk)

        combined = b"".join(collected).decode("utf-8")
        assert '{"name": "智能网关"}' in combined

    @pytest.mark.asyncio
    async def test_mcp_get_proxy_split_utf8_stream(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """E2E White-Box: GET /mcp correctly reassembles split UTF-8 emojis
        through SafeStreamingResponse.
        """
        raw_sse = (
            b"retry: 3000\n\n"
            b'event: message\ndata: {"status": "running \xf0\x9f\x8c\x8d \xf0\x9f\x9a\x80"}\n\n'
        )
        # Split into 3 arbitrary chunks slicing right through emoji byte sequences
        chunk_a = raw_sse[:45]
        chunk_b = raw_sse[45:50]
        chunk_c = raw_sse[50:]

        async def split_stream():
            yield chunk_a
            yield chunk_b
            yield chunk_c

        respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").respond(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=split_stream(),
        )

        async with async_client.stream("GET", "/mcp") as resp:
            assert resp.status_code == 200
            content = await resp.aread()
            text = content.decode("utf-8")
            assert "running 🌍 🚀" in text

    @pytest.mark.asyncio
    async def test_ndjson_pull_split_utf8_chunks(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """Adversarial: POST /v1/models/pull with stream=True handles split UTF-8 in NDJSON."""
        line1 = b'{"status": "\xe6\xad\xa3\xe5\x9c\xa8\xe4\xb8\x8b\xe8\xbd\xbd \xf0\x9f\x9a\x80"}\n'
        line2 = b'{"status": "completed", "percent": 100}\n'

        # Slice line1 midway through character
        split_pos = len(line1) - 4
        chunk1 = line1[:split_pos]
        chunk2 = line1[split_pos:] + line2

        async def ndjson_gen():
            yield chunk1
            yield chunk2

        respx_mock.post(f"{OLLAYA_URL}/api/pull").respond(
            status_code=200,
            headers={"content-type": "application/x-ndjson"},
            stream=ndjson_gen(),
        )

        async with async_client.stream(
            "POST", "/v1/models/pull", json={"model": "decider", "stream": True}
        ) as resp:
            assert resp.status_code == 200
            lines = [line async for line in resp.aiter_lines() if line]
            assert len(lines) == 2
            data0 = json.loads(lines[0])
            assert "正在下载 🚀" in data0["status"]
            data1 = json.loads(lines[1])
            assert data1["percent"] == 100

    def test_format_sse_event_unicode_and_multiline(self):
        """Unit: format_sse_event handles Unicode characters, dicts, and multi-line strings."""
        event_bytes = format_sse_event(
            event="custom_alert",
            data={"msg": "Outage in 🇨🇭 Zurich region\nImmediate triage required"},
            event_id="evt-42",
            retry=5000,
        )
        decoded = event_bytes.decode("utf-8")
        assert "event: custom_alert\n" in decoded
        assert "id: evt-42\n" in decoded
        assert "retry: 5000\n" in decoded
        assert "data: " in decoded
        assert "Zurich region" in decoded


# ===========================================================================
# 2. Client Disconnect Timing & Cancellation
# ===========================================================================


class TestClientDisconnectLifecycle:
    """Verifies that client disconnects arriving at different lifecycle phases

    cleanly cancel upstream reads and trigger resource cleanup without leaks.
    """

    @pytest.mark.asyncio
    async def test_sse_disconnect_before_first_chunk(self):
        """Client disconnects before upstream produces the first chunk."""
        # Upstream takes 0.5s to yield; client disconnects at 0.02s
        stream = MockAsyncByteStream([b"data: chunk\n\n"], delay=0.5)
        resp = httpx.Response(200, stream=stream)
        req = MockDisconnectRequest(disconnect_delay=0.02)

        bridge = sse_stream_bridge(req, resp)  # type: ignore[arg-type]
        received = []
        async for chunk in bridge:
            received.append(chunk)

        assert len(received) == 0
        assert stream.closed is True

    @pytest.mark.asyncio
    async def test_sse_disconnect_between_chunks(self):
        """Client disconnects between chunk 1 and chunk 2."""

        # Yield first chunk fast, but second chunk delayed; disconnect in between
        async def multi_chunk():
            yield b"data: chunk1\n\n"
            await asyncio.sleep(0.5)
            yield b"data: chunk2\n\n"

        stream = MockAsyncByteStream(multi_chunk)
        resp = httpx.Response(200, stream=stream)
        req = MockDisconnectRequest(disconnect_delay=0.05)

        bridge = sse_stream_bridge(req, resp)  # type: ignore[arg-type]
        received = []
        async for chunk in bridge:
            received.append(chunk)

        assert len(received) == 1
        assert b"chunk1" in received[0]
        assert stream.closed is True

    @pytest.mark.asyncio
    async def test_sse_disconnect_during_keepalive_quiet_window(self):
        """Client disconnects during 15s quiet ping window; must abort in <100ms."""

        # Upstream yields keepalive ping, then delays 5.0 seconds
        async def quiet_stream():
            yield b":\n\n"
            await asyncio.sleep(5.0)
            yield b'data: {"after_quiet": true}\n\n'

        stream = MockAsyncByteStream(quiet_stream)
        resp = httpx.Response(200, stream=stream)
        # Disconnect triggers at 0.03s during the quiet sleep
        req = MockDisconnectRequest(disconnect_delay=0.03)

        t0 = asyncio.get_running_loop().time()
        bridge = sse_stream_bridge(req, resp)  # type: ignore[arg-type]
        received = []
        async for chunk in bridge:
            received.append(chunk)
        elapsed = asyncio.get_running_loop().time() - t0

        assert len(received) == 1
        assert received[0] == b":\n\n"
        # Disconnect was caught promptly without waiting 5 seconds
        assert elapsed < 0.5
        assert stream.closed is True

    @pytest.mark.asyncio
    async def test_ndjson_disconnect_before_first_chunk(self):
        """ndjson_stream_bridge aborts cleanly if client disconnects before first chunk."""
        stream = MockAsyncByteStream([b'{"status": "starting"}\n'], delay=0.5)
        resp = httpx.Response(200, stream=stream)
        req = MockDisconnectRequest(disconnect_delay=0.02)

        bridge = ndjson_stream_bridge(req, resp)  # type: ignore[arg-type]
        received = []
        async for chunk in bridge:
            received.append(chunk)

        assert len(received) == 0
        assert stream.closed is True

    @pytest.mark.asyncio
    async def test_ndjson_disconnect_between_chunks(self):
        """ndjson_stream_bridge aborts upstream when client disconnects mid-stream."""

        async def pull_chunks():
            yield b'{"status": "manifest"}\n'
            await asyncio.sleep(0.5)
            yield b'{"status": "layer1"}\n'

        stream = MockAsyncByteStream(pull_chunks)
        resp = httpx.Response(200, stream=stream)
        req = MockDisconnectRequest(disconnect_delay=0.05)

        bridge = ndjson_stream_bridge(req, resp)  # type: ignore[arg-type]
        received = []
        async for chunk in bridge:
            received.append(chunk)

        assert len(received) == 1
        assert b"manifest" in received[0]
        assert stream.closed is True

    @pytest.mark.asyncio
    async def test_safe_streaming_response_client_disconnect_handled(self):
        """SafeStreamingResponse catches ClientDisconnect during send() without crashing ASGI."""

        async def sample_gen():
            yield b"chunk1"
            yield b"chunk2"

        response = SafeStreamingResponse(sample_gen())

        call_count = 0

        async def mock_send(message: dict) -> None:
            nonlocal call_count
            call_count += 1
            if message.get("type") == "http.response.body" and message.get("body") == b"chunk1":
                raise ClientDisconnect()

        # Should complete cleanly without raising ClientDisconnect to caller
        await response.stream_response(mock_send)
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_safe_streaming_response_broken_pipe_oserror_handled(self):
        """SafeStreamingResponse catches BrokenPipe / OSError during transmission."""

        async def sample_gen():
            yield b"data_to_pipe"

        response = SafeStreamingResponse(sample_gen())

        async def mock_send(message: dict) -> None:
            if message.get("type") == "http.response.body":
                raise OSError(32, "Broken pipe")

        await response.stream_response(mock_send)

    @pytest.mark.asyncio
    async def test_safe_streaming_response_aclose_error_suppressed(self):
        """SafeStreamingResponse suppresses exceptions during body_iterator.aclose()."""

        class FailingGenerator:
            async def __aiter__(self):
                yield b"ok"

            async def aclose(self):
                raise RuntimeError("Failed to clean up generator socket")

        response = SafeStreamingResponse(FailingGenerator())
        sent = []

        async def mock_send(message: dict) -> None:
            sent.append(message)

        # Must not raise RuntimeError
        await response.stream_response(mock_send)
        assert any(m.get("type") == "http.response.start" for m in sent)


# ===========================================================================
# 3. Upstream Streaming Anomalies
# ===========================================================================


class TestUpstreamStreamingAnomalies:
    """Verifies robustness against incomplete NDJSON lines, huge chunk sizes,

    and mid-stream upstream resets/timeouts.
    """

    @pytest.mark.asyncio
    async def test_ndjson_fragmented_partial_lines_forwarded(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """Adversarial: Upstream sends fragmented NDJSON lines across small chunk fragments."""
        fragments = [
            b'{"status": ',
            b'"downloading", ',
            b'"completed": 500, ',
            b'"total": 1000}\n',
            b'{"status": "fin',
            b'ished"}\n',
        ]

        async def frag_stream():
            for f in fragments:
                yield f

        respx_mock.post(f"{OLLAYA_URL}/api/pull").respond(
            status_code=200,
            headers={"content-type": "application/x-ndjson"},
            stream=frag_stream(),
        )

        async with async_client.stream(
            "POST", "/v1/models/pull", json={"model": "decider", "stream": True}
        ) as resp:
            assert resp.status_code == 200
            lines = [line async for line in resp.aiter_lines() if line]
            assert len(lines) == 2
            assert json.loads(lines[0])["completed"] == 500
            assert json.loads(lines[1])["status"] == "finished"

    @pytest.mark.asyncio
    async def test_ndjson_huge_chunk_sizes_4mb(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """Adversarial: Upstream transmits 4MB chunk in NDJSON stream without buffer overflow."""
        huge_payload = {"status": "layer_progress", "data": "A" * (4 * 1024 * 1024)}
        huge_bytes = json.dumps(huge_payload).encode("utf-8") + b"\n"

        async def huge_gen():
            yield huge_bytes

        respx_mock.post(f"{OLLAYA_URL}/api/pull").respond(
            status_code=200,
            headers={"content-type": "application/x-ndjson"},
            stream=huge_gen(),
        )

        async with async_client.stream(
            "POST", "/v1/models/pull", json={"model": "decider", "stream": True}
        ) as resp:
            assert resp.status_code == 200
            total_read = 0
            async for chunk in resp.aiter_bytes():
                total_read += len(chunk)
            assert total_read >= 4 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_sse_huge_chunk_sizes_4mb(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """Adversarial: Upstream transmits 4MB SSE event frame without memory crash."""
        large_event = b'event: message\ndata: {"bulk": "' + (b"B" * (4 * 1024 * 1024)) + b'"}\n\n'

        async def large_sse_gen():
            yield large_event

        respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").respond(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=large_sse_gen(),
        )

        async with async_client.stream("GET", "/mcp") as resp:
            assert resp.status_code == 200
            total_bytes = 0
            async for chunk in resp.aiter_bytes():
                total_bytes += len(chunk)
            assert total_bytes >= 4 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_sse_empty_zero_byte_chunks_preserved(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """Adversarial: Empty zero-byte chunks interspersed do not disrupt streaming."""

        async def empty_interspersed():
            yield b""
            yield b'event: message\ndata: {"num": 1}\n\n'
            yield b""
            yield b'event: message\ndata: {"num": 2}\n\n'
            yield b""

        respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").respond(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=empty_interspersed(),
        )

        async with async_client.stream("GET", "/mcp") as resp:
            assert resp.status_code == 200
            lines = [line async for line in resp.aiter_lines() if line]
            assert any('{"num": 1}' in line for line in lines)
            assert any('{"num": 2}' in line for line in lines)

    @pytest.mark.asyncio
    async def test_sse_upstream_read_error_emits_error_frame(self):
        """sse_stream_bridge maps ReadError mid-stream to an SSE error event."""

        async def failing_stream():
            yield b'event: message\ndata: {"ok": 1}\n\n'
            raise httpx.ReadError("Connection abruptly terminated by upstream")

        stream = MockAsyncByteStream(failing_stream)
        resp = httpx.Response(200, stream=stream)
        req = MockDisconnectRequest(disconnect_delay=10.0)

        bridge = sse_stream_bridge(req, resp)  # type: ignore[arg-type]
        chunks = []
        async for c in bridge:
            chunks.append(c)

        combined = b"".join(chunks).decode("utf-8")
        assert "event: error" in combined
        assert "bad_gateway" in combined
        assert stream.closed is True

    @pytest.mark.asyncio
    async def test_ndjson_upstream_timeout_terminates_cleanly(self):
        """ndjson_stream_bridge terminates cleanly without crash on upstream timeout."""

        async def timing_out_stream():
            yield b'{"status": "first"}\n'
            raise httpx.ReadTimeout("Timeout waiting for model layer")

        stream = MockAsyncByteStream(timing_out_stream)
        resp = httpx.Response(200, stream=stream)
        req = MockDisconnectRequest(disconnect_delay=10.0)

        bridge = ndjson_stream_bridge(req, resp)  # type: ignore[arg-type]
        chunks = []
        async for c in bridge:
            chunks.append(c)

        assert len(chunks) == 1
        assert b"first" in chunks[0]
        assert stream.closed is True


# ===========================================================================
# 4. Model Deletion URL Handling & Special Characters
# ===========================================================================


class TestModelDeletionURLHandling:
    """Verifies DELETE /v1/models/{model_name} against colons, dots, hyphens,

    percent-encoding, unicode, and upstream fault mappings.
    """

    @pytest.mark.asyncio
    async def test_delete_model_standard_with_tag(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """DELETE /v1/models/decider:latest correctly passes model name with tag to upstream."""
        route = respx_mock.delete(f"{OLLAYA_URL}/api/delete").respond(
            status_code=200, json={"status": "success", "message": "Model removed"}
        )
        resp = await async_client.delete("/v1/models/decider:latest")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        sent_body = json.loads(route.calls.last.request.read())
        assert sent_body == {"model": "decider:latest"}

    @pytest.mark.asyncio
    async def test_delete_model_encoded_colon(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """DELETE /v1/models/kev%3A4b decodes %3A to colon in upstream JSON payload."""
        route = respx_mock.delete(f"{OLLAYA_URL}/api/delete").respond(
            status_code=200, json={"status": "success", "message": "Model removed"}
        )
        resp = await async_client.delete("/v1/models/kev%3A4b")
        assert resp.status_code == 200
        sent_body = json.loads(route.calls.last.request.read())
        assert sent_body == {"model": "kev:4b"}

    @pytest.mark.asyncio
    async def test_delete_model_plus_and_at_signs(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """DELETE handles plus and at signs in model identifiers."""
        route = respx_mock.delete(f"{OLLAYA_URL}/api/delete").respond(
            status_code=200, json={"status": "success"}
        )
        resp = await async_client.delete("/v1/models/model+tag@sha256")
        assert resp.status_code == 200
        sent_body = json.loads(route.calls.last.request.read())
        assert sent_body == {"model": "model+tag@sha256"}

    @pytest.mark.asyncio
    async def test_delete_model_dots_and_hyphens(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """DELETE handles dots and hyphens (e.g. filename style)."""
        route = respx_mock.delete(f"{OLLAYA_URL}/api/delete").respond(
            status_code=200, json={"status": "success"}
        )
        resp = await async_client.delete("/v1/models/my-custom-model.v1.0.onnx")
        assert resp.status_code == 200
        sent_body = json.loads(route.calls.last.request.read())
        assert sent_body == {"model": "my-custom-model.v1.0.onnx"}

    @pytest.mark.asyncio
    async def test_delete_model_encoded_spaces(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """DELETE /v1/models/my%20model decodes space properly."""
        route = respx_mock.delete(f"{OLLAYA_URL}/api/delete").respond(
            status_code=200, json={"status": "success"}
        )
        resp = await async_client.delete("/v1/models/my%20model")
        assert resp.status_code == 200
        sent_body = json.loads(route.calls.last.request.read())
        assert sent_body == {"model": "my model"}

    @pytest.mark.asyncio
    async def test_delete_model_encoded_unicode(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """DELETE /v1/models/%E6%A8%A1%E5%9E%8B decodes UTF-8 string '模型'."""
        route = respx_mock.delete(f"{OLLAYA_URL}/api/delete").respond(
            status_code=200, json={"status": "success"}
        )
        resp = await async_client.delete("/v1/models/%E6%A8%A1%E5%9E%8B")
        assert resp.status_code == 200
        sent_body = json.loads(route.calls.last.request.read())
        assert sent_body == {"model": "模型"}

    @pytest.mark.asyncio
    async def test_delete_model_slash_behavior(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """Adversarial Observation: In Starlette, path segment without :path converter

        routes namespace/model to 404 because slash is treated as path separator.
        """
        resp = await async_client.delete("/v1/models/namespace%2Fmodel")
        # Starlette matches 4 path segments against 3-segment template, yielding 404
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_model_upstream_timeout_returns_504(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """DELETE maps upstream timeout to HTTP 504 Gateway Timeout."""
        respx_mock.delete(f"{OLLAYA_URL}/api/delete").mock(
            side_effect=httpx.ReadTimeout("Timeout waiting for delete")
        )
        resp = await async_client.delete("/v1/models/decider")
        assert resp.status_code == 504
        assert "timed out" in resp.json().get("detail", "")

    @pytest.mark.asyncio
    async def test_delete_model_upstream_crash_returns_502(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """DELETE maps upstream 500 to HTTP 502 Bad Gateway."""
        respx_mock.delete(f"{OLLAYA_URL}/api/delete").respond(
            status_code=500, text="Internal disk error on delete"
        )
        resp = await async_client.delete("/v1/models/decider")
        assert resp.status_code == 502
        assert "Upstream model delete server error" in resp.json().get("detail", "")

    @pytest.mark.asyncio
    async def test_delete_model_upstream_404_passthrough(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """DELETE preserves 404 when model is not found upstream."""
        respx_mock.delete(f"{OLLAYA_URL}/api/delete").respond(
            status_code=404, json={"error": "model 'ghost' not found"}
        )
        resp = await async_client.delete("/v1/models/ghost")
        assert resp.status_code == 404
        assert "ghost" in resp.text


# ===========================================================================
# 5. Concurrency Stress (50 Concurrent Mixed Requests)
# ===========================================================================


class TestConcurrencyStress50Requests:
    """Stress tests the decision-gateway with 50 simultaneous mixed requests

    across decisions, model pulls, deletions, and streaming SSE endpoints.
    """

    @pytest.mark.asyncio
    async def test_50_concurrent_mixed_requests_success(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
    ):
        """Stress: 50 concurrent mixed requests across all endpoints all return 200 OK

        without race conditions, socket leaks, or cross-request pollution.
        """

        # Dynamic callback ensures each concurrent request receives a unique response
        def inference_callback(request):
            body = json.loads(request.content.decode("utf-8"))
            m = body.get("model", "decider")
            return httpx.Response(
                200,
                json={"answers": {"q1": "val"}, "model": m, "usage": {"tokens": 20}},
            )

        respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(side_effect=inference_callback)
        respx_mock.post(f"{OLLAYA_URL}/api/pull").mock(
            side_effect=lambda req: httpx.Response(200, json={"status": "success"})
        )
        respx_mock.delete(f"{OLLAYA_URL}/api/delete").mock(
            side_effect=lambda req: httpx.Response(
                200, json={"status": "success", "message": "deleted"}
            )
        )
        respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").mock(
            side_effect=lambda req: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b'retry: 3000\n\nevent: message\ndata: {"status": "ok"}\n\n',
            )
        )
        respx_mock.post(f"{OLLAYA_MCP_URL}/mcp").mock(
            side_effect=lambda req: httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={"jsonrpc": "2.0", "result": {"serverInfo": {"name": "mcp"}}},
            )
        )
        respx_mock.get(f"{OLLAYA_URL}/api/tags").mock(
            side_effect=lambda req: httpx.Response(
                200, json={"models": [{"name": "decider:latest", "details": {}}]}
            )
        )

        async def send_worker(idx: int) -> tuple[str, int]:
            category = idx % 8
            if category == 0:
                p = dict(sample_payloads["choice"])
                p["sla"] = "fast"
                r = await async_client.post("/v1/auto", json=p)
                assert r.headers.get("x-selected-model") == "laya"
                return ("auto_fast", r.status_code)
            elif category == 1:
                p = dict(sample_payloads["choice"])
                p["sla"] = "smart"
                r = await async_client.post("/v1/systemone", json=p)
                assert r.headers.get("x-selected-model") == "decider"
                return ("systemone_smart", r.status_code)
            elif category == 2:
                r = await async_client.post(
                    "/v1/models/pull", json={"model": f"stress-model-{idx}", "stream": False}
                )
                return ("pull_sync", r.status_code)
            elif category == 3:
                r = await async_client.delete(f"/v1/models/stress-model-{idx}:latest")
                return ("delete_model", r.status_code)
            elif category == 4:
                async with async_client.stream("GET", "/mcp") as stream:
                    content = await stream.aread()
                    assert b"message" in content
                    return ("mcp_sse", stream.status_code)
            elif category == 5:
                r = await async_client.post("/mcp", json={"jsonrpc": "2.0", "id": idx})
                return ("mcp_post", r.status_code)
            elif category == 6:
                r = await async_client.get("/v1/capabilities")
                return ("capabilities", r.status_code)
            else:
                r = await async_client.get("/health")
                return ("health", r.status_code)

        tasks = [asyncio.create_task(send_worker(i)) for i in range(50)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 50
        status_codes = [code for _, code in results]
        assert all(c == 200 for c in status_codes), f"Expected all 200, got: {status_codes}"

    @pytest.mark.asyncio
    async def test_50_concurrent_requests_with_upstream_partial_failures(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
    ):
        """Stress: Under partial upstream failures (timeouts, crashes, disconnects),

        all 50 concurrent requests cleanly resolve to expected 200/502/504 statuses
        without crashing the gateway or hanging.
        """

        def flaky_inference(request):
            body = json.loads(request.content.decode("utf-8"))
            state = body.get("state", "")
            if "timeout" in state:
                raise httpx.ReadTimeout("Simulated inference timeout")
            elif "crash" in state:
                return httpx.Response(500, json={"error": "CUDA OOM"})
            elif "disconnect" in state:
                raise httpx.ConnectError("Simulated network drop")
            return httpx.Response(200, json={"answers": {"q": 1}, "model": "decider"})

        respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(side_effect=flaky_inference)

        async def send_flaky(idx: int) -> int:
            mod = idx % 4
            p = dict(sample_payloads["choice"])
            if mod == 0:
                p["state"] = f"normal request {idx}"
                r = await async_client.post("/v1/auto", json=p)
                assert r.status_code == 200
                return r.status_code
            elif mod == 1:
                p["state"] = f"timeout request {idx}"
                r = await async_client.post("/v1/auto", json=p)
                assert r.status_code == 504
                return r.status_code
            elif mod == 2:
                p["state"] = f"crash request {idx}"
                r = await async_client.post("/v1/auto", json=p)
                assert r.status_code == 502
                return r.status_code
            else:
                p["state"] = f"disconnect request {idx}"
                r = await async_client.post("/v1/auto", json=p)
                assert r.status_code == 502
                return r.status_code

        tasks = [asyncio.create_task(send_flaky(i)) for i in range(50)]
        statuses = await asyncio.gather(*tasks)

        assert len(statuses) == 50
        counts = {200: statuses.count(200), 502: statuses.count(502), 504: statuses.count(504)}
        assert counts[200] >= 12
        assert counts[504] >= 12
        assert counts[502] >= 24


# ===========================================================================
# 6. Lifespan Lifecycle, Background Tasks & Clean Shutdown
# ===========================================================================


class TestLifespanAndBackgroundTasks:
    """Verifies FastAPI lifespan context manager, background task cancellation,

    shared client pooling, and header sanitization.
    """

    @pytest.mark.asyncio
    async def test_lifespan_startup_and_shutdown_cleans_tasks_and_client(self):
        """Lifespan manages shared httpx.AsyncClient pool and cleanly cancels background tasks."""
        async with lifespan(app):
            # Assert startup invariants
            assert hasattr(app.state, "client")
            assert isinstance(app.state.client, httpx.AsyncClient)
            assert app.state.client.is_closed is False
            assert len(background_tasks) >= 1

        # Assert shutdown invariants
        assert len(background_tasks) == 0
        assert app.state.client is None

    @pytest.mark.asyncio
    async def test_lifespan_task_cancellation_suppresses_cancelled_error(self):
        """When preload_models_background is cancelled, CancelledError is suppressed."""
        task = asyncio.create_task(preload_models_background())
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

        # Allow task to start running and hit asyncio.sleep
        await asyncio.sleep(0.02)
        assert task in background_tasks
        assert not task.done()

        # Cancel task as lifespan shutdown does
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert task.cancelled()
        assert task not in background_tasks

    @pytest.mark.asyncio
    async def test_multiple_sequential_lifespan_runs(self):
        """Lifespan can be entered and exited repeatedly without residual background state."""
        for _ in range(3):
            async with lifespan(app):
                assert app.state.client is not None
            assert len(background_tasks) == 0
            assert app.state.client is None

    def test_sanitize_upstream_headers_rfc7230_compliance(self):
        """sanitize_upstream_headers strips RFC 7230/9110 hop-by-hop headers case-insensitively."""
        headers = {
            "Connection": "keep-alive",
            "Keep-Alive": "timeout=5, max=1000",
            "Proxy-Authenticate": "Basic",
            "Proxy-Authorization": "Basic xxx",
            "TE": "trailers",
            "Trailers": "X-Checksum",
            "Transfer-Encoding": "chunked",
            "Upgrade": "websocket",
            "Content-Length": "12345",
            "Host": "localhost:8000",
            "Content-Type": "application/json",
            "X-Gateway-Tracking": "true",
            "Cache-Control": "no-cache",
        }
        cleaned = sanitize_upstream_headers(headers)

        for h in HOP_BY_HOP_HEADERS:
            assert h not in cleaned
            assert h.upper() not in cleaned
            assert h.title() not in cleaned

        assert cleaned["Content-Type"] == "application/json"
        assert cleaned["X-Gateway-Tracking"] == "true"
        assert cleaned["Cache-Control"] == "no-cache"

    def test_sanitize_upstream_headers_from_list_of_tuples(self):
        """sanitize_upstream_headers accepts list of tuples."""
        tuples = [
            ("connection", "close"),
            ("x-test", "val"),
            ("transfer-encoding", "gzip"),
        ]
        cleaned = sanitize_upstream_headers(tuples)
        assert "connection" not in cleaned
        assert "transfer-encoding" not in cleaned
        assert cleaned["x-test"] == "val"
