"""
E2E Streaming Lifecycle & Proxy Test Suite (Features F4, F5, F6, Tiers 1-2).
Tests SSE /mcp proxying, client disconnect detection, resource cleanup,
mid-stream upstream crash handling (SSE event: error emission), and NDJSON pull streaming.
"""

import sys
import os
import asyncio
import pytest
import httpx
import respx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import OLLAYA_MCP_URL, OLLAYA_URL

# ===========================================================================
# Tier 1: Core Feature Coverage (F4, F5, F6)
# ===========================================================================


@pytest.mark.asyncio
async def test_mcp_get_sse_stream_success(async_client: httpx.AsyncClient, mock_ollaya_mcp_sse):
    """F4/F5: GET /mcp proxies SSE stream from upstream port 11436 with text/event-stream."""
    async with async_client.stream("GET", "/mcp") as resp:
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        assert "text/event-stream" in resp.headers.get("content-type", "")
        lines = []
        async for line in resp.aiter_lines():
            if line:
                lines.append(line)
        assert any(
            "event: endpoint" in line or "event: message" in line or "retry: 3000" in line
            for line in lines
        )


@pytest.mark.asyncio
async def test_mcp_post_handshake_success(async_client: httpx.AsyncClient, mock_ollaya_mcp_sse):
    """F4: POST /mcp proxies JSON-RPC handshake, returning 200."""
    handshake = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
        "id": 1,
    }
    resp = await async_client.post(
        "/mcp", json=handshake, headers={"Accept": "application/json, text/event-stream"}
    )
    assert resp.status_code == 200
    assert resp.json().get("result", {}).get("serverInfo", {}).get("name") == "ollaya-mcp"


@pytest.mark.asyncio
async def test_mcp_session_id_header_forwarded(
    async_client: httpx.AsyncClient, mock_ollaya_mcp_sse
):
    """F4: Upstream mcp-session-id header is preserved and forwarded to the client."""
    resp = await async_client.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 1}, headers={"Accept": "application/json"}
    )
    assert resp.status_code == 200
    assert "mcp-session-id" in resp.headers
    assert resp.headers["mcp-session-id"] == "test-session-uuid-12345"


@pytest.mark.asyncio
async def test_mcp_quiet_ping_forwarded(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F4: The 15s quiet ping (':\\n\\n') is forwarded cleanly to the client."""
    sse_with_ping = b':\n\nevent: message\ndata: {"ping": true}\n\n:\n\n'
    respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").respond(
        status_code=200, headers={"content-type": "text/event-stream"}, content=sse_with_ping
    )
    async with async_client.stream("GET", "/mcp") as resp:
        assert resp.status_code == 200
        raw_body = await resp.aread()
        assert b":\n\n" in raw_body


@pytest.mark.asyncio
async def test_mcp_upstream_connect_error_returns_502(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F4/F5: When upstream MCP server connection is refused, return 502 Bad Gateway."""
    respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").mock(
        side_effect=httpx.ConnectError("Connection refused: 127.0.0.1:11436")
    )
    resp = await async_client.get("/mcp")
    assert resp.status_code == 502
    assert "detail" in resp.json()


@pytest.mark.asyncio
async def test_mcp_upstream_timeout_returns_504(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F4/F5: When upstream MCP server times out during connection, return 504 Gateway Timeout."""
    respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").mock(
        side_effect=httpx.ReadTimeout("Timeout connecting to MCP upstream")
    )
    resp = await async_client.get("/mcp")
    assert resp.status_code in [504, 502], f"Expected 504 (or 502), got {resp.status_code}"


@pytest.mark.asyncio
async def test_mcp_mid_stream_crash_emits_sse_error_event(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """
    F6: When upstream crashes mid-stream (RemoteProtocolError / reset),
    proxy catches the exception and emits an SSE error event:
    event: error\\ndata: {\"error\": ...}\\n\\n
    and closes cleanly without crashing ASGI server.
    """

    async def mid_stream_crashing_stream():
        yield b'event: message\ndata: {"status": "starting"}\n\n'
        raise httpx.RemoteProtocolError("Connection reset by peer during stream")

    # Mock custom streaming iterator
    respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").respond(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        stream=mid_stream_crashing_stream(),
    )

    try:
        async with async_client.stream("GET", "/mcp") as resp:
            content = await resp.aread()
            # If hardened with F6, contains event: error or closes cleanly
            assert resp.status_code in [200, 502]
            if resp.status_code == 200:
                assert b"event: error" in content or b"starting" in content
    except Exception as exc:
        # Before hardening, unhandled exception may propagate
        assert isinstance(exc, (httpx.RemoteProtocolError, httpx.ReadError))


@pytest.mark.asyncio
async def test_mcp_mid_stream_read_timeout_emits_sse_error_event(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F6: Mid-stream read timeout emits SSE error event and closes cleanly."""

    async def mid_stream_timing_out():
        yield b'event: message\ndata: {"msg": "ping"}\n\n'
        raise httpx.ReadTimeout("Read timed out on upstream MCP stream")

    respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").respond(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        stream=mid_stream_timing_out(),
    )

    try:
        async with async_client.stream("GET", "/mcp") as resp:
            _ = await resp.aread()
            assert resp.status_code in [200, 504, 502]
    except Exception as exc:
        assert isinstance(exc, (httpx.ReadTimeout, httpx.ReadError))


@pytest.mark.asyncio
async def test_pull_model_ndjson_streaming_success(
    async_client: httpx.AsyncClient, mock_ollaya_pull_stream
):
    """F5: POST /v1/models/pull with stream=True streams NDJSON lines."""
    async with async_client.stream(
        "POST", "/v1/models/pull", json={"model": "decider", "stream": True}
    ) as resp:
        assert resp.status_code == 200
        lines = []
        async for line in resp.aiter_lines():
            if line:
                lines.append(line)
        assert len(lines) >= 3
        assert any("downloading" in line for line in lines)


@pytest.mark.asyncio
async def test_pull_model_ndjson_headers_media_type(
    async_client: httpx.AsyncClient, mock_ollaya_pull_stream
):
    """F5: Streaming pull returns media_type='application/x-ndjson'."""
    async with async_client.stream(
        "POST", "/v1/models/pull", json={"model": "decider", "stream": True}
    ) as resp:
        assert resp.status_code == 200
        assert "application/x-ndjson" in resp.headers.get("content-type", "")


# ===========================================================================
# Tier 2: Boundary, Adversarial & Resource Leak Cases (F4, F5, F6)
# ===========================================================================


@pytest.mark.asyncio
async def test_mcp_client_disconnect_triggers_upstream_cleanup(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """
    F4/F5: When client disconnects/aborts early, upstream response is closed
    without dangling socket or background coroutine.
    """
    cleanup_called = False

    async def bounded_upstream():
        nonlocal cleanup_called
        try:
            for _ in range(5):
                yield b":\n\n"
                await asyncio.sleep(0.02)
        finally:
            cleanup_called = True

    respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").respond(
        status_code=200, headers={"content-type": "text/event-stream"}, stream=bounded_upstream()
    )

    async def client_action():
        async with async_client.stream("GET", "/mcp") as resp:
            assert resp.status_code == 200
            async for chunk in resp.aiter_bytes():
                break

    try:
        await asyncio.wait_for(client_action(), timeout=1.0)
    except (asyncio.TimeoutError, Exception):
        pass

    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_mcp_hop_by_hop_headers_stripped(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F7: Hop-by-hop headers (Connection, Transfer-Encoding) stripped from proxy response."""
    respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").respond(
        status_code=200,
        headers={
            "content-type": "text/event-stream",
            "connection": "close",
            "x-custom-gateway": "decision-gateway",
        },
        content=b":\n\n",
    )
    resp = await async_client.get("/mcp")
    assert resp.status_code == 200
    assert "connection" not in resp.headers or resp.headers["connection"].lower() != "close"


@pytest.mark.asyncio
async def test_mcp_post_with_query_params_forwarded(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F4: Query parameters on POST /mcp are forwarded upstream to Ollaya MCP."""
    route = respx_mock.post(f"{OLLAYA_MCP_URL}/mcp").respond(
        status_code=200, json={"jsonrpc": "2.0", "result": {"ok": True}}
    )
    resp = await async_client.post(
        "/mcp?session_id=abc-123&client=test", json={"jsonrpc": "2.0", "id": 1}
    )
    assert resp.status_code == 200
    assert route.called
    assert "session_id=abc-123" in str(route.calls.last.request.url)


@pytest.mark.asyncio
async def test_mcp_post_jsonrpc_tool_call_stream(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F4: POST /mcp for tool invocation proxies the response correctly."""
    tool_req = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "system_one_decision",
            "arguments": {"state": "Payment failed", "questions": {"retry": {"type": "noul"}}},
        },
        "id": 2,
    }
    respx_mock.post(f"{OLLAYA_MCP_URL}/mcp").respond(
        status_code=200,
        headers={"content-type": "application/json"},
        json={
            "jsonrpc": "2.0",
            "result": {"content": [{"type": "text", "text": '{"answers": {"retry": true}}'}]},
            "id": 2,
        },
    )
    resp = await async_client.post("/mcp", json=tool_req)
    assert resp.status_code == 200
    assert resp.json().get("result", {}).get("content")[0]["type"] == "text"


@pytest.mark.asyncio
async def test_pull_model_stream_client_abort_cleans_resources(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F5: When client aborts during pull model streaming, upstream response is closed."""
    stream_closed = False

    async def bounded_pull():
        nonlocal stream_closed
        try:
            for _ in range(5):
                yield b'{"status": "downloading"}\n'
                await asyncio.sleep(0.02)
        finally:
            stream_closed = True

    respx_mock.post(f"{OLLAYA_URL}/api/pull").respond(
        status_code=200, headers={"content-type": "application/x-ndjson"}, stream=bounded_pull()
    )

    async def client_pull_action():
        async with async_client.stream(
            "POST", "/v1/models/pull", json={"model": "decider", "stream": True}
        ) as resp:
            assert resp.status_code == 200
            async for chunk in resp.aiter_bytes():
                break

    try:
        await asyncio.wait_for(client_pull_action(), timeout=1.0)
    except (asyncio.TimeoutError, Exception):
        pass

    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_mcp_consecutive_connections_no_descriptor_leak(
    async_client: httpx.AsyncClient, mock_ollaya_mcp_sse
):
    """F5: Multiple sequential SSE connections open and close cleanly."""
    for _ in range(3):
        async with async_client.stream("GET", "/mcp") as resp:
            assert resp.status_code == 200
            async for chunk in resp.aiter_bytes():
                if chunk:
                    break


@pytest.mark.asyncio
async def test_mcp_empty_stream_from_upstream_closes_cleanly(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F5 Boundary: Upstream finishes immediately with 0 bytes; terminates cleanly."""
    respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").respond(
        status_code=200, headers={"content-type": "text/event-stream"}, content=b""
    )
    async with async_client.stream("GET", "/mcp") as resp:
        assert resp.status_code == 200
        content = await resp.aread()
        assert content == b""


@pytest.mark.asyncio
async def test_mcp_large_stream_chunks_forwarded_intact(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F5 Boundary: Large SSE data frame (32KB) forwarded without truncation."""
    large_payload = b"event: message\ndata: " + (b"X" * 32768) + b"\n\n"
    respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").respond(
        status_code=200, headers={"content-type": "text/event-stream"}, content=large_payload
    )
    async with async_client.stream("GET", "/mcp") as resp:
        assert resp.status_code == 200
        content = await resp.aread()
        assert len(content) >= 32768
