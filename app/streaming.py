"""
Decision Gateway Streaming Lifecycle & Safe Response Utilities.
Implements Milestone M3 (Features F4, F5, F6, F7).

Provides:
- Hop-by-hop header sanitization (RFC 7230/9110).
- SSE framing utilities.
- SafeStreamingResponse: Guarantees generator aclose() on all exit paths.
- sse_stream_bridge: Event-driven concurrent disconnect listener and mid-stream
  error shielding for SSE endpoints (/mcp).
- ndjson_stream_bridge: Disconnect-aware bridge for NDJSON endpoints (/v1/models/pull).
"""

import asyncio
import inspect
import json
import logging
from typing import Any, AsyncIterator, Iterable, Mapping, Optional, Set, Union

import httpx
from starlette.requests import ClientDisconnect, Request
from starlette.responses import StreamingResponse
from starlette.types import Send

logger = logging.getLogger("app.streaming")

# RFC 7230 / RFC 9110 hop-by-hop headers that must never be forwarded by a proxy
HOP_BY_HOP_HEADERS: Set[str] = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}


def sanitize_upstream_headers(
    headers: Union[Mapping[str, str], Iterable[tuple[str, str]]],
) -> dict[str, str]:
    """
    Strips hop-by-hop headers from upstream response headers to prevent
    HTTP framing errors and proxy conflicts (RFC 7230 / RFC 9110).
    """
    if hasattr(headers, "items"):
        items = headers.items()  # type: ignore[union-attr]
    else:
        items = headers  # type: ignore[assignment]
    return {str(k): str(v) for k, v in items if str(k).lower() not in HOP_BY_HOP_HEADERS}


def format_sse_event(
    event: Optional[str] = None,
    data: Any = "",
    event_id: Optional[str] = None,
    retry: Optional[int] = None,
) -> bytes:
    """
    Formats a standards-compliant Server-Sent Events (SSE) data frame.
    """
    lines: list[str] = []
    if event is not None:
        lines.append(f"event: {event}")
    if event_id is not None:
        lines.append(f"id: {event_id}")
    if retry is not None:
        lines.append(f"retry: {retry}")

    if isinstance(data, (dict, list)):
        payload = json.dumps(data)
    else:
        payload = str(data)

    if payload:
        for line in payload.splitlines():
            lines.append(f"data: {line}")
    elif event is None and event_id is None and retry is None:
        lines.append("data: ")

    return ("\n".join(lines) + "\n\n").encode("utf-8")


class SafeStreamingResponse(StreamingResponse):
    """
    Hardened StreamingResponse ensuring async generator cleanup and socket
    closure on broken pipe, client disconnect, or mid-stream exceptions.

    Guarantees:
    1. Hop-by-hop headers are filtered during header initialization.
    2. await self.body_iterator.aclose() is always invoked in a finally: block.
    3. ClientDisconnect and OSError during send() are handled cleanly without
       unhandled ASGI application crashes.
    """

    def __init__(
        self,
        content: Any,
        status_code: int = 200,
        headers: Optional[Union[Mapping[str, str], Iterable[tuple[str, str]]]] = None,
        media_type: Optional[str] = None,
        background: Any = None,
    ) -> None:
        clean_headers = sanitize_upstream_headers(headers) if headers is not None else None
        super().__init__(
            content=content,
            status_code=status_code,
            headers=clean_headers,
            media_type=media_type,
            background=background,
        )

    async def stream_response(self, send: Send) -> None:
        try:
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                }
            )
            async for chunk in self.body_iterator:
                if not isinstance(chunk, (bytes, memoryview)):
                    chunk = chunk.encode(self.charset)
                await send(
                    {
                        "type": "http.response.body",
                        "body": chunk,
                        "more_body": True,
                    }
                )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                }
            )
        except (ClientDisconnect, OSError):
            logger.info("Client disconnected during streaming response transmission.")
        finally:
            if hasattr(self.body_iterator, "aclose"):
                try:
                    res = self.body_iterator.aclose()
                    if inspect.isawaitable(res):
                        await res
                except (Exception, asyncio.CancelledError) as exc:
                    logger.warning("Error during body_iterator aclose: %s", exc)


async def sse_stream_bridge(
    request: Request,
    upstream_resp: httpx.Response,
    client: Optional[httpx.AsyncClient] = None,
) -> AsyncIterator[bytes]:
    """
    Bridges upstream SSE stream chunks to the client with immediate client disconnect
    detection (<100ms) during quiet windows, and maps mid-stream upstream crashes
    to structured SSE error frames (Feature F6).
    """

    async def watch_disconnect() -> None:
        try:
            while True:
                message = await request.receive()
                if message.get("type") == "http.disconnect":
                    return
        except asyncio.CancelledError:
            pass
        except Exception:
            return

    watch_task = asyncio.create_task(watch_disconnect())
    upstream_iter = upstream_resp.aiter_raw().__aiter__()
    chunk_task: Optional[asyncio.Task[Any]] = None

    async def next_chunk() -> bytes:
        return await upstream_iter.__anext__()

    try:
        while not watch_task.done():
            chunk_task = asyncio.create_task(next_chunk())
            done, pending = await asyncio.wait(
                [chunk_task, watch_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # If client disconnect was detected, abort upstream reading immediately
            if watch_task in done:
                logger.info(
                    "Client disconnect detected during SSE quiet window; aborting upstream."
                )
                chunk_task.cancel()
                try:
                    await chunk_task
                except (asyncio.CancelledError, StopAsyncIteration, Exception):
                    pass
                chunk_task = None
                break

            try:
                chunk = chunk_task.result()
                chunk_task = None
                yield chunk
            except StopAsyncIteration:
                # Clean upstream end-of-stream
                break
            except httpx.TimeoutException as exc:
                logger.warning("Upstream MCP stream read timeout: %s", exc)
                yield format_sse_event(
                    event="error",
                    data={
                        "error": "gateway_timeout",
                        "detail": f"Upstream MCP server timed out: {exc}",
                    },
                )
                break
            except (httpx.RemoteProtocolError, httpx.NetworkError, httpx.ReadError) as exc:
                logger.warning("Upstream MCP server connection lost mid-stream: %s", exc)
                yield format_sse_event(
                    event="error",
                    data={"error": "bad_gateway", "detail": f"Upstream disconnected: {exc}"},
                )
                break
            except httpx.RequestError as exc:
                logger.warning("Upstream MCP stream request error: %s", exc)
                yield format_sse_event(
                    event="error",
                    data={"error": "bad_gateway", "detail": f"Upstream error: {exc}"},
                )
                break
    finally:
        if chunk_task is not None and not chunk_task.done():
            chunk_task.cancel()
            try:
                await chunk_task
            except (asyncio.CancelledError, StopAsyncIteration, Exception):
                pass
        if not watch_task.done():
            watch_task.cancel()
            try:
                await watch_task
            except (asyncio.CancelledError, Exception):
                pass
        await upstream_resp.aclose()
        if client is not None:
            await client.aclose()


async def ndjson_stream_bridge(
    request: Request,
    upstream_resp: httpx.Response,
    client: Optional[httpx.AsyncClient] = None,
) -> AsyncIterator[bytes]:
    """
    Bridges raw/NDJSON chunks from upstream (e.g. model pull progress) while watching
    for early client abortion and guaranteeing immediate upstream resource cleanup.
    """

    async def watch_disconnect() -> None:
        try:
            while True:
                message = await request.receive()
                if message.get("type") == "http.disconnect":
                    return
        except asyncio.CancelledError:
            pass
        except Exception:
            return

    watch_task = asyncio.create_task(watch_disconnect())
    upstream_iter = upstream_resp.aiter_raw().__aiter__()
    chunk_task: Optional[asyncio.Task[Any]] = None

    async def next_chunk() -> bytes:
        return await upstream_iter.__anext__()

    try:
        while not watch_task.done():
            chunk_task = asyncio.create_task(next_chunk())
            done, pending = await asyncio.wait(
                [chunk_task, watch_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if watch_task in done:
                logger.info("Client aborted model pull stream; cancelling upstream pull.")
                chunk_task.cancel()
                try:
                    await chunk_task
                except (asyncio.CancelledError, StopAsyncIteration, Exception):
                    pass
                chunk_task = None
                break

            try:
                chunk = chunk_task.result()
                chunk_task = None
                yield chunk
            except StopAsyncIteration:
                break
            except (httpx.TimeoutException, httpx.RemoteProtocolError, httpx.RequestError) as exc:
                logger.warning("Upstream error during model pull stream: %s", exc)
                break
    finally:
        if chunk_task is not None and not chunk_task.done():
            chunk_task.cancel()
            try:
                await chunk_task
            except (asyncio.CancelledError, StopAsyncIteration, Exception):
                pass
        if not watch_task.done():
            watch_task.cancel()
            try:
                await watch_task
            except (asyncio.CancelledError, Exception):
                pass
        await upstream_resp.aclose()
        if client is not None:
            await client.aclose()
