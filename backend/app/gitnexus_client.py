import json
import logging
import queue
import shlex
import subprocess
import threading
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any


logger = logging.getLogger("legacylens.gitnexus")


class GitNexusClientError(RuntimeError):
    pass


@dataclass
class GitNexusToolCall:
    name: str
    args: dict[str, Any]


class GitNexusMCPTransport:
    """Minimal MCP stdio client with serialized requests and auto-restart."""

    def __init__(
        self,
        command: str,
        startup_timeout_seconds: float = 20.0,
        call_timeout_seconds: float = 20.0,
    ) -> None:
        self.command = command
        self.startup_timeout_seconds = max(float(startup_timeout_seconds), 3.0)
        self.call_timeout_seconds = max(float(call_timeout_seconds), 3.0)

        self._proc: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stop_reader = threading.Event()
        self._send_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._next_id = 1
        self._initialized = False

    def close(self) -> None:
        with self._state_lock:
            self._shutdown_locked()

    def restart(self) -> None:
        with self._state_lock:
            self._shutdown_locked()
            self._start_locked()

    def list_tools(self) -> list[dict[str, Any]]:
        payload = self.request("tools/list", params={})
        return list((payload.get("result", {}) or {}).get("tools", []) or [])

    def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        payload = self.request(
            "tools/call",
            params={"name": name, "arguments": args},
        )
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        if not isinstance(result, dict):
            raise GitNexusClientError(f"Invalid MCP result for tool={name!r}")

        content = result.get("content", [])
        if not isinstance(content, list) or not content:
            raise GitNexusClientError(f"Missing MCP content for tool={name!r}")
        first = content[0] if isinstance(content[0], dict) else {}
        text = str(first.get("text", "")).strip()
        if not text:
            return {}
        if text.startswith("Error:"):
            raise GitNexusClientError(text)
        return parse_tool_text_json(text)

    def request(self, method: str, params: dict[str, Any], timeout_seconds: float | None = None) -> dict[str, Any]:
        # Ensure process exists + handshake before sending.
        with self._state_lock:
            self._ensure_ready_locked()

        return self._send_request_no_ensure(method=method, params=params, timeout_seconds=timeout_seconds)

    def _send_request_no_ensure(
        self,
        method: str,
        params: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        timeout = timeout_seconds if timeout_seconds is not None else self.call_timeout_seconds
        request_id, wait_q = self._allocate_request_slot()
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        try:
            with self._send_lock:
                self._write_message(message)
            payload = wait_q.get(timeout=max(timeout, 1.0))
        except queue.Empty as exc:
            self._remove_pending(request_id)
            raise GitNexusClientError(f"MCP timeout for method={method!r}") from exc
        finally:
            self._remove_pending(request_id)

        if not isinstance(payload, dict):
            raise GitNexusClientError(f"Invalid MCP payload for method={method!r}")
        if "error" in payload:
            raise GitNexusClientError(f"MCP error for method={method!r}: {payload['error']}")
        return payload

    def _allocate_request_slot(self) -> tuple[int, queue.Queue[dict[str, Any]]]:
        with self._pending_lock:
            request_id = self._next_id
            self._next_id += 1
            wait_q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
            self._pending[request_id] = wait_q
            return request_id, wait_q

    def _remove_pending(self, request_id: int) -> None:
        with self._pending_lock:
            self._pending.pop(request_id, None)

    def _dispatch_incoming(self, payload: dict[str, Any]) -> None:
        request_id = payload.get("id")
        if not isinstance(request_id, int):
            return
        with self._pending_lock:
            wait_q = self._pending.get(request_id)
        if wait_q is None:
            return
        try:
            wait_q.put_nowait(payload)
        except queue.Full:
            pass

    def _ensure_ready_locked(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._start_locked()
        if not self._initialized:
            self._initialize_locked()

    def _start_locked(self) -> None:
        args = resolve_gitnexus_command(self.command)
        if not args:
            raise GitNexusClientError("GITNEXUS_MCP_COMMAND is empty")

        self._stop_reader.clear()
        self._initialized = False
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()

    def _shutdown_locked(self) -> None:
        self._initialized = False
        self._stop_reader.set()
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        with self._pending_lock:
            for wait_q in self._pending.values():
                try:
                    wait_q.put_nowait({"error": {"message": "transport closed"}})
                except queue.Full:
                    pass
            self._pending.clear()

    def _initialize_locked(self) -> None:
        payload = self._send_request_no_ensure(
            "initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "legacylens-backend", "version": "0.1"},
            },
            timeout_seconds=self.startup_timeout_seconds,
        )
        if "result" not in payload:
            raise GitNexusClientError("MCP initialize failed: missing result")
        with self._send_lock:
            self._write_message(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                }
            )
        self._initialized = True

    def _write_message(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise GitNexusClientError("MCP process is not running")
        # GitNexus MCP uses newline-delimited JSON over stdio.
        data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    def _reader_loop(self) -> None:
        while not self._stop_reader.is_set():
            try:
                payload = self._read_message()
            except Exception as exc:
                logger.warning("GitNexus MCP reader stopped: %s", exc)
                break
            if payload is None:
                break
            self._dispatch_incoming(payload)

    def _stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while not self._stop_reader.is_set():
            line = proc.stderr.readline()
            if not line:
                break
            message = line.decode("utf-8", "replace").strip()
            if message:
                logger.debug("gitnexus[mcp]: %s", message)

    def _read_message(self) -> dict[str, Any] | None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return None
        while True:
            line = proc.stdout.readline()
            if not line:
                return None

            decoded = line.decode("utf-8", "replace").strip()
            if not decoded:
                continue

            # Backward compatibility for framed stdio servers.
            if decoded.lower().startswith("content-length:"):
                headers = {"content-length": decoded.split(":", 1)[1].strip()}
                while True:
                    header_line = proc.stdout.readline()
                    if not header_line:
                        return None
                    header_decoded = header_line.decode("utf-8", "replace")
                    if header_decoded in ("\r\n", "\n"):
                        break
                    if ":" not in header_decoded:
                        continue
                    key, value = header_decoded.split(":", 1)
                    headers[key.strip().lower()] = value.strip()

                try:
                    length = int(headers.get("content-length", "0"))
                except ValueError:
                    continue
                if length <= 0:
                    continue

                body = proc.stdout.read(length)
                if not body:
                    return None
                try:
                    parsed = json.loads(body.decode("utf-8", "replace"))
                except JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
                continue

            try:
                parsed = json.loads(decoded)
            except JSONDecodeError:
                # Ignore non-JSON stdout lines and continue reading.
                continue
            if isinstance(parsed, dict):
                return parsed


def parse_tool_text_json(text: str) -> Any:
    stripped = text.lstrip()
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(stripped)
        return parsed
    except Exception:
        # Fallback when non-JSON text is returned.
        return {"raw_text": text}


def resolve_gitnexus_command(command: str) -> list[str]:
    configured = str(command or "").strip()
    if not configured:
        return []
    lower = configured.lower()
    if "gitnexus@latest" in lower and "mcp" in lower:
        try:
            candidates = sorted(
                Path.home().glob(".npm/_npx/*/node_modules/gitnexus/dist/cli/index.js"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except Exception:
            candidates = []
        if candidates:
            return ["node", str(candidates[0]), "mcp"]
    return shlex.split(configured)


class GitNexusClient:
    def __init__(
        self,
        command: str,
        default_repo: str | None = None,
        startup_timeout_seconds: float = 20.0,
        call_timeout_seconds: float = 20.0,
    ) -> None:
        self.default_repo = default_repo.strip() if isinstance(default_repo, str) and default_repo.strip() else None
        self.transport = GitNexusMCPTransport(
            command=command,
            startup_timeout_seconds=startup_timeout_seconds,
            call_timeout_seconds=call_timeout_seconds,
        )

    def close(self) -> None:
        self.transport.close()

    def list_tools(self) -> list[dict[str, Any]]:
        return self.transport.list_tools()

    def list_repos(self) -> list[dict[str, Any]]:
        result = self._call(GitNexusToolCall(name="list_repos", args={}))
        return result if isinstance(result, list) else []

    def query(
        self,
        query: str,
        repo: str | None = None,
        task_context: str | None = None,
        goal: str | None = None,
        limit: int = 5,
        max_symbols: int = 10,
        include_content: bool = False,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "query": query,
            "limit": max(1, int(limit)),
            "max_symbols": max(1, int(max_symbols)),
            "include_content": bool(include_content),
        }
        if task_context:
            args["task_context"] = task_context
        if goal:
            args["goal"] = goal
        selected_repo = self._selected_repo(repo)
        if selected_repo:
            args["repo"] = selected_repo
        result = self._call(GitNexusToolCall(name="query", args=args))
        return result if isinstance(result, dict) else {}

    def context(
        self,
        name: str,
        repo: str | None = None,
        file_path: str | None = None,
        include_content: bool = False,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"name": name, "include_content": bool(include_content)}
        if file_path:
            args["file_path"] = file_path
        selected_repo = self._selected_repo(repo)
        if selected_repo:
            args["repo"] = selected_repo
        result = self._call(GitNexusToolCall(name="context", args=args))
        return result if isinstance(result, dict) else {}

    def impact(
        self,
        target: str,
        direction: str,
        repo: str | None = None,
        max_depth: int = 3,
        min_confidence: float = 0.7,
        include_tests: bool = False,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "target": target,
            "direction": direction,
            "maxDepth": max(1, int(max_depth)),
            "minConfidence": max(0.0, min(1.0, float(min_confidence))),
            "includeTests": bool(include_tests),
        }
        selected_repo = self._selected_repo(repo)
        if selected_repo:
            args["repo"] = selected_repo
        result = self._call(GitNexusToolCall(name="impact", args=args))
        return result if isinstance(result, dict) else {}

    def _selected_repo(self, repo: str | None) -> str | None:
        if isinstance(repo, str) and repo.strip():
            return repo.strip()
        return self.default_repo

    def _call(self, call: GitNexusToolCall) -> Any:
        # Retry once after transport restart to recover from dead child process.
        for attempt in range(2):
            try:
                return self.transport.call_tool(call.name, call.args)
            except GitNexusClientError:
                if attempt >= 1:
                    raise
                logger.warning("GitNexus MCP tool failed (%s). Restarting transport.", call.name)
                self.transport.restart()
        return {}
