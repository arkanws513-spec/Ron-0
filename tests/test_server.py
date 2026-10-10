from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ron.server import make_server


def test_http_api_uses_python_core_and_reports_native_status():
    server = make_server(host="127.0.0.1", port=0, require_native=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/api/health", timeout=3) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert health["runtime"] == "ron-0-python-core"
        assert health["single_core"] is True
        assert health["external_model_used"] is False

        request = Request(
            base + "/api/chat",
            data=json.dumps({"message": "مرحبا", "session_id": "test_session"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert isinstance(payload["reply"], str) and payload["reply"].strip()
        assert payload["metadata"]["native_weights_loaded"] is health["native_weights_loaded"]

        # Python source and training artifacts must not be served as static files.
        try:
            urlopen(base + "/ron/core.py", timeout=3)
            assert False, "private implementation file unexpectedly served"
        except HTTPError as exc:
            assert exc.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_http_api_rejects_invalid_chat_payload():
    server = make_server(host="127.0.0.1", port=0, require_native=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        request = Request(
            base + "/api/chat",
            data=json.dumps({"message": "x" * 4001, "session_id": "bad"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(request, timeout=3)
            assert False, "oversized message unexpectedly accepted"
        except HTTPError as exc:
            assert exc.code == 400
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload["error"] == "invalid_message"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
