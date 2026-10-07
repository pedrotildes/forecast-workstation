"""Desktop entry point for the packaged Forecast Workstation.

Starts the FastAPI server (which also serves the front-end) on a local port in a
background thread and opens it in a native window via pywebview. Set
``FORECAST_HEADLESS=1`` to run only the server (used to test the bundle).
"""
from __future__ import annotations

import multiprocessing
import os
import socket
import threading
import time
import urllib.request


def _free_port(preferred: int = 8766) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def _serve(port: int) -> None:
    import uvicorn

    from app.main import app
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # not on the main thread
    server.run()


def _wait_until_ready(port: int, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    return False


def main() -> None:
    # HTTPS for urllib (Cartopy's Natural Earth downloads) inside the frozen bundle
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    port = _free_port()
    if os.environ.get("FORECAST_HEADLESS"):
        print(f"Forecast Workstation (headless) em http://127.0.0.1:{port}", flush=True)
        _serve(port)
        return

    import webview

    threading.Thread(target=_serve, args=(port,), daemon=True).start()
    _wait_until_ready(port)
    webview.create_window(
        "Forecast Workstation",
        f"http://127.0.0.1:{port}",
        width=1500,
        height=950,
        min_size=(1100, 700),
    )
    webview.start()


if __name__ == "__main__":
    # Required: the map/sounding renderers run in a spawn-based process pool, and in a
    # frozen app every child process re-executes this entry point.
    multiprocessing.freeze_support()
    main()
