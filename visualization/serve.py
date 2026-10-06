"""Zero-dependency local server to launch the 3D-Brain visualizer on macOS."""

import http.server
from pathlib import Path
import socketserver
import sys
import webbrowser

PORT = 8080


class Handler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        root_dir = Path(__file__).resolve().parent.parent
        super().__init__(*args, directory=str(root_dir), **kwargs)


class ThreadedServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    with ThreadedServer(("", port), Handler) as httpd:
        url = f"http://localhost:{port}/visualization/player.html"
        print(f"=======================================================")
        print(f"  3D-Brain Interactive Observability Console Running   ")
        print(f"  Open in browser: {url}")
        print(f"=======================================================")
        if "--open" in sys.argv:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    main()
