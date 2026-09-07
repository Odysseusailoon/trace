"""Serve the demo directory on localhost:8901 without browser caching."""

from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class DemoHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    handler = partial(DemoHandler, directory=str(directory))
    HTTPServer(("127.0.0.1", 8901), handler).serve_forever()
