import http.server, os
os.chdir(os.path.expanduser("~/demo"))
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()
http.server.HTTPServer(("0.0.0.0", 8901), H).serve_forever()
