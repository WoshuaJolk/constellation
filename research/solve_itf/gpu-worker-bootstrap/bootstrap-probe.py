from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
import json
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  if self.path!='/bootstrap-status':self.send_error(404);return
  try:data=Path('/run/itf-bootstrap-status.json').read_bytes()
  except FileNotFoundError:data=b'{"stage":"starting"}'
  self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
HTTPServer(('0.0.0.0',80),Handler).serve_forever()
