from pathlib import Path
import sys,json,time
Path('/run/itf-bootstrap-status.json').write_text(json.dumps(dict(stage=sys.argv[1],detail=sys.argv[2] if len(sys.argv)>2 else '',utc=time.time())))
