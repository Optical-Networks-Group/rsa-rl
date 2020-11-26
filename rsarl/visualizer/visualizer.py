
import sys
from rsarl.logger import RSADB
from rsarl.visualizer import build_dash

def run_cli():
    db_path = sys.argv[1]
    db = RSADB(db_path)
    # build & run dash
    app = build_dash(db)
    # Turn off reloader if inside Jupyter
    app.run_server(debug=False, use_reloader=False)  

def run(db_path='rsa-rl.db', is_debug=False):
    db = RSADB()
    # build & run dash
    app = build_dash(db)
    # Turn off reloader if inside Jupyter
    app.run_server(debug=is_debug, use_reloader=False)  
