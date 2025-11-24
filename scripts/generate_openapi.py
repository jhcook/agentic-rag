
import sys
import os
import yaml
import json
from pathlib import Path

# Add the project root to the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.servers.rest_server import app

def generate_openapi():
    openapi_schema = app.openapi()
    
    # Save as YAML
    with open("openapi.yaml", "w") as f:
        yaml.dump(openapi_schema, f, sort_keys=False)
    
    # Save as JSON (optional, but good to have)
    with open("openapi.json", "w") as f:
        json.dump(openapi_schema, f, indent=2)
        
    print("OpenAPI specification generated in openapi.yaml and openapi.json")

if __name__ == "__main__":
    generate_openapi()
