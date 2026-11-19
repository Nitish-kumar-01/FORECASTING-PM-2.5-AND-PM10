"""
Configuration and Precaution Loaders for PM2.5 and PM10 Forecasting System.
Provides access to config.yaml and precautions.json.
"""

import json
from pathlib import Path
from typing import Any, Dict

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
PRECAUTIONS_PATH = CONFIG_DIR / "precautions.json"


def load_precautions(path: Path = PRECAUTIONS_PATH) -> Dict[str, Any]:
    """Load precautions JSON file with CPCB health advisories."""
    if not path.exists():
        raise FileNotFoundError(f"Precautions file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_yaml_fallback(text: str) -> Dict[str, Any]:
    """
    Fallback parser for basic YAML structures when PyYAML is not yet installed.
    Handles key-value pairs, nested dictionaries, numbers, booleans, and string lists.
    """
    import re
    result: Dict[str, Any] = {}
    stack = [(-1, result)]
    
    for raw_line in text.splitlines():
        # Strip comments
        line = re.sub(r'#.*$', '', raw_line).rstrip()
        if not line.strip():
            continue
        
        indent = len(line) - len(line.lstrip())
        content = line.strip()
        
        # Pop stack until finding parent with strictly smaller indent
        while stack and stack[-1][0] >= indent:
            stack.pop()
        current_dict = stack[-1][1]
        
        # Key-value or section header
        if ":" in content:
            key, val = content.split(":", 1)
            key = key.strip()
            val = val.strip()
            
            if not val:
                new_dict: Dict[str, Any] = {}
                if isinstance(current_dict, dict):
                    current_dict[key] = new_dict
                stack.append((indent, new_dict))
            else:
                # Value parsing
                parsed_val: Any = val
                if val.lower() == "true":
                    parsed_val = True
                elif val.lower() == "false":
                    parsed_val = False
                elif val.lower() in ("null", "none"):
                    parsed_val = None
                elif val.startswith("[") and val.endswith("]"):
                    # Basic list
                    items = [item.strip().strip("'\"") for item in val[1:-1].split(",") if item.strip()]
                    parsed_items = []
                    for item in items:
                        try:
                            if "." in item:
                                parsed_items.append(float(item))
                            else:
                                parsed_items.append(int(item))
                        except ValueError:
                            parsed_items.append(item)
                    parsed_val = parsed_items
                else:
                    val_clean = val.strip("'\"")
                    try:
                        if "." in val_clean:
                            parsed_val = float(val_clean)
                        else:
                            parsed_val = int(val_clean)
                    except ValueError:
                        parsed_val = val_clean
                        
                if isinstance(current_dict, dict):
                    current_dict[key] = parsed_val
        elif content.startswith("- "):
            val = content[2:].strip().strip("'\"")
            if isinstance(current_dict, list):
                current_dict.append(val)
    return result


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from YAML file, using PyYAML if present or fallback."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    try:
        import yaml
        return yaml.safe_load(content)
    except ImportError:
        return _parse_yaml_fallback(content)
