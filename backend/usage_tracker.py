import json
import os
from datetime import datetime

USAGE_FILE = "api_usage_log.json"

def log_api_call(model_id: str, status: str = "success"):
    """Logs a Roboflow API call to track quota usage."""
    data = {"total_calls": 0, "history": []}
    
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            pass
            
    data["total_calls"] += 1
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_id,
        "status": status
    }
    # Keep last 100 entries in history to avoid file bloating
    data["history"] = ([entry] + data["history"])[:100]
    
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f, indent=4)

def get_usage_stats():
    """Returns total calls made in current session."""
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, "r") as f:
            return json.load(f).get("total_calls", 0)
    return 0
