import json
import os

# originally mc_constants.1.16.json
path = os.path.join(os.path.dirname(__file__), "assets/constants.json")
all_data = json.load(open(path))
ALL_ITEMS = [item["type"] for item in all_data["items"]]
