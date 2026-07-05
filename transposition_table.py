import json
import ast

def load_cache (filename : str) :
    with open(filename) as f: raw = json.load(f)
    return {ast.literal_eval(k) : v for k, v in raw.items()}


def save_cache(cache : dict, filename : str) :
    serialize = {str(k) : v for k, v in cache.items()}

    with open(filename, 'w') as f :
        json.dump(serialize, f, indent=4, ensure_ascii=False)