"""Load immutable upstream files with isolated imports and verify provenance."""

import ast
import builtins
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

VENDOR = Path(__file__).resolve().parent / "vendor"


def verify_sources():
    manifest = json.loads((VENDOR / "manifest.json").read_text(encoding="utf-8"))
    count = 0
    for source in manifest:
        for item in source["files"]:
            path = VENDOR / source["name"] / item["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                raise RuntimeError("Upstream file changed: " + str(path))
            count += 1
    return {"verified_files": count, "revisions": {s["name"]: s["revision"] for s in manifest}}


@lru_cache(maxsize=None)
def upstream(project, filename):
    """Redirect only this module's absolute 'models.*' imports to its own repo."""
    path = VENDOR / project / filename
    name = "ablation._upstream_" + project + "_" + filename.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name.startswith("models."):
            if not fromlist:
                raise ImportError("Expected an explicit upstream from-import: " + name)
            return upstream(project, name.replace(".", "/") + ".py")
        return original_import(name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=local_import)
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def reve_position_source():
    """Compile the two original PE definitions verbatim, without REVE's trainer imports.

    The complete, unedited file remains in vendor. No AST nodes or function bodies
    are rewritten. This avoids requiring transformers/flash-attn/PyTorch 2.2 just
    to call the two PE definitions in the project's PyTorch 2.0.1 environment.
    """
    import math
    import torch
    from torch import nn

    path = VENDOR / "reve/src/models/encoder.py"
    source = path.read_text(encoding="utf-8")
    definitions = {node.name: node for node in ast.parse(source).body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    module = ModuleType("ablation._upstream_reve_position")
    module.__dict__.update(torch=torch, nn=nn, math=math)
    sys.modules[module.__name__] = module
    for name in ("FourierEmb4D", "mlp_pos_embedding"):
        node = definitions[name]
        text = "\n" * (node.lineno - 1) + ast.get_source_segment(source, node)
        exec(compile(text, str(path), "exec"), module.__dict__)
    return module


if __name__ == "__main__":
    print(json.dumps(verify_sources(), indent=2))
