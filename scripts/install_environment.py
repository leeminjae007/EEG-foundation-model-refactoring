"""기존 환경을 읽기만 하여 GR9-1의 실제 torch binary를 독립 venv에 복사한다."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    shutil.copymode(source, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-env", type=Path, required=True)
    parser.add_argument("--libraries-only", action="store_true")
    args = parser.parse_args()
    source = args.source_env.resolve()
    environment = ROOT / ".venv"
    python = environment / "bin/python"
    if not python.exists():
        subprocess.run([str(source / "bin/python"), "-m", "venv", "--copies", str(environment)], check=True)
    site = Path("lib/python3.8/site-packages")
    packages = ("torch", "torch-2.0.1-py3.8.egg-info", "torchgen")
    if args.libraries_only:
        packages = ()
    for name in packages:
        print("Copying", name, flush=True)
        for path in (source / site / name).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                destination = environment / path.relative_to(source)
                copy_file(path, destination)
    libraries = set()
    # 실제 eegfm의 torch는 공용 설치를 가리키는 symlink다. 대상의 native lib를 복사한다.
    native_root = (source / site / "torch").resolve().parents[2]
    for library in (source / site / "torch/lib").glob("*.so*"):
        result = subprocess.run(["ldd", str(library)], capture_output=True, text=True, check=True)
        for value in re.findall(r"=> (/\S+)", result.stdout):
            path = Path(value)
            resolved = path.resolve()
            if str(resolved).startswith(str(native_root) + "/") and "/site-packages/" not in str(resolved):
                libraries.add(path)
    # MKL과 CUDA JIT가 실행 중 여는 라이브러리도 같은 환경에서 복사한다.
    for pattern in ("libmkl*.so*", "libnvrtc*.so*"):
        libraries.update(native_root.glob(pattern))
    for path in sorted(libraries):
        print("Copying", path.name, flush=True)
        copy_file(path, environment / "lib" / path.name)
    report = {"source_env": str(source), "packages": ["torch", "torchgen"],
              "libraries": sorted(str(path) for path in libraries)}
    (ROOT / "docs/torch_runtime_copy.json").write_text(json.dumps(report, indent=2))
    os.environ["PIP_CACHE_DIR"] = str(ROOT / "cache/pip")
    os.environ["TMPDIR"] = str(ROOT / "tmp")
    os.environ["PYTHONNOUSERSITE"] = "1"
    subprocess.run([str(python), "-m", "pip", "install", "--no-cache-dir",
                    "-r", str(ROOT / "requirements.txt")], check=True)
    result = subprocess.run([str(python), "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (ROOT / "requirements-lock.txt").write_text(result.stdout)


if __name__ == "__main__":
    main()
