"""Execute fresh kernels, preserve successful defaults, and package review copies."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def main() -> None:
    """Run numbered examples using this environment and a temporary kernel spec."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument("--parameters", default="{}", help="JSON object of visible parameter overrides.")
    parser.add_argument("--output-dir", type=Path, help="Required separate destination for overrides.")
    parser.add_argument("--timeout", type=int, default=300, help="Seconds allowed per cell, including JIT warm-up.")
    args = parser.parse_args()
    overrides = json.loads(args.parameters)
    if not isinstance(overrides, dict):
        parser.error("--parameters must be a JSON object")
    directory = Path(__file__).resolve().parent
    paths = [p.resolve() for p in args.notebooks] or sorted(directory.glob("[0-9][0-9]_*.ipynb"))
    output_dir = args.output_dir.resolve() if args.output_dir else None
    if overrides and output_dir is None:
        parser.error("Use --output-dir with overrides to preserve the saved default.")
    if output_dir is not None:
        protected = {p.resolve() for p in directory.glob("*.ipynb")} | set(paths)
        if output_dir == directory or any(output_dir / p.name in protected for p in paths):
            parser.error("--output-dir must be separate from source notebooks and saved defaults.")
    os.environ["PLOOMBER_STATS_ENABLED"] = "false"
    with TemporaryDirectory(prefix="vgi-anomaly-kernel-") as temporary:
        kernel_dir = Path(temporary) / "anomaly-examples"
        kernel_dir.mkdir()
        (kernel_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Python (vgi-anomaly notebooks)",
                    "language": "python",
                }
            )
        )
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            parameter_cell = next(c for c in notebook.cells if "parameters" in c.metadata.get("tags", []))
            names = {
                node.targets[0].id
                for node in ast.parse(parameter_cell.source).body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            }
            if unknown := overrides.keys() - names:
                raise ValueError(f"Unknown parameters: {sorted(unknown)}")
            for key, value in overrides.items():
                parameter_cell.source = re.sub(
                    rf"^{re.escape(key)}\s*=.*$",
                    lambda _, name=key, replacement=value: f"{name} = {replacement!r}",
                    parameter_cell.source,
                    flags=re.MULTILINE,
                )
            manager = KernelManager(
                kernel_name="anomaly-examples",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]),
            )
            start = monotonic()
            print(f"Running {path.name} ...", flush=True)
            client = NotebookClient(
                notebook,
                km=manager,
                timeout=args.timeout,
                allow_errors=False,
                resources={"metadata": {"path": str(directory)}},
            )
            try:
                client.execute(cleanup_kc=True)
            except Exception:
                print(f"FAILED: {path.name}; saved notebook left unchanged.", flush=True)
                raise
            nbformat.validate(notebook)
            destination = output_dir / path.name if output_dir else path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if output_dir:
                shutil.copytree(directory / "assets", output_dir / "assets", dirs_exist_ok=True)
                for name in ("README.md", "STANDARD.md", "notebook_support.py", "pyproject.toml", "uv.lock"):
                    shutil.copy2(directory / name, output_dir / name)
            temporary_output = destination.with_suffix(".ipynb.tmp")
            try:
                nbformat.write(notebook, temporary_output)
                temporary_output.replace(destination)
            finally:
                temporary_output.unlink(missing_ok=True)
            code = [c for c in notebook.cells if c.cell_type == "code"]
            charts = sum("image/png" in o.get("data", {}) for c in code for o in c.outputs)
            print(f"OK: {len(code)} code cells, {charts} charts, {monotonic() - start:.1f}s; {destination}", flush=True)


if __name__ == "__main__":
    main()
