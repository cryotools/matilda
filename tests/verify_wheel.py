"""Verify that a MATILDA wheel contains the maintained package sources."""

from __future__ import annotations

import argparse
import ast
from email.parser import Parser
from pathlib import Path
import zipfile


PACKAGE_FILES = {
    "matilda/__init__.py",
    "matilda/core.py",
    "matilda/mspot_glacier.py",
    "matilda/parameters.json",
}


def source_version(project_root: Path) -> str:
    tree = ast.parse((project_root / "matilda/__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    return ast.literal_eval(node.value)
    raise AssertionError("matilda.__version__ is not assigned")


def verify_wheel(wheel_path: Path, project_root: Path) -> None:
    with zipfile.ZipFile(wheel_path) as archive:
        members = set(archive.namelist())
        package_members = {
            member for member in members if member.startswith("matilda/")
        }
        assert package_members == PACKAGE_FILES, (
            f"unexpected package contents: {sorted(package_members)}"
        )
        assert not any(".DS_Store" in member for member in members)
        assert not any(
            member.startswith(("build/", "dist/", "docs/_build/"))
            for member in members
        )

        for member in PACKAGE_FILES:
            assert archive.read(member) == (project_root / member).read_bytes(), (
                f"wheel member does not match maintained source: {member}"
            )

        metadata_member = next(
            member for member in members if member.endswith(".dist-info/METADATA")
        )
        wheel_metadata = Parser().parsestr(
            archive.read(metadata_member).decode("utf-8")
        )
        assert wheel_metadata["Version"] == source_version(project_root)
        assert wheel_metadata["Requires-Python"] == ">=3.11"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).parents[1],
    )
    arguments = parser.parse_args()
    verify_wheel(arguments.wheel.resolve(), arguments.project_root.resolve())
    print(f"Verified {arguments.wheel.name}")


if __name__ == "__main__":
    main()
