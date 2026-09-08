# src/data/download.py
"""
Roboflow dataset download + basic integrity helpers
for the Leaf Lesion Severity pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from roboflow import Roboflow


def download_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int = 1,
    fmt: str = "yolov8",
    location: str | Path = "dataset",
    overwrite: bool = True,
) -> Path:
    """
    Download a Roboflow dataset version in the requested format.

    Returns
    -------
    Path
        Absolute path to the downloaded dataset root.
    """
    location = Path(location)
    location.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    version_obj = project_obj.version(version)

    dataset = version_obj.download(
        fmt,
        location=str(location),
        overwrite=overwrite,
    )
    return Path(dataset.location)


def load_data_yaml(dataset_root: str | Path) -> Dict[str, Any]:
    """Load and return the data.yaml dictionary from a YOLO export."""
    dataset_root = Path(dataset_root)
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")

    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def verify_class_mapping(
    names: list[str],
    leaf_keywords: list[str],
    lesion_keywords: list[str],
) -> Dict[str, str]:
    """
    Map every class name to 'leaf', 'lesion', or 'other'.

    Raises a clear warning message for unmatched classes.
    """
    mapping: Dict[str, str] = {}
    unmatched = []

    for name in names:
        n = str(name).lower()
        if any(k.lower() in n for k in leaf_keywords):
            mapping[name] = "leaf"
        elif any(k.lower() in n for k in lesion_keywords):
            mapping[name] = "lesion"
        else:
            mapping[name] = "other"
            unmatched.append(name)

    if unmatched:
        print(
            f"WARNING: these class names do not match leaf/lesion keywords "
            f"and will be ignored by severity scoring: {unmatched}"
        )
    else:
        print("All dataset classes map to 'leaf' or 'lesion'. Safe to proceed.")

    return mapping


# ------------------------------------------------------------------
# Convenience entry-point used by the notebook / CLI
# ------------------------------------------------------------------
def run_download_from_config(config: Dict[str, Any]) -> Path:
    """
    Helper that reads the nested config structure and downloads the dataset.

    Expected keys (matching config.yaml):
        config["roboflow"]["api_key"]
        config["roboflow"]["workspace"]
        ...
    """
    rf = config["roboflow"]
    paths = config.get("paths", {})

    dataset_root = download_dataset(
        api_key=rf["api_key"] or os.getenv("ROBOFLOW_API_KEY", ""),
        workspace=rf["workspace"],
        project=rf["project"],
        version=rf.get("version", 1),
        fmt=rf.get("format", "yolov8"),
        location=paths.get("dataset_dir", "dataset"),
        overwrite=True,
    )

    data_yaml = load_data_yaml(dataset_root)
    print("Dataset downloaded to:", dataset_root)
    print("Classes:", data_yaml.get("names"))

    # Optional class-mapping check
    classes = config.get("classes", {})
    verify_class_mapping(
        names=data_yaml.get("names", []),
        leaf_keywords=classes.get("leaf", ["0", "leaf"]),
        lesion_keywords=classes.get("lesion", ["1", "spot", "lesion"]),
    )

    return dataset_root


if __name__ == "__main__":
    # Minimal CLI example
    import argparse
    import yaml as _yaml

    parser = argparse.ArgumentParser(description="Download Roboflow dataset")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = _yaml.safe_load(f)

    run_download_from_config(cfg)
