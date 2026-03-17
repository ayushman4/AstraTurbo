"""Project management for AstraTurbo.

Handles saving/loading projects and importing legacy XML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def save_project(data: dict, filepath: str | Path) -> None:
    """Save project data to a YAML file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_project(filepath: str | Path) -> dict:
    """Load project data from a YAML file."""
    filepath = Path(filepath)
    with open(filepath) as f:
        return yaml.safe_load(f)


def import_bladedesigner_xml(filepath: str | Path) -> dict:
    """Import a legacy XML project file.

    Returns a dictionary representation of the XML tree that can
    be used to reconstruct AstraTurbo objects.
    """
    import xml.etree.ElementTree as ET

    filepath = Path(filepath)
    tree = ET.parse(filepath)
    root = tree.getroot()

    def _parse(element: ET.Element) -> dict:
        result: dict[str, Any] = dict(element.attrib)
        for child in element:
            child_data = _parse(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        if element.text and element.text.strip():
            text = element.text.strip()
            try:
                return float(text)
            except ValueError:
                return text
        return result

    return {root.tag: _parse(root)}
