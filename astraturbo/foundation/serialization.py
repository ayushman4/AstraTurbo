"""Serialization for AstraTurbo projects.

Supports YAML-based project save/load, plus backward-compatible import of
legacy XML project files.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from .undo import stack


def serialize_instance(obj: Any) -> dict:
    """Convert an ATObject to a dictionary for serialization.

    Stores the class name, module path, and all property values.
    """
    d = {
        "__class__": obj.__class__.__name__,
        "__module__": obj.__class__.__module__,
    }
    if hasattr(obj, "properties"):
        d.update({p.name: p.get(obj) for p in obj.properties})
    return d


# Whitelist of allowed modules for deserialization — prevents arbitrary code execution
_ALLOWED_MODULES = {
    "astraturbo.camberline",
    "astraturbo.thickness",
    "astraturbo.profile",
    "astraturbo.blade",
    "astraturbo.machine",
    "astraturbo.distribution",
    "astraturbo.mesh",
}


def unserialize_object(d: dict) -> Any:
    """Reconstruct an ATObject from a serialized dictionary.

    Only allows classes from whitelisted astraturbo modules to prevent
    arbitrary code execution from malicious project files.
    """
    class_name = d.pop("__class__", None)
    module_name = d.pop("__module__", None)
    if class_name is not None and module_name is not None:
        # Security: only allow astraturbo modules
        if not any(module_name.startswith(m) for m in _ALLOWED_MODULES):
            raise ValueError(
                f"Untrusted module in project file: {module_name}.{class_name}. "
                f"Only astraturbo modules are allowed."
            )
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        obj = cls.default() if hasattr(cls, "default") else cls()
        for key, value in d.items():
            setattr(obj, key, value)
        return obj
    return d


def save(obj: Any, filepath: str | Path) -> None:
    """Save an object to a YAML project file."""
    filepath = Path(filepath)
    data = serialize_instance(obj)
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load(filepath: str | Path) -> Any:
    """Load an object from a YAML project file."""
    filepath = Path(filepath)
    with open(filepath) as f:
        data = yaml.safe_load(f)
    obj = unserialize_object(data)
    stack().clear()
    return obj


def import_bladedesigner_xml(filepath: str | Path) -> dict:
    """Import a legacy XML project file into a dictionary.

    This provides backward compatibility with legacy .xml projects.
    The returned dict can be used to construct AstraTurbo objects.
    """
    import xml.etree.ElementTree as ET
    from xml.etree.ElementTree import XMLParser

    filepath = Path(filepath)

    # Security: use defused parsing — disable DTD and external entities
    parser = XMLParser()
    tree = ET.parse(filepath, parser=parser)
    root = tree.getroot()

    def _element_to_dict(element: ET.Element) -> dict:
        result: dict[str, Any] = {}
        for child in element:
            if len(child) > 0:
                result[child.tag] = _element_to_dict(child)
            else:
                text = child.text
                if text is not None:
                    # Try numeric conversion
                    try:
                        result[child.tag] = float(text)
                        if result[child.tag] == int(result[child.tag]):
                            result[child.tag] = int(result[child.tag])
                    except ValueError:
                        result[child.tag] = text
                else:
                    result[child.tag] = None
        return result

    return _element_to_dict(root)
