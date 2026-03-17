"""SQLite-backed design database for AstraTurbo.

Stores turbomachinery design parameters, results, and metadata
in a local SQLite database with search, comparison, and export
capabilities.

Default location: ~/.astraturbo/designs.db

Usage::

    db = DesignDatabase()

    design_id = db.save_design(
        name="Compressor Stage 1",
        parameters={"pressure_ratio": 1.5, "cl0": 1.0},
        results={"efficiency": 0.88, "mass_flow": 10.5},
        metadata={"author": "engineer", "project": "TurboX"},
    )

    design = db.load_design(design_id)
    matches = db.search("compressor pressure_ratio>1.3")
    comparison = db.compare(id1, id2)
    db.export_csv("designs_export.csv")
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any


# Default database path
DEFAULT_DB_DIR = Path.home() / ".astraturbo"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "designs.db"

# SQL schema
_SCHEMA = """
CREATE TABLE IF NOT EXISTS designs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    parameters TEXT NOT NULL,
    results TEXT NOT NULL,
    tags TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_designs_name ON designs(name);
CREATE INDEX IF NOT EXISTS idx_designs_created ON designs(created_at);
CREATE INDEX IF NOT EXISTS idx_designs_tags ON designs(tags);
"""


class DesignDatabase:
    """SQLite-backed database for turbomachinery designs.

    Provides CRUD operations plus search, comparison, and export
    for design parameters and results.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize the database connection.

        Creates the database file and schema if they don't exist.

        Args:
            db_path: Path to the SQLite database file. If None, uses
                the default path at ~/.astraturbo/designs.db.
        """
        if db_path is None:
            db_path = DEFAULT_DB_PATH

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # Create schema
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> DesignDatabase:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def save_design(
        self,
        name: str,
        parameters: dict[str, Any],
        results: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str = "",
    ) -> int:
        """Save a design to the database.

        Args:
            name: Human-readable design name.
            parameters: Dictionary of design parameters (serialized as JSON).
            results: Dictionary of design results/performance metrics.
            metadata: Additional metadata (author, date, project, etc.).
            tags: List of string tags for categorization.
            notes: Free-text notes.

        Returns:
            Integer design ID.
        """
        now = time.time()
        params_json = json.dumps(parameters, default=_json_serializer)
        results_json = json.dumps(results or {}, default=_json_serializer)
        meta_json = json.dumps(metadata or {}, default=_json_serializer)
        tags_str = ",".join(tags) if tags else ""

        cursor = self._conn.execute(
            """INSERT INTO designs (name, created_at, updated_at,
               parameters, results, tags, notes, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, now, now, params_json, results_json, tags_str, notes, meta_json),
        )
        self._conn.commit()
        return cursor.lastrowid

    def load_design(self, design_id: int) -> dict[str, Any]:
        """Load a design by its ID.

        Args:
            design_id: Integer design ID.

        Returns:
            Dictionary with all design fields.

        Raises:
            KeyError: If design not found.
        """
        row = self._conn.execute(
            "SELECT * FROM designs WHERE id = ?", (design_id,)
        ).fetchone()

        if row is None:
            raise KeyError(f"Design {design_id} not found")

        return self._row_to_dict(row)

    def update_design(
        self,
        design_id: int,
        name: str | None = None,
        parameters: dict[str, Any] | None = None,
        results: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> None:
        """Update an existing design.

        Only non-None arguments are updated.

        Args:
            design_id: Design to update.
            name: New name (or None to keep).
            parameters: New parameters (or None to keep).
            results: New results (or None to keep).
            metadata: New metadata (or None to keep).
            tags: New tags (or None to keep).
            notes: New notes (or None to keep).
        """
        updates = []
        values = []

        if name is not None:
            updates.append("name = ?")
            values.append(name)
        if parameters is not None:
            updates.append("parameters = ?")
            values.append(json.dumps(parameters, default=_json_serializer))
        if results is not None:
            updates.append("results = ?")
            values.append(json.dumps(results, default=_json_serializer))
        if metadata is not None:
            updates.append("metadata = ?")
            values.append(json.dumps(metadata, default=_json_serializer))
        if tags is not None:
            updates.append("tags = ?")
            values.append(",".join(tags))
        if notes is not None:
            updates.append("notes = ?")
            values.append(notes)

        if not updates:
            return

        updates.append("updated_at = ?")
        values.append(time.time())
        values.append(design_id)

        set_clause = ", ".join(updates)
        self._conn.execute(
            f"UPDATE designs SET {set_clause} WHERE id = ?",
            values,
        )
        self._conn.commit()

    def delete_design(self, design_id: int) -> bool:
        """Delete a design by ID.

        Args:
            design_id: Design to delete.

        Returns:
            True if a design was deleted.
        """
        cursor = self._conn.execute(
            "DELETE FROM designs WHERE id = ?", (design_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def search(
        self,
        query: str = "",
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search designs by name, tags, or parameter values.

        The query string supports:
          - Plain text: searches name and notes
          - "param:key>value": filter by parameter value
          - "result:key>value": filter by result value

        Args:
            query: Search query string.
            tags: Filter by tags (all must match).
            limit: Maximum number of results.

        Returns:
            List of matching design dictionaries.
        """
        # Start with all designs
        sql = "SELECT * FROM designs WHERE 1=1"
        params: list[Any] = []

        # Text search in name and notes
        if query:
            # Check for structured queries
            parts = query.split()
            text_parts = []

            for part in parts:
                if ":" in part and (">" in part or "<" in part or "=" in part):
                    # Structured query: handled in Python post-filter
                    continue
                else:
                    text_parts.append(part)

            if text_parts:
                text = " ".join(text_parts)
                sql += " AND (name LIKE ? OR notes LIKE ?)"
                params.extend([f"%{text}%", f"%{text}%"])

        # Tag filter
        if tags:
            for tag in tags:
                sql += " AND tags LIKE ?"
                params.append(f"%{tag}%")

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        results = [self._row_to_dict(row) for row in rows]

        # Apply structured query filters in Python (for JSON field queries)
        if query:
            for part in query.split():
                if ":" not in part:
                    continue

                prefix, condition = part.split(":", 1)
                results = self._apply_filter(results, prefix, condition)

        return results

    def _apply_filter(
        self,
        designs: list[dict],
        prefix: str,
        condition: str,
    ) -> list[dict]:
        """Apply a structured filter like 'param:pressure_ratio>1.3'."""
        # Parse operator
        for op in [">=", "<=", "!=", ">", "<", "="]:
            if op in condition:
                key, val_str = condition.split(op, 1)
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
                break
        else:
            return designs

        filtered = []
        for d in designs:
            if prefix == "param":
                field_dict = d.get("parameters", {})
            elif prefix == "result":
                field_dict = d.get("results", {})
            else:
                continue

            actual = field_dict.get(key)
            if actual is None:
                continue

            try:
                actual_num = float(actual)
                val_num = float(val)
                if op == ">" and actual_num > val_num:
                    filtered.append(d)
                elif op == ">=" and actual_num >= val_num:
                    filtered.append(d)
                elif op == "<" and actual_num < val_num:
                    filtered.append(d)
                elif op == "<=" and actual_num <= val_num:
                    filtered.append(d)
                elif op == "=" and abs(actual_num - val_num) < 1e-10:
                    filtered.append(d)
                elif op == "!=" and abs(actual_num - val_num) > 1e-10:
                    filtered.append(d)
            except (ValueError, TypeError):
                if op == "=" and str(actual) == str(val):
                    filtered.append(d)
                elif op == "!=" and str(actual) != str(val):
                    filtered.append(d)

        return filtered

    def compare(
        self,
        id1: int,
        id2: int,
    ) -> dict[str, Any]:
        """Compare two designs side-by-side.

        Returns a dictionary highlighting differences in parameters
        and results between the two designs.

        Args:
            id1: First design ID.
            id2: Second design ID.

        Returns:
            Dictionary with comparison data:
                'design_1': full design dict
                'design_2': full design dict
                'parameter_differences': list of differing parameters
                'result_differences': list of differing results
                'summary': text summary
        """
        d1 = self.load_design(id1)
        d2 = self.load_design(id2)

        param_diffs = []
        all_param_keys = set(d1["parameters"].keys()) | set(d2["parameters"].keys())
        for key in sorted(all_param_keys):
            v1 = d1["parameters"].get(key)
            v2 = d2["parameters"].get(key)
            if v1 != v2:
                param_diffs.append({
                    "parameter": key,
                    "design_1": v1,
                    "design_2": v2,
                    "difference": self._compute_diff(v1, v2),
                })

        result_diffs = []
        all_result_keys = set(d1["results"].keys()) | set(d2["results"].keys())
        for key in sorted(all_result_keys):
            v1 = d1["results"].get(key)
            v2 = d2["results"].get(key)
            if v1 != v2:
                result_diffs.append({
                    "result": key,
                    "design_1": v1,
                    "design_2": v2,
                    "difference": self._compute_diff(v1, v2),
                })

        summary = (
            f"Comparing '{d1['name']}' (ID={id1}) vs '{d2['name']}' (ID={id2}): "
            f"{len(param_diffs)} parameter differences, "
            f"{len(result_diffs)} result differences."
        )

        return {
            "design_1": d1,
            "design_2": d2,
            "parameter_differences": param_diffs,
            "result_differences": result_diffs,
            "summary": summary,
        }

    def _compute_diff(self, v1: Any, v2: Any) -> Any:
        """Compute difference between two values."""
        if v1 is None or v2 is None:
            return None
        try:
            return float(v2) - float(v1)
        except (ValueError, TypeError):
            return None

    def list_designs(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        order_by: str = "created_at",
        ascending: bool = False,
    ) -> list[dict[str, Any]]:
        """List designs with optional filters.

        Args:
            filters: Dictionary of field -> value filters.
                Supports: 'name', 'tags', 'after' (timestamp),
                'before' (timestamp).
            limit: Maximum results.
            order_by: Sort field ('created_at', 'name', 'updated_at').
            ascending: Sort direction.

        Returns:
            List of design summary dictionaries.
        """
        sql = "SELECT id, name, created_at, updated_at, tags, notes FROM designs WHERE 1=1"
        params: list[Any] = []

        if filters:
            if "name" in filters:
                sql += " AND name LIKE ?"
                params.append(f"%{filters['name']}%")
            if "tags" in filters:
                for tag in (filters["tags"] if isinstance(filters["tags"], list) else [filters["tags"]]):
                    sql += " AND tags LIKE ?"
                    params.append(f"%{tag}%")
            if "after" in filters:
                sql += " AND created_at > ?"
                params.append(filters["after"])
            if "before" in filters:
                sql += " AND created_at < ?"
                params.append(filters["before"])

        allowed_order = {"created_at", "name", "updated_at", "id"}
        if order_by not in allowed_order:
            order_by = "created_at"

        direction = "ASC" if ascending else "DESC"
        sql += f" ORDER BY {order_by} {direction} LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "name": row["name"],
                "created_at": datetime.fromtimestamp(row["created_at"]).isoformat(),
                "updated_at": datetime.fromtimestamp(row["updated_at"]).isoformat(),
                "tags": row["tags"].split(",") if row["tags"] else [],
                "notes": row["notes"],
            })

        return results

    def export_csv(
        self,
        filepath: str | Path,
        design_ids: list[int] | None = None,
    ) -> int:
        """Export designs to a CSV file.

        Flattens the JSON parameter and result fields into columns.

        Args:
            filepath: Output CSV file path.
            design_ids: Specific IDs to export (None = all).

        Returns:
            Number of designs exported.
        """
        if design_ids:
            placeholders = ",".join("?" * len(design_ids))
            rows = self._conn.execute(
                f"SELECT * FROM designs WHERE id IN ({placeholders})",
                design_ids,
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM designs ORDER BY id"
            ).fetchall()

        if not rows:
            return 0

        # Collect all parameter and result keys
        param_keys: set[str] = set()
        result_keys: set[str] = set()
        designs = []

        for row in rows:
            d = self._row_to_dict(row)
            designs.append(d)
            param_keys.update(d["parameters"].keys())
            result_keys.update(d["results"].keys())

        param_keys_sorted = sorted(param_keys)
        result_keys_sorted = sorted(result_keys)

        # Write CSV
        header = (
            ["id", "name", "created_at", "tags", "notes"]
            + [f"param_{k}" for k in param_keys_sorted]
            + [f"result_{k}" for k in result_keys_sorted]
        )

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for d in designs:
                row_data = [
                    d["id"],
                    d["name"],
                    d["created_at"],
                    ",".join(d["tags"]),
                    d["notes"],
                ]
                for k in param_keys_sorted:
                    row_data.append(d["parameters"].get(k, ""))
                for k in result_keys_sorted:
                    row_data.append(d["results"].get(k, ""))
                writer.writerow(row_data)

        return len(designs)

    def count(self) -> int:
        """Return total number of designs in the database."""
        row = self._conn.execute("SELECT COUNT(*) FROM designs").fetchone()
        return row[0] if row else 0

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a database row to a design dictionary."""
        return {
            "id": row["id"],
            "name": row["name"],
            "created_at": datetime.fromtimestamp(row["created_at"]).isoformat(),
            "updated_at": datetime.fromtimestamp(row["updated_at"]).isoformat(),
            "parameters": json.loads(row["parameters"]),
            "results": json.loads(row["results"]),
            "tags": row["tags"].split(",") if row["tags"] else [],
            "notes": row["notes"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
        }


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for numpy types and other non-serializable objects."""
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass

    if isinstance(obj, datetime):
        return obj.isoformat()

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
