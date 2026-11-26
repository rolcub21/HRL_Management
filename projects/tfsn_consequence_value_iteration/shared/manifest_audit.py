"""Fail when the active continuous reference bypasses its method manifest."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from .settings import MANIFEST_PATH, METHOD_MANIFEST


REPOSITORY = Path(__file__).resolve().parents[3]
POINTMAZE_REFERENCE_MODULES = (
    "projects/tfsn_consequence_value_iteration/pointmaze/roadmap.py",
    "projects/tfsn_consequence_value_iteration/pointmaze/planner.py",
    "projects/tfsn_consequence_value_iteration/pointmaze/prepare_bridges.py",
    "projects/tfsn_consequence_value_iteration/pointmaze/passages.py",
    "projects/tfsn_consequence_value_iteration/pointmaze/fields.py",
)
FORBIDDEN_POINTMAZE_POLICY_TEXT = (
    "MOVEMENT_REQUEST_SIGMAS",
    "movement_request_sigmas",
    "acquisition_detection_radius",
    "STALL_RADIUS_M",
)
POINTMAZE_FREEZE = REPOSITORY / (
    "projects/tfsn_consequence_value_iteration/pointmaze/R11_FREEZE.json"
)
MANIFEST_CATEGORIES = {
    "global_computational",
    "data_derived",
    "global_decision_principles",
    "task_knowledge",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_leaves(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_leaves(child, prefix + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from numeric_leaves(child, prefix + (str(index),))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def result_failure_set(path):
    report = json.loads(path.read_text())
    return {
        row["goal_name"]
        for row in report["task_metrics"]
        if float(row["acquisition_at_horizon"]) == 0.0
    }


def audit_freeze(failures):
    freeze = json.loads(POINTMAZE_FREEZE.read_text())
    if freeze.get("method_revision") != METHOD_MANIFEST.get("method_revision"):
        failures.append("R11 freeze revision differs from shared manifest")
    if freeze.get("shared_method_manifest_sha256") != sha256(MANIFEST_PATH):
        failures.append("R11 freeze does not bind the current shared manifest")

    radius = freeze.get("detection_radius_decision", {})
    if radius.get("planner_reads_benchmark_detection_radius") is not False:
        failures.append("R11 freeze does not forbid planner detection radius")
    if radius.get("acquisition_search_scale") != "data_derived_landing_scale":
        failures.append("R11 freeze does not use landing scale for acquisition")

    expected = set(freeze.get("known_failures", {}).get("goal_names", []))
    for evidence in freeze.get("development_evidence", []):
        path = REPOSITORY / evidence["path"]
        if sha256(path) != evidence["sha256"]:
            failures.append(f"frozen evidence changed: {evidence['revision']}")
            continue
        if result_failure_set(path) != expected:
            failures.append(
                f"frozen failure set changed in {evidence['revision']}"
            )
    return freeze


def module_numeric_constants(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    rows = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(
            value.value, (int, float)
        ):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                rows.append((target.id, value.value, node.lineno))
    return rows


def audit_pointmaze():
    failures = []
    categories = METHOD_MANIFEST.get("categories", {})
    if set(categories) != MANIFEST_CATEGORIES:
        failures.append("shared manifest category inventory changed")
    for path, value in numeric_leaves(METHOD_MANIFEST):
        if path == ("schema_version",):
            continue
        if path[:2] != ("categories", "global_computational"):
            failures.append(
                f"numeric setting outside global computational manifest: "
                f"{'.'.join(path)}={value}"
            )
    task_knowledge = categories.get("task_knowledge", {})
    if task_knowledge.get("detection_radius") != (
        "not_a_planner_input; benchmark_success_radius_is_evaluator_only"
    ):
        failures.append("shared manifest detection-radius decision changed")
    for relative in POINTMAZE_REFERENCE_MODULES:
        path = REPOSITORY / relative
        text = path.read_text()
        for forbidden in FORBIDDEN_POINTMAZE_POLICY_TEXT:
            if forbidden in text:
                failures.append(f"{relative}: forbidden policy input {forbidden}")
        for name, value, line in module_numeric_constants(path):
            failures.append(
                f"{relative}:{line}: undeclared module constant {name}={value}"
            )
    protocol_path = REPOSITORY / (
        "projects/tfsn_consequence_value_iteration/pointmaze/protocol.json"
    )
    protocol = json.loads(protocol_path.read_text())
    if "global_settings" in protocol:
        failures.append("PointMaze protocol duplicates shared global settings")
    if protocol.get("shared_method_manifest") != str(
        MANIFEST_PATH.relative_to(REPOSITORY)
    ):
        failures.append("PointMaze protocol does not name the shared manifest")
    if protocol.get("method_revision") != METHOD_MANIFEST.get("method_revision"):
        failures.append("PointMaze method revision differs from shared manifest")
    freeze = audit_freeze(failures)
    if failures:
        raise ValueError("\n".join(failures))
    return {
        "schema_version": 1,
        "method_revision": METHOD_MANIFEST["method_revision"],
        "domain": "pointmaze",
        "manifest": str(MANIFEST_PATH),
        "active_modules": list(POINTMAZE_REFERENCE_MODULES),
        "setting_categories": sorted(MANIFEST_CATEGORIES),
        "numeric_settings_outside_global_manifest": [],
        "detection_radius_decision": freeze["detection_radius_decision"],
        "known_failures": freeze["known_failures"]["goal_names"],
        "freeze_record": str(POINTMAZE_FREEZE),
        "undeclared_policy_constants": [],
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("pointmaze",), required=True)
    args = parser.parse_args()
    report = audit_pointmaze()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
