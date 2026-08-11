"""Version-2 lifecycle for the capacity-aware rolling-GA comparison.

Version 1 correctly executed the four repaired assignment sources, but its
post-serialization validator accidentally treated JSON object insertion order
as part of the lexicographic objective.  ``sort_keys=True`` necessarily
changes that order on disk.  This version leaves the policy, source IDs,
source versions, objectives, instances, and execution grid unchanged.  It
changes only artifact validation:

* ``best_cost_components`` must have the exact objective-specific key set;
* ``objective_lexicographic_fields`` is the sole, separately authenticated
  ordering contract used to reconstruct the scalar objective; and
* JSON mapping order has no semantics.

The V1 runner is loaded as an isolated implementation module so its file and
completed provisional artifacts remain byte-for-byte reproducible.  Both the
V1 implementation and this lifecycle wrapper are bound into the V2 contract.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence


PROTOCOL = "vcg_v2_3_capacity_aware_partial_ga_repair_85xxx_development_v2"
SCHEMA_VERSION = 2
DEFAULT_OUTPUT_DIRECTORY = "vcg-v2-3-capacity-aware-ga-repair-v2-85k"

_HERE = Path(__file__).resolve().parent
_V1_PATH = _HERE / "compare_vcg_v2_3_capacity_aware_ga_repair.py"
_IMPL_MODULE_NAME = "_vcg_v2_3_capacity_aware_ga_repair_v2_impl"
_SPEC = importlib.util.spec_from_file_location(_IMPL_MODULE_NAME, _V1_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import failure
    raise ImportError(f"cannot load immutable V1 implementation: {_V1_PATH}")
_impl = importlib.util.module_from_spec(_SPEC)
sys.modules[_IMPL_MODULE_NAME] = _impl
_SPEC.loader.exec_module(_impl)

_ORIGINAL_BUILD_PARSER = _impl._build_parser
_ORIGINAL_BUILD_REPAIR_CONTRACT = _impl.build_repair_contract
_ORIGINAL_CAPACITY_AUDIT_ISSUES = _impl._capacity_audit_issues
_ORIGINAL_CURRENT_SOURCE_PATHS = _impl._current_source_paths
_ORIGINAL_LOAD_OR_EXECUTE_REPAIRED = _impl._load_or_execute_repaired


def _expected_lexicographic_fields(method: str) -> tuple[str, ...]:
    if method == _impl.REPAIRED_2009_METHOD:
        return tuple(_impl.LEGACY_LEX_FIELDS)
    if method in _impl.REPAIRED_METHODS:
        return tuple(_impl.DURATION_LEX_FIELDS)
    raise _impl.CapacityRepairComparisonError(
        f"unknown repaired method for cost audit: {method}"
    )


def _capacity_audit_issues(
    raw: Mapping,
    *,
    method: str,
    seed: int,
    expected_pickup_cell: Sequence[int] = _impl.FROZEN_PICKUP_CELL,
    expected_delivery_count: int = _impl.EXPECTED_DELIVERIES,
) -> list[str]:
    """Authenticate cost semantics without assigning meaning to map order.

    The immutable V1 validator remains the complete source/reservation audit.
    A validation-only copy is canonicalized into the separately recorded
    lexicographic order after (and only after) the original serialized mapping
    is proven to contain the exact objective-specific key set.
    """

    expected_lex = _expected_lexicographic_fields(method)
    copied = copy.deepcopy(raw)
    source = copied.get("selector_audit", {}).get("source_audit", {})
    decisions = source.get("decisions", ()) if isinstance(source, Mapping) else ()
    key_issues: list[str] = []
    if isinstance(decisions, (tuple, list)):
        for index, record in enumerate(decisions):
            if not isinstance(record, Mapping):
                continue
            components = record.get("best_cost_components")
            if not isinstance(components, Mapping):
                continue
            if set(components) != set(expected_lex):
                key_issues.append(f"decision[{index}]:cost_component_key_set")
                continue
            # Mapping order is deliberately discarded.  The separately
            # authenticated objective_lexicographic_fields controls scalar
            # reconstruction in the immutable validator.
            record["best_cost_components"] = {
                field: components[field] for field in expected_lex
            }
    return [
        *key_issues,
        *_ORIGINAL_CAPACITY_AUDIT_ISSUES(
            copied,
            method=method,
            seed=seed,
            expected_pickup_cell=expected_pickup_cell,
            expected_delivery_count=expected_delivery_count,
        ),
    ]


def _current_source_paths() -> dict[str, Path]:
    paths = dict(_ORIGINAL_CURRENT_SOURCE_PATHS())
    paths["compare_vcg_v2_3_capacity_aware_ga_repair_v2.py"] = Path(
        __file__
    ).resolve()
    return dict(sorted(paths.items()))


def _load_or_execute_repaired(
    *,
    output_dir: Path,
    method: str,
    seed: int,
    input_contract: Mapping,
    identities: Mapping[int, Mapping],
    execute: bool,
    executor,
):
    """Make the serialized ledger authoritative for every new execution.

    The immutable implementation performs all atomic-write and full-ledger
    validation.  V2 adds an immediate execute=False pass after any permitted
    execution call, then compares and returns the deserialized row.  Thus no
    in-memory-only row can enter aggregation even when its JSON representation
    differs structurally (for example, tuple/list or mapping order).
    """

    observed = _ORIGINAL_LOAD_OR_EXECUTE_REPAIRED(
        output_dir=output_dir,
        method=method,
        seed=seed,
        input_contract=input_contract,
        identities=identities,
        execute=execute,
        executor=executor,
    )
    if observed is None or not execute:
        return observed
    reloaded = _ORIGINAL_LOAD_OR_EXECUTE_REPAIRED(
        output_dir=output_dir,
        method=method,
        seed=seed,
        input_contract=input_contract,
        identities=identities,
        execute=False,
        executor=lambda: (_ for _ in ()).throw(
            AssertionError("serialized V2 reload attempted policy execution")
        ),
    )
    _impl._require_equal(
        f"V2 serialized row round-trip {method}:{seed}",
        _impl._digest_json(reloaded),
        _impl._digest_json(observed),
    )
    return reloaded


def build_repair_contract(*, sources, matched, device: str) -> dict:
    contract = _ORIGINAL_BUILD_REPAIR_CONTRACT(
        sources=sources,
        matched=matched,
        device=device,
    )
    contract["artifact_validation_v2"] = {
        "v1_completed_output_status": "preserved_provisional_not_reused",
        "policy_or_assignment_source_change": False,
        "cost_component_mapping_order_has_semantics": False,
        "exact_cost_component_key_set_required": True,
        "lexicographic_order_source": "objective_lexicographic_fields",
        "objective_lexicographic_fields_separately_authenticated": True,
        "execute_serialize_reload_required_before_completion": True,
    }
    contract.pop("contract_sha256", None)
    contract["contract_sha256"] = _impl._digest_json(contract)
    return contract


def _build_parser():
    parser = _ORIGINAL_BUILD_PARSER()
    for action in parser._actions:
        if action.dest == "output_dir":
            action.default = _HERE / "results" / DEFAULT_OUTPUT_DIRECTORY
            break
    else:  # pragma: no cover - immutable V1 parser contract changed
        raise RuntimeError("immutable V1 parser has no output_dir argument")
    parser.description = (
        "Authenticate the completed matched development comparison and "
        "execute the exact four-by-twelve V2 capacity-aware GA repair grid "
        "with order-independent serialized cost-component validation"
    )
    return parser


# Configure only the isolated implementation module.  Importing this file
# never mutates the canonical V1 module used to authenticate its artifacts.
_impl.PROTOCOL = PROTOCOL
_impl.SCHEMA_VERSION = SCHEMA_VERSION
_impl._capacity_audit_issues = _capacity_audit_issues
_impl._current_source_paths = _current_source_paths
_impl.build_repair_contract = build_repair_contract
_impl._build_parser = _build_parser
_impl._load_or_execute_repaired = _load_or_execute_repaired


def main(argv: Optional[Sequence[str]] = None) -> dict:
    return _impl.main(argv)


def __getattr__(name: str):
    """Expose the immutable implementation API for focused audit/tests."""

    return getattr(_impl, name)


if __name__ == "__main__":
    main()
