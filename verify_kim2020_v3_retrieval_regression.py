#!/usr/bin/env python3
"""Verify the eight retrieval-admission failures repaired by executor v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KIM_STOCHASTIC = "kim2020_a3c_spatial_adapted__stochastic"
KIM_MAP = "kim2020_a3c_spatial_adapted__map"
EXPECTED_COMPARISON_VERSION = "kim2020_spatial_matched_track_b_v3"
EXPECTED_OPTION_VERSION = "retrieve_deliver_option_v3"
EXPECTED_EXECUTOR_VERSION = "named_atomic_retrieval_executor_v3"
EXPECTED_START_CONTRACT = (
    "canonical_plan_plus_complete_current_live_multileg_v3"
)

# (method, schedule seed, policy seed, historical route failure)
RETRIEVAL_V3_REGRESSION_CASES = (
    (KIM_STOCHASTIC, 44008, 20_000_000, "delivery_route_failed"),
    (KIM_MAP, 44030, 20_000_005, "delivery_route_failed"),
    (KIM_STOCHASTIC, 44031, 20_000_000, "delivery_route_failed"),
    (KIM_STOCHASTIC, 44033, 20_000_002, "delivery_route_failed"),
    (KIM_STOCHASTIC, 44040, 20_000_004, "delivery_route_failed"),
    (KIM_STOCHASTIC, 44047, 20_000_000, "relocation_route_failed"),
    (KIM_STOCHASTIC, 44047, 20_000_002, "relocation_route_failed"),
    (KIM_STOCHASTIC, 44049, 20_000_000, "delivery_route_failed"),
)


def _index(payload):
    result = {}
    for run in payload.get("raw_runs", ()):
        key = (
            run.get("comparison_method"),
            run.get("schedule_seed"),
            run.get("assignment_policy_seed"),
        )
        if key in result:
            raise AssertionError(f"duplicate raw regression run: {key!r}")
        result[key] = run
    return result


def verify_regression(before, after):
    """Assert that every exact v2 failure is successful under v3."""

    if after.get("config", {}).get("comparison_version") != (
        EXPECTED_COMPARISON_VERSION
    ):
        raise AssertionError("after artifact is not a v3 comparison")
    old_runs = _index(before)
    new_runs = _index(after)
    verified = []
    for method, schedule_seed, policy_seed, historical_reason in (
        RETRIEVAL_V3_REGRESSION_CASES
    ):
        key = (method, schedule_seed, policy_seed)
        if key not in old_runs or key not in new_runs:
            raise AssertionError(f"missing exact regression run: {key!r}")
        old = old_runs[key]
        expected_failure = (
            f"macro_failure:StrictRetrieveDeliverOption:{historical_reason}"
        )
        if old.get("method_failure_reason") != expected_failure:
            raise AssertionError(
                f"historical failure mismatch for {key!r}: "
                f"{old.get('method_failure_reason')!r}"
            )

        new = new_runs[key]
        audit = new.get("urgency_scheduler_audit", {})
        checks = {
            "protocol_valid": new.get("protocol_valid") is True,
            "strict_method_success": (
                float(new.get("strict_method_success", 0.0)) == 1.0
            ),
            "no_method_failure": new.get("method_failure_reason") is None,
            "reservation_integrity": (
                new.get("reservation_integrity") is True
            ),
            "no_illegal_drops": int(new.get("illegal_drops", -1)) == 0,
            "option_v3": (
                audit.get("retrieve_deliver_option_version")
                == EXPECTED_OPTION_VERSION
            ),
            "executor_v3": (
                audit.get("retrieval_executor_version")
                == EXPECTED_EXECUTOR_VERSION
            ),
            "start_contract_v3": (
                audit.get("retrieval_start_contract")
                == EXPECTED_START_CONTRACT
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise AssertionError(
                f"v3 regression failed for {key!r}: {', '.join(failed)}"
            )
        verified.append(
            {
                "method": method,
                "schedule_seed": schedule_seed,
                "policy_seed": policy_seed,
                "historical_failure": historical_reason,
                "v3_protocol_valid": True,
            }
        )
    return verified


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.before.open(encoding="utf-8") as handle:
        before = json.load(handle)
    with args.after.open(encoding="utf-8") as handle:
        after = json.load(handle)
    verified = verify_regression(before, after)
    print(
        json.dumps(
            {
                "verified_case_count": len(verified),
                "all_protocol_valid": True,
                "cases": verified,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

