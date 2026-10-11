"""Versioned construction of Track-B controller action interfaces."""

from __future__ import annotations

from collections import Counter

from example.Options.DeliverOption import DeliverOption
from example.Options.AcceptStoreOption import (
    AcceptStoreOption,
    ReservedAcceptStoreOption,
)
from example.Options.PickupRipeOption import PickupRipeOption
from example.Options.RetrieveDeliverOption import (
    RetrieveDeliverOption,
    StrictRetrieveDeliverOption,
)
from example.Options.StrategicDeferOption import StrategicDeferOption
from example.Options.pickupOption import PickupOption
from example.Options.storeOption import StoreOption
from gated_agent import (
    ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE,
    LEGACY_SPLIT_CONTROLLER_ACTION_INTERFACE,
    SCHEDULER_CONTROLLER_ACTION_INTERFACE,
)
from primitive_option import PrimitiveOption
from PSLAP.neutral_protocol import neutral_relocation_selector
from example.urgency_scheduler import (
    RESERVED_CELL_ACTION_INTERFACE,
    URGENCY_FIRST_ACTION_INTERFACE,
)
from relational_scheduler import RELATIONAL_ACTION_INTERFACE


FULLY_LEARNED_RESERVED_ACTION_INTERFACE = (
    "relational_parameterized_reserved_macro_v2"
)


def build_controller_options(
    env,
    selector,
    *,
    controller_action_interface=LEGACY_SPLIT_CONTROLLER_ACTION_INTERFACE,
    retrieval_lead_time=20.0,
    include_retrieval_handling_steps=False,
    max_defer_steps=10,
):
    """Populate one exact, checkpoint-versioned controller interface."""

    if env.options:
        raise ValueError("controller options must be built on an empty option set")
    for action in env.get_action_space():
        env.options.add(PrimitiveOption(action, env))

    if controller_action_interface in (
        URGENCY_FIRST_ACTION_INTERFACE,
        RELATIONAL_ACTION_INTERFACE,
        RESERVED_CELL_ACTION_INTERFACE,
        FULLY_LEARNED_RESERVED_ACTION_INTERFACE,
    ):
        accept_class = (
            ReservedAcceptStoreOption
            if controller_action_interface
            in (
                RESERVED_CELL_ACTION_INTERFACE,
                FULLY_LEARNED_RESERVED_ACTION_INTERFACE,
            )
            else AcceptStoreOption
        )
        env.options.add(accept_class(env, selector))
        env.options.add(
            StrategicDeferOption(
                env,
                max_defer_steps=max_defer_steps,
                relocation_selector=neutral_relocation_selector,
            )
        )
        for block_index in range(len(env.blocks)):
            env.options.add(
                StrictRetrieveDeliverOption(
                    env,
                    block_index,
                    relocation_selector=neutral_relocation_selector,
                )
            )
        return

    if (
        controller_action_interface
        == ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE
    ):
        env.options.add(AcceptStoreOption(env, selector))
        env.options.add(
            StrategicDeferOption(
                env,
                max_defer_steps=max_defer_steps,
                relocation_selector=neutral_relocation_selector,
            )
        )
        for block_index in range(len(env.blocks)):
            env.options.add(
                RetrieveDeliverOption(
                    env,
                    block_index,
                    relocation_selector=neutral_relocation_selector,
                )
            )
        return

    if controller_action_interface == SCHEDULER_CONTROLLER_ACTION_INTERFACE:
        env.options.add(PickupOption(env))
        env.options.add(StoreOption(env))
        env.options.add(selector)
        env.options.add(
            StrategicDeferOption(
                env,
                max_defer_steps=max_defer_steps,
                relocation_selector=neutral_relocation_selector,
            )
        )
        for block_index in range(len(env.blocks)):
            env.options.add(
                RetrieveDeliverOption(
                    env,
                    block_index,
                    relocation_selector=neutral_relocation_selector,
                )
            )
        return

    if controller_action_interface != LEGACY_SPLIT_CONTROLLER_ACTION_INTERFACE:
        raise ValueError(
            "Unknown controller action interface: "
            f"{controller_action_interface!r}"
        )
    for option_class in (PickupOption, DeliverOption, StoreOption):
        env.options.add(option_class(env))
    env.options.add(
        PickupRipeOption(
            env,
            retrieval_lead_time=retrieval_lead_time,
            include_handling_steps=include_retrieval_handling_steps,
        )
    )
    env.options.add(selector)


def scheduler_episode_audit(env):
    """Return per-episode execution evidence for the scheduling interface."""

    options = getattr(env, "options", ())
    retrievals = sorted(
        (
            option
            for option in options
            if isinstance(option, RetrieveDeliverOption)
        ),
        key=lambda option: option.block_index,
    )
    defer = next(
        (
            option
            for option in options
            if isinstance(option, StrategicDeferOption)
        ),
        None,
    )
    inbound = next(
        (option for option in options if isinstance(option, AcceptStoreOption)),
        None,
    )
    retrieval_failure_reasons = Counter()
    retrieval_failure_events = []
    for option in retrievals:
        retrieval_failure_reasons.update(option.episode_failure_outcomes)
        retrieval_failure_events.extend(option.episode_failure_events)
    return {
        "retrieve_successes": int(
            sum(option.episode_success_count for option in retrievals)
        ),
        "retrieve_failures": int(
            sum(option.episode_failure_count for option in retrievals)
        ),
        "retrieve_relocations": int(
            sum(option.episode_relocations for option in retrievals)
        ),
        "retrieve_replans": int(
            sum(option.episode_replans for option in retrievals)
        ),
        "retrieve_failure_reasons": dict(retrieval_failure_reasons),
        "retrieve_failure_events": retrieval_failure_events,
        "retrieve_success_labels": [
            option.target_label
            for option in retrievals
            if option.episode_success_count
        ],
        "retrieve_outcomes": [
            {
                "block_label": option.target_label,
                **option.last_outcome,
            }
            for option in retrievals
            if isinstance(option.last_outcome, dict)
        ],
        "defer_outcomes": (
            dict(defer.episode_outcome_counts) if defer is not None else {}
        ),
        "inbound_successes": (
            int(inbound.episode_success_count) if inbound is not None else 0
        ),
        "inbound_failures": (
            int(inbound.episode_failure_count) if inbound is not None else 0
        ),
        "inbound_outcomes": (
            list(inbound.episode_outcomes) if inbound is not None else []
        ),
        "accept_store_option_version": (
            type(inbound).VERSION if inbound is not None else None
        ),
        "assignment_commitment_contract": (
            getattr(
                inbound,
                "COMMITMENT_CONTRACT",
                "post_pickup_recompute_v1",
            )
            if inbound is not None
            else None
        ),
        "reservation_contract": (
            getattr(inbound, "RESERVATION_CONTRACT", None)
            if inbound is not None
            else None
        ),
        "reservation_bound_count": int(
            getattr(inbound, "episode_reservation_bound_count", 0)
        ),
        "reservation_commit_count": int(
            getattr(inbound, "episode_reservation_commit_count", 0)
        ),
        "reservation_invalidation_count": int(
            getattr(inbound, "episode_reservation_invalidation_count", 0)
        ),
        "reservation_execution_match_count": int(
            getattr(
                inbound,
                "episode_reservation_execution_match_count",
                0,
            )
        ),
        "bound_proposal_ids": list(
            getattr(inbound, "episode_bound_proposal_ids", ())
        ),
        "committed_proposal_ids": list(
            getattr(inbound, "episode_committed_proposal_ids", ())
        ),
    }


__all__ = ["build_controller_options", "scheduler_episode_audit"]
