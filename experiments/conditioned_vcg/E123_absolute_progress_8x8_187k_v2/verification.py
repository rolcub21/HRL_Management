"""Controlled real-replay parity and timing; never updates the source files."""

from time import perf_counter
from copy import deepcopy
import random

import torch

from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import fast_inputs
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1.model import ProgressAgent
from .acceleration import SharedProgressAgent, compact_transitions, cache_session
from . import protocol as p


def equal_values(first, second):
    if torch.is_tensor(first):
        return torch.is_tensor(second) and torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(equal_values(first[k], second[k]) for k in first)
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(equal_values(a, b) for a, b in zip(first, second))
    return first == second


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def td_once(agent, batch):
    synchronize(agent.device)
    started = perf_counter()
    prediction, target, anchor, pairs = agent._td_batch(batch)
    synchronize(agent.device)
    elapsed = perf_counter() - started
    result = (prediction.detach().cpu(), target.detach().cpu(), anchor.detach().cpu(), pairs)
    return result, elapsed


def preflight(output, device, seed):
    from .run import prepare, source_payload, controller, read, write, require
    contract = prepare(output)
    require(device != "cuda" or torch.cuda.is_available(), "CUDA unavailable for requested parity probe")
    source_path, payload = source_payload("A_outflow_bounded", seed)
    source_sha = p.parent.parent.attribution.sha256(source_path)
    original_gate = read(p.SOURCE_OUTPUT / "preflight.json")
    require(original_gate["passed"] and original_gate["contract_sha256"] == p.source.contract()["contract_sha256"],
            "original liveness gate is not authenticated")
    state = dict(payload["agent_state"])
    replay = dict(state["replay"])
    replay["memory"], sharing = compact_transitions(replay["memory"])
    state["replay"] = replay
    compact = {**payload, "agent_state": state}
    baseline = ProgressAgent.from_checkpoint(payload, device=device, resumable=True, seed=seed)
    corrected = SharedProgressAgent.from_checkpoint(compact, device=device, resumable=True, seed=seed)
    # Optimizer loading may retain CPU scalar-step tensor identities. The two
    # disposable probes must not share those mutable tensors with each other.
    baseline.optimizer.load_state_dict(deepcopy(payload["agent_state"]["optimizer"]))
    corrected.optimizer.load_state_dict(deepcopy(payload["agent_state"]["optimizer"]))
    probe_rng = random.Random()
    probe_rng.setstate(baseline.rng.getstate())
    # Equal RNG draws and replay ordering select the same transitions.
    reference_batch = baseline.replay.sample_mode_balanced(baseline.config.batch_size, probe_rng)
    probe_rng.setstate(corrected.rng.getstate())
    corrected_batch = corrected.replay.sample_mode_balanced(corrected.config.batch_size, probe_rng)
    require(all(a == b for a, b in zip(reference_batch, corrected_batch)), "migration changed sampled transitions")
    print(f"E123 v2 gate | {device} | batch={len(reference_batch)} "
          f"next_candidates={sum(len(t.next_candidates) for t in reference_batch)}", flush=True)
    with cache_session(), controller("A_outflow_bounded"):
        reference, cold = td_once(baseline, reference_batch)
        reference_warm, warm = td_once(baseline, reference_batch)
        fast_inputs._REGISTRY.clear()  # simulate the v1 episode/validation boundary
        reference_again, after_boundary = td_once(baseline, reference_batch)
    with cache_session(), controller("A_outflow_bounded"):
        actual, corrected_cold = td_once(corrected, corrected_batch)
        actual_warm, corrected_warm = td_once(corrected, corrected_batch)
        actual_again, retained = td_once(corrected, corrected_batch)
        retained_stats = fast_inputs.cache_stats()
    td_equal = all(equal_values(reference, result) for result in
                   (reference_warm, reference_again, actual, actual_warm, actual_again))
    # One disposable real Adam update checks more than target equality.
    # The two local sampler RNGs have not been advanced by the timing batches.
    with cache_session(), controller("A_outflow_bounded"):
        old_loss = baseline.learn()
    with cache_session(), controller("A_outflow_bounded"):
        new_loss = corrected.learn()
    old_state = baseline.checkpoint_state(include_replay=False)
    new_state = corrected.checkpoint_state(include_replay=False)
    update_equal = equal_values(old_state, new_state) and old_loss == new_loss
    restored = SharedProgressAgent.from_checkpoint(corrected.checkpoint(include_replay=True),
                                                  device=device, resumable=True, seed=seed)
    resume_equal = equal_values(new_state, restored.checkpoint_state(include_replay=False))
    unchanged_source = p.parent.parent.attribution.sha256(source_path) == source_sha
    result = {"contract_sha256": contract["contract_sha256"], "passed": bool(td_equal and update_equal and resume_equal and unchanged_source),
              "device": str(device), "source_checkpoint": str(source_path), "source_checkpoint_sha256": source_sha,
              "source_episode": payload["completed_training_episodes"], "batch_size": len(reference_batch),
              "next_candidate_records": sum(len(t.next_candidates) for t in reference_batch),
              "predictions_targets_anchor_exactly_equal": td_equal,
              "adam_update_optimizer_rng_counters_exactly_equal": update_equal,
              "resumable_checkpoint_exactly_equal": resume_equal, "source_file_unchanged": unchanged_source,
              "sharing": sharing, "loss": new_loss, "reference_loss": old_loss,
              "td_seconds": {"v1_cold": cold, "v1_warm": warm, "v1_after_episode_clear": after_boundary,
                             "v2_cold": corrected_cold, "v2_warm": corrected_warm, "v2_retained_across_boundary": retained},
              "retained_input_cache": retained_stats,
              "scope": "fixed real replay batch; not an end-to-end episode speedup claim",
              "training_runs": 0, "disposable_optimizer_updates": 2, "episode_runs": 0,
              "original_absolute_liveness_gate_passed": True}
    write(output / "preflight.json", result)
    require(result["passed"], "E123-v2 numerical/update/resume parity gate failed; do not train")
    return result
