"""Shared acquisition-gain composition for the R12 two-by-two factorial."""

from __future__ import annotations

import numpy as np

from .latent_detection import LatentRadiusPosterior, SigmaKernelPosterior


OBSERVATION_LEVELS = ("sigma_kernel", "latent_radius")
PASSAGE_LEVELS = ("nearest_recorded", "shared_learned")


class FactorialAcquisition:
    """Compose one observation-memory level with one passage-source level."""

    def __init__(
        self,
        *,
        recorded_goal_xy,
        sigma,
        observation_level,
        passage_level,
        learned_passage=None,
        matched_passage=None,
    ):
        if observation_level not in OBSERVATION_LEVELS:
            raise ValueError(f"unknown observation level: {observation_level}")
        if passage_level not in PASSAGE_LEVELS:
            raise ValueError(f"unknown passage level: {passage_level}")
        if passage_level == "shared_learned" and learned_passage is None:
            raise ValueError("shared_learned requires the shared passage runtime")
        if passage_level == "nearest_recorded" and matched_passage is None:
            raise ValueError("nearest_recorded requires a domain geometry adapter")
        self.recorded_goal_xy = np.asarray(recorded_goal_xy, np.float32)
        self.sigma = float(sigma)
        self.observation_level = observation_level
        self.passage_level = passage_level
        self.learned_passage = learned_passage
        self.matched_passage = matched_passage
        self.reset()

    def reset(self):
        posterior = (
            LatentRadiusPosterior
            if self.observation_level == "latent_radius"
            else SigmaKernelPosterior
        )
        self.posterior = posterior(self.recorded_goal_xy, self.sigma)

    def observe(self, xy, reward):
        self.posterior.update(xy, reward)

    def passages(self, states, commands, endpoint_xy):
        states = np.asarray(states, np.float32)
        commands = np.asarray(commands, np.float32)
        endpoint = np.asarray(endpoint_xy, np.float32).reshape(-1, 2)
        if self.passage_level == "shared_learned":
            return self.learned_passage.predict(states, commands)
        value = self.matched_passage(states[:, :2], endpoint)
        value = np.asarray(value, np.float32)
        if value.ndim != 3 or value.shape[0] != len(states) or value.shape[2] != 2:
            raise ValueError("matched passage adapter must return [candidate,point,2]")
        return value[None]

    def gains(self, states, commands, endpoint_xy):
        return self.gains_for(self.posterior, states, commands, endpoint_xy)

    def gains_for(self, posterior, states, commands, endpoint_xy):
        passages = self.passages(states, commands, endpoint_xy)
        member = np.stack([
            posterior.contact_probability(value) for value in passages
        ])
        # One global conservative rule for learned-passage uncertainty.  The
        # nearest-recorded control has one member and is therefore unchanged.
        return member.min(axis=0).astype(np.float32), member.astype(np.float32)

    def diagnostic(self):
        value = {
            "observation_memory_factor": self.observation_level,
            "passage_prediction_factor": self.passage_level,
            "candidate_goal_source": "unique_recorded_roadmap_state_xy",
            "candidate_goal_count": int(len(self.posterior.goal_xy)),
            "posterior_exhausted": bool(self.posterior.exhausted),
            "posterior_exhaustion_behavior": "fatal_model_falsification",
        }
        if isinstance(self.posterior, LatentRadiusPosterior):
            summary = self.posterior.summary()
            value.update({
                "rho_min_data_derived_m": summary.rho_min,
                "rho_max_data_derived_m": summary.rho_max,
                "feasible_goal_candidates": summary.feasible_candidate_count,
                "posterior_marginal_evidence": summary.marginal_evidence,
            })
        return value
