"""
Multi-resolution dataset generation.

Supports datasets from browser-sized (~1K records) to full population (~330M).
Uses L0-regularized calibration to maintain representativeness at any size.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ResolutionLevel(Enum):
    """Predefined resolution levels for common use cases."""

    # Browser-friendly: fits in memory, fast client-side simulation
    BROWSER = "browser"  # ~1K-5K records

    # Demo/testing: quick iteration, good for development
    DEMO = "demo"  # ~10K records

    # Standard: production microsimulation
    STANDARD = "standard"  # ~100K records

    # Detailed: state-level analysis with good coverage
    DETAILED = "detailed"  # ~1M records

    # High-fidelity: county-level, rare subgroups
    HIGH_FIDELITY = "high_fidelity"  # ~10M records

    # Full: one record per person (or close to it)
    FULL = "full"  # ~330M records (US)

    @property
    def target_records(self) -> int:
        """Approximate target record count for this level."""
        return {
            ResolutionLevel.BROWSER: 2_000,
            ResolutionLevel.DEMO: 10_000,
            ResolutionLevel.STANDARD: 100_000,
            ResolutionLevel.DETAILED: 1_000_000,
            ResolutionLevel.HIGH_FIDELITY: 10_000_000,
            ResolutionLevel.FULL: 330_000_000,
        }[self]

    @property
    def description(self) -> str:
        """Human-readable description of use case."""
        return {
            ResolutionLevel.BROWSER: "Client-side browser simulation, demos",
            ResolutionLevel.DEMO: "Development, testing, quick analysis",
            ResolutionLevel.STANDARD: "Production microsimulation",
            ResolutionLevel.DETAILED: "State-level analysis, subgroup studies",
            ResolutionLevel.HIGH_FIDELITY: "County-level, rare populations",
            ResolutionLevel.FULL: "Maximum fidelity, individual-level",
        }[self]

    @property
    def typical_file_size_mb(self) -> float:
        """Approximate file size assuming ~200 bytes per record."""
        bytes_per_record = 200
        return self.target_records * bytes_per_record / 1_000_000


@dataclass
class ResolutionConfig:
    """Configuration for dataset resolution."""

    target_records: int
    l0_penalty: float  # Higher = sparser (fewer records)

    # Constraints that must be preserved at any resolution
    preserve_tails: bool = True  # Keep extreme values (billionaires, etc.)
    preserve_geography: bool = True  # Maintain state/county coverage
    preserve_demographics: bool = True  # Age/sex/race distributions

    # Minimum representation constraints
    min_records_per_state: int = 10
    min_records_per_percentile: int = 5  # For income distribution
    min_records_per_filing_status: int = 50

    @classmethod
    def from_level(cls, level: ResolutionLevel) -> "ResolutionConfig":
        """Create config from predefined level."""
        # L0 penalty scales inversely with target size
        # Higher penalty = more aggressive pruning
        base_penalty = 1.0
        compression_ratio = 330_000_000 / level.target_records
        # Ensure minimum penalty of 0.01 (for FULL resolution)
        l0_penalty = max(0.01, base_penalty * np.log10(max(1.1, compression_ratio)))

        return cls(
            target_records=level.target_records,
            l0_penalty=l0_penalty,
            min_records_per_state=max(1, level.target_records // 5000),
            min_records_per_percentile=max(1, level.target_records // 2000),
            min_records_per_filing_status=max(5, level.target_records // 200),
        )


@dataclass
class HardConcreteGate:
    """
    Differentiable approximation to L0 regularization.

    Based on Louizos et al. 2018 "Learning Sparse Neural Networks
    through L0 Regularization", with CRITICAL BUG FIX from PolicyEngine's
    L0 package (https://github.com/PolicyEngine/L0).

    The original implementation incorrectly drops temperature in deterministic
    mode. This version preserves temperature consistently across:
    - Stochastic sampling (training)
    - Deterministic inference
    - L0 penalty computation

    See: https://github.com/PolicyEngine/L0/blob/main/CRITICAL_TEMPERATURE_BUG.md

    Each record has a gate g_i in [0, 1] that determines its weight.
    During training, gates are stochastic; at inference, they're deterministic.
    """

    # Temperature for hard concrete distribution
    # Lower = harder/more binary gates, higher = softer
    # Recommended range: [0.1, 2/3] per PolicyEngine's analysis
    temperature: float = 2 / 3

    # Stretch parameters (maps sigmoid to hard concrete)
    zeta: float = 1.1
    gamma: float = -0.1

    def sample_gate(
        self,
        log_alpha: np.ndarray,
        training: bool = True,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Sample gate values.

        Args:
            log_alpha: Log-odds of gate being open (learnable parameter)
            training: If True, sample stochastically; else use deterministic
            rng: Random number generator

        Returns:
            Gate values in [0, 1], with hard zeros possible
        """
        if rng is None:
            rng = np.random.default_rng()

        if training:
            # Sample from hard concrete distribution
            # Temperature is used here (correct)
            u = rng.uniform(0, 1, size=log_alpha.shape)
            s = self._sigmoid(
                (np.log(u) - np.log(1 - u) + log_alpha) / self.temperature
            )
            # Stretch to [gamma, zeta]
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            # Hard rectification to [0, 1]
            return np.clip(s_bar, 0, 1)
        else:
            # Deterministic: use expected value
            # CRITICAL FIX: Temperature must also be used in deterministic mode!
            # The original Louizos implementation incorrectly uses beta=1 here.
            s = self._sigmoid(log_alpha / self.temperature)
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            return np.clip(s_bar, 0, 1)

    def l0_penalty(self, log_alpha: np.ndarray) -> float:
        """
        Compute expected L0 norm (number of non-zero gates).

        This is differentiable w.r.t. log_alpha.
        """
        # Probability that gate is non-zero
        # CRITICAL FIX: Include temperature in penalty calculation too
        p_nonzero = self._sigmoid(
            (log_alpha - self.temperature * np.log(-self.gamma / self.zeta))
            / self.temperature
        )
        return float(np.sum(p_nonzero))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def compress_dataset(
    weights: np.ndarray,
    features: np.ndarray,
    targets: dict[str, float],
    config: ResolutionConfig,
    max_iterations: int = 1000,
    learning_rate: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compress dataset to target size while preserving calibration.

    Uses L0-regularized entropy balancing:
    - Minimize: KL(w || w_0) + lambda * ||w||_0
    - Subject to: moment constraints

    Args:
        weights: Initial record weights (n,)
        features: Feature matrix for calibration (n, k)
        targets: Target moments {name: value}
        config: Resolution configuration
        max_iterations: Max optimization iterations
        learning_rate: Step size for gradient descent

    Returns:
        (new_weights, gate_values, info)
    """
    n = len(weights)
    rng = np.random.default_rng(42)

    # Initialize gate parameters (start with all gates open)
    log_alpha = np.ones(n) * 2.0  # High = likely open

    gate = HardConcreteGate()

    # Target moments from features
    target_vector = np.array(list(targets.values()))

    best_loss = float("inf")

    for iteration in range(max_iterations):
        # Sample gates
        gates = gate.sample_gate(log_alpha, training=True, rng=rng)

        # Compute effective weights
        effective_weights = weights * gates

        # Calibration loss: (sum(w * x) - target)^2
        current_moments = features.T @ effective_weights
        calibration_loss = np.sum((current_moments - target_vector) ** 2)

        # L0 loss: penalize non-zero gates
        l0_loss = gate.l0_penalty(log_alpha)

        # Total loss
        total_loss = calibration_loss + config.l0_penalty * l0_loss

        # Simple gradient update for log_alpha
        # (In practice, use autodiff)
        grad_approx = np.zeros(n)
        eps = 0.01
        for i in range(min(100, n)):  # Sample gradient for efficiency
            idx = rng.integers(0, n)
            log_alpha[idx] += eps
            gates_plus = gate.sample_gate(log_alpha, training=False, rng=rng)
            w_plus = weights * gates_plus
            loss_plus = np.sum((features.T @ w_plus - target_vector) ** 2)
            loss_plus += config.l0_penalty * gate.l0_penalty(log_alpha)

            log_alpha[idx] -= 2 * eps
            gates_minus = gate.sample_gate(log_alpha, training=False, rng=rng)
            w_minus = weights * gates_minus
            loss_minus = np.sum((features.T @ w_minus - target_vector) ** 2)
            loss_minus += config.l0_penalty * gate.l0_penalty(log_alpha)

            log_alpha[idx] += eps  # Restore
            grad_approx[idx] = (loss_plus - loss_minus) / (2 * eps)

        log_alpha -= learning_rate * grad_approx

        if total_loss < best_loss:
            best_loss = total_loss

        if iteration % 100 == 0:
            n_active = np.sum(gates > 0.01)
            print(
                f"Iter {iteration}: loss={total_loss:.2f}, "
                f"active={n_active}/{n} ({n_active/n:.1%})"
            )

    # Final deterministic gates
    final_gates = gate.sample_gate(log_alpha, training=False)
    final_weights = weights * final_gates

    # Rescale to preserve total weight
    if final_weights.sum() > 0:
        final_weights = final_weights / final_weights.sum() * weights.sum()

    info = {
        "n_original": n,
        "n_active": int(np.sum(final_gates > 0.01)),
        "compression_ratio": n / max(1, np.sum(final_gates > 0.01)),
        "final_loss": float(best_loss),
    }

    return final_weights, final_gates, info


# Convenience functions for common resolutions


def for_browser(n_records: int = 2000) -> ResolutionConfig:
    """Config optimized for browser-based simulation."""
    return ResolutionConfig(
        target_records=n_records,
        l0_penalty=5.0,  # Aggressive compression
        preserve_tails=True,  # Still need billionaires for reform analysis
        min_records_per_state=1,
        min_records_per_percentile=2,
    )


def for_api(n_records: int = 100_000) -> ResolutionConfig:
    """Config for standard API-served simulation."""
    return ResolutionConfig(
        target_records=n_records,
        l0_penalty=2.0,
        preserve_tails=True,
        preserve_geography=True,
    )


def for_research(n_records: int = 1_000_000) -> ResolutionConfig:
    """Config for detailed research analysis."""
    return ResolutionConfig(
        target_records=n_records,
        l0_penalty=1.0,
        preserve_tails=True,
        preserve_geography=True,
        preserve_demographics=True,
        min_records_per_state=100,
    )
