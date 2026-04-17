"""Compare synthesis methods on SCF benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.scf_dgp_benchmark import load_scf_surveys
from microplex.dgp import PopulationDGP
from microplex.dgp_methods import (
    GaussianCopulaMethod,
    MeanImputationMethod,
    VAEMethod,
    run_method_comparison,
)


def main():
    print("=" * 70)
    print("Multi-Source Population Synthesis: Method Comparison")
    print("=" * 70)
    print()

    surveys, shared_cols = load_scf_surveys()

    # QRF without ZI (threshold=1.0 means nothing is zero-inflated)
    qrf_no_zi = PopulationDGP(random_state=42, zero_inflation_threshold=1.0)
    qrf_no_zi.name = "QRF"

    # QRF with ZI (default threshold=0.1)
    qrf_zi = PopulationDGP(random_state=42, zero_inflation_threshold=0.1)
    qrf_zi.name = "QRF+ZI"

    methods = [
        MeanImputationMethod(),
        GaussianCopulaMethod(),
        qrf_no_zi,
        qrf_zi,
        VAEMethod(latent_dim=16, hidden_dim=64, epochs=100),
    ]

    # Try to add CTGAN
    try:
        from microplex.dgp_methods import CTGANMethod
        methods.append(CTGANMethod(epochs=100))
    except ImportError:
        print("(CTGAN not available - install sdv for GAN comparison)")
        print()

    run_method_comparison(
        surveys=surveys,
        shared_cols=shared_cols,
        methods=methods,
        holdout_frac=0.2,
        seed=42,
    )

    print()
    print("Interpretation:")
    print("  - Coverage: fraction of holdout records with nearby synthetic")
    print("  - Higher coverage → synthetic data spans real distribution")
    print("  - QRF uses conditional factorization P(X|shared)")
    print("  - VAE learns latent structure, may capture cross-survey dependencies")


if __name__ == "__main__":
    main()
