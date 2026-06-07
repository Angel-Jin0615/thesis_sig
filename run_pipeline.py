from __future__ import annotations

from src.clustering.build_signature_dataset import build_signature_dataset
from src.clustering.cluster_stocks import cluster_stocks
from src.data.download_data import download_universe
from src.data.process_data import process_all
from src.evaluation.evaluate_generation import evaluate_generated_scenarios
from src.models.run_signature_bootstrap import run_generation
from src.utils.config import ensure_directories_from_config, load_config


def main() -> None:
    config = load_config()
    ensure_directories_from_config(config)

    print("=" * 80)
    print("1) Downloading data")
    download_universe(config)

    print("=" * 80)
    print("2) Processing data")
    process_all(config)

    print("=" * 80)
    print("3) Building signature dataset (caching + parallel)")
    build_signature_dataset(config)

    print("=" * 80)
    print("4) Clustering stocks")
    cluster_stocks(config)

    print("=" * 80)
    print("5) Generating scenarios (historical + signature bootstrap)")
    run_generation(config)

    print("=" * 80)
    print("6) Evaluating generated scenarios")
    evaluate_generated_scenarios(config)

    print("=" * 80)
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
