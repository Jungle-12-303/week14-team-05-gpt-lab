"""Run B pretraining hyperparameter experiments."""

from __future__ import annotations

import argparse

from _pretrain_runner import add_common_args, run_experiment_suite


EXPERIMENTS = [
    {"id": "B1_bs2", "group": "B1", "change": "batch_size", "value": 2, "overrides": {"batch_size": 2}},
    {"id": "B1_bs4", "group": "B1", "change": "batch_size", "value": 4, "overrides": {"batch_size": 4}},
    {"id": "B1_bs8", "group": "B1", "change": "batch_size", "value": 8, "overrides": {"batch_size": 8}},
    {"id": "B1_bs16", "group": "B1", "change": "batch_size", "value": 16, "overrides": {"batch_size": 16}},
    {"id": "B2_lr1e-4", "group": "B2", "change": "learning_rate", "value": 1e-4, "overrides": {"learning_rate": 1e-4}},
    {"id": "B2_lr3e-4", "group": "B2", "change": "learning_rate", "value": 3e-4, "overrides": {"learning_rate": 3e-4}},
    {"id": "B2_lr5e-4", "group": "B2", "change": "learning_rate", "value": 5e-4, "overrides": {"learning_rate": 5e-4}},
    {"id": "B3_drop0.0", "group": "B3", "change": "drop_rate", "value": 0.0, "overrides": {"drop_rate": 0.0}},
    {"id": "B3_drop0.1", "group": "B3", "change": "drop_rate", "value": 0.1, "overrides": {"drop_rate": 0.1}},
    {"id": "B3_drop0.2", "group": "B3", "change": "drop_rate", "value": 0.2, "overrides": {"drop_rate": 0.2}},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        experiment_choices=["all", "B1", "B2", "B3", *[item["id"] for item in EXPERIMENTS]],
        default_owner="YEONGBEEN",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment_suite(args, suite_name="B_pretrain_hparams", experiments=EXPERIMENTS)


if __name__ == "__main__":
    main()
