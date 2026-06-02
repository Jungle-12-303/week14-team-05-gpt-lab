"""Run C pretraining architecture experiments."""

from __future__ import annotations

import argparse

from _pretrain_runner import add_common_args, run_experiment_suite


EXPERIMENTS = [
    {"id": "C1_ctx64", "group": "C1", "change": "context_length", "value": 64, "overrides": {"context_length": 64}},
    {"id": "C1_ctx128", "group": "C1", "change": "context_length", "value": 128, "overrides": {"context_length": 128}},
    {"id": "C2_layers1", "group": "C2", "change": "n_layers", "value": 1, "overrides": {"n_layers": 1}},
    {"id": "C2_layers2", "group": "C2", "change": "n_layers", "value": 2, "overrides": {"n_layers": 2}},
    {"id": "C2_layers4", "group": "C2", "change": "n_layers", "value": 4, "overrides": {"n_layers": 4}},
    {"id": "C3_dim64", "group": "C3", "change": "emb_dim", "value": 64, "overrides": {"emb_dim": 64}},
    {"id": "C3_dim128", "group": "C3", "change": "emb_dim", "value": 128, "overrides": {"emb_dim": 128}},
    {"id": "C3_dim192", "group": "C3", "change": "emb_dim", "value": 192, "overrides": {"emb_dim": 192}},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        experiment_choices=["all", "C1", "C2", "C3", *[item["id"] for item in EXPERIMENTS]],
        default_owner="BEOMSANG",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment_suite(args, suite_name="C_pretrain_architecture", experiments=EXPERIMENTS)


if __name__ == "__main__":
    main()
