"""Run A pretraining stability experiments."""

from __future__ import annotations

import argparse

from _pretrain_runner import add_common_args, run_experiment_suite


EXPERIMENTS = [
    {
        "id": "A0",
        "change": "screening baseline",
        "overrides": {},
    },
    {
        "id": "A0_basic",
        "change": "Basic submission baseline",
        "overrides": {"context_length": 128},
    },
    {
        "id": "A1",
        "change": "warmup + cosine decay",
        "overrides": {"scheduler": "warmup_cosine"},
    },
    {
        "id": "A2",
        "change": "gradient clipping",
        "overrides": {"grad_clip_norm": 1.0},
    },
    {
        "id": "A3",
        "change": "weight_decay=0.01",
        "overrides": {"weight_decay": 0.01},
    },
    {
        "id": "A4",
        "change": "warmup + cosine + clipping + weight_decay",
        "overrides": {
            "scheduler": "warmup_cosine",
            "grad_clip_norm": 1.0,
            "weight_decay": 0.01,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        experiment_choices=["all", *[item["id"] for item in EXPERIMENTS]],
        default_owner="JAEHWAN",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment_suite(args, suite_name="A_pretrain_stability", experiments=EXPERIMENTS)


if __name__ == "__main__":
    main()
