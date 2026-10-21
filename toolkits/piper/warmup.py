# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare labels, cache frozen features, and run offline Piper RLT warmup."""

import argparse
import json
from pathlib import Path


def main() -> None:
    """Expose init/cache/train through one configuration and a small local CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "cache", "relabel-cache", "train"))
    parser.add_argument(
        "--source-cache",
        type=Path,
        help="relabel-cache: verified legacy frozen features",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "examples/offline_rl/config/piper_rlt_warmup.yaml",
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--dataset",
        type=Path,
        help="init: draft labels for an existing LeRobot dataset",
    )
    sources.add_argument(
        "--collection",
        type=Path,
        action="append",
        help="init: recorded annotations.json; repeat to merge runs",
    )
    parser.add_argument(
        "--skip-unlabeled",
        action="store_true",
        help="init: explicitly exclude episodes with missing, unpaired or invalid phases",
    )
    parser.add_argument(
        "--resume", action="store_true", help="train: resume output/latest.json"
    )
    parser.add_argument(
        "--until-step",
        type=int,
        help="train: checkpoint at this absolute step within the fixed budget",
    )
    args, overrides = parser.parse_known_args()
    if (args.command == "relabel-cache") != (args.source_cache is not None):
        parser.error(
            "relabel-cache requires --source-cache; other commands do not accept it."
        )
    if any("=" not in item or item.startswith("--") for item in overrides):
        parser.error(
            "Extra arguments must use dotted overrides, e.g. training.batch_size=64."
        )
    if (
        args.dataset or args.collection or args.skip_unlabeled
    ) and args.command != "init":
        parser.error("Data source and --skip-unlabeled options apply to init.")
    if (args.resume or args.until_step is not None) and args.command != "train":
        parser.error("--resume and --until-step apply to train.")
    if args.command == "init" and not (args.dataset or args.collection):
        parser.error("init requires --dataset or --collection.")
    from rlinf.runners.rlt_warmup_runner import RLTWarmupRunner, load_warmup_config

    cfg = load_warmup_config(args.config, overrides)
    if args.command == "init":
        from rlinf.data.datasets.rlt_warmup import (
            create_annotation_template,
            split_collection_annotations,
        )

        if args.collection:
            path = split_collection_annotations(
                args.collection,
                cfg.annotations,
                seed=cfg.seed,
                skip_unlabeled=args.skip_unlabeled,
            )
        else:
            path = create_annotation_template(
                args.dataset, cfg.annotations, seed=cfg.seed
            )
        print(f"Annotations: {path}")
    elif args.command == "cache":
        from rlinf.data.datasets.rlt_warmup import prepare_piper_cache

        manifest = prepare_piper_cache(
            cfg.stage1,
            cfg.annotations,
            cfg.cache,
            chunk_length=cfg.chunk_length,
            stride=cfg.stride,
            seed=cfg.seed,
            device=cfg.device,
            video_backend=cfg.video_backend,
        )
        print(
            json.dumps(
                {
                    "cache": str(Path(cfg.cache).resolve()),
                    "episodes": len(manifest["episodes"]),
                    "fingerprint": manifest["content_fingerprint"],
                },
                indent=2,
            )
        )
    elif args.command == "relabel-cache":
        from rlinf.data.datasets.rlt_warmup import relabel_cached_segments

        manifest = relabel_cached_segments(
            args.source_cache, cfg.annotations, cfg.cache, stage1=cfg.stage1
        )
        print(
            json.dumps(
                {"cache": cfg.cache, "fingerprint": manifest["content_fingerprint"]},
                indent=2,
            )
        )
    else:
        result = RLTWarmupRunner(cfg).run(
            resume=args.resume, until_step=args.until_step
        )
        print(
            json.dumps({k: v for k, v in result.items() if k != "validation"}, indent=2)
        )


if __name__ == "__main__":
    main()
