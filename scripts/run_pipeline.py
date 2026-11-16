import os
import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parent.parent))

from synth_damage_gen.config import load_config
from synth_damage_gen.pipeline.session import Session, SessionConfig


def main(args):
    cfg = load_config("configs/default.yaml")

    session_cfg = SessionConfig(
        session_id=1,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        generate_report=not args.skip_report
    )

    session = Session(cfg, session_cfg)
    session.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/synthetic_dataset")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--skip_report", action="store_true")
    args = parser.parse_args()
    main(args)