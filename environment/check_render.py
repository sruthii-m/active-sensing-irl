"""Step through ForageEnv and save the full-grid render next to the agent's
partial POV for the same episode/seed, to confirm they show the same
underlying layout before any learning code is written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .forage_env import make_forage_env

OUT_DIR = Path(__file__).parent / "render_check_output"


def _side_by_side(full: np.ndarray, partial: np.ndarray) -> Image.Image:
    h = max(full.shape[0], partial.shape[0])
    full_img = Image.fromarray(full)
    partial_img = Image.fromarray(partial)
    canvas = Image.new("RGB", (full_img.width + partial_img.width + 10, h), "white")
    canvas.paste(full_img, (0, 0))
    canvas.paste(partial_img, (full_img.width + 10, 0))
    return canvas


def main(num_steps: int = 5, seed: int = 0) -> None:
    env = make_forage_env(render_mode="rgb_array")
    env.reset(seed=seed)
    OUT_DIR.mkdir(exist_ok=True)

    for step in range(num_steps + 1):
        full = env.get_frame(agent_pov=False, highlight=True)
        partial = env.get_frame(agent_pov=True)
        _side_by_side(full, partial).save(OUT_DIR / f"step_{step:02d}.png")
        if step < num_steps:
            action = env.action_space.sample()
            env.step(action)

    print(f"Wrote {num_steps + 1} side-by-side renders to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(num_steps=args.num_steps, seed=args.seed)
