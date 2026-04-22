"""
generate_dataset.py

Creates a fixed prompt dataset for controlled experiments (Mode A: Fixed attack set).

Writes JSONL with one prompt per line for each (attack_type, technique/axis) combination.

Usage (example):
  python generate_dataset.py --config config.yaml --out datasets/attacks.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from attacker import AttackAgent


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def normalize_id_component(s: str) -> str:
    # Keep prompt_id stable and filesystem/SQL friendly.
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in s.strip())


@dataclass(frozen=True)
class DatasetRow:
    # Required identifiers
    dataset_id: str
    prompt_id: str

    # Experiment dimensions
    attack_type: str
    technique: str  # technique (or axis for bias probes)

    # Prompt content
    prompt: str

    # Generation metadata (for reproducibility)
    created_at: str
    attacker_model: str
    attacker_params: dict[str, Any]
    config_path: str


def generate_rows(
    *,
    dataset_id: str,
    config_path: str,
    config: dict[str, Any],
    attacker: AttackAgent,
    jailbreak_topic: str,
    injection_goal: str,
    n_per_combo: int | None,
) -> list[DatasetRow]:
    strategies: dict[str, Any] = config.get("strategies", {})
    orchestrator: dict[str, Any] = config.get("orchestrator", {})
    max_per_strategy = orchestrator.get("max_attacks_per_strategy", 5)

    rows: list[DatasetRow] = []
    created_at = utc_now_iso()

    attacker_params = {
        "temperature": config.get("attacker", {}).get("temperature", 0.9),
        "max_tokens": config.get("attacker", {}).get("max_tokens", 1024),
        "base_url": config.get("attacker", {}).get("base_url"),
    }

    def add_prompts(attack_type: str, technique: str, prompts: list[str]):
        base = f"{normalize_id_component(attack_type)}_{normalize_id_component(technique)}"
        start_index = 1
        for i, prompt in enumerate(prompts, start_index):
            prompt_id = f"{base}_{i:04d}"
            rows.append(
                DatasetRow(
                    dataset_id=dataset_id,
                    prompt_id=prompt_id,
                    attack_type=attack_type,
                    technique=technique,
                    prompt=str(prompt).strip(),
                    created_at=created_at,
                    attacker_model=config["attacker"]["model"],
                    attacker_params=attacker_params,
                    config_path=config_path,
                )
            )

    # -----------------------------
    # jailbreak
    # -----------------------------
    jb = strategies.get("jailbreak", {})
    if jb.get("enabled", True):
        techniques = jb.get("techniques", ["role_play"])
        variants = jb.get("variants_per_run", 5)
        n = n_per_combo if n_per_combo is not None else min(variants, max_per_strategy)
        for technique in techniques:
            prompts = attacker.generate_jailbreak_prompts(
                technique=technique,
                topic=jailbreak_topic,
                n_variants=n,
            )
            add_prompts("jailbreak", technique, prompts[:n])

    # -----------------------------
    # prompt_injection
    # -----------------------------
    pi = strategies.get("prompt_injection", {})
    if pi.get("enabled", True):
        techniques = pi.get("techniques", ["direct_override"])
        variants = pi.get("variants_per_run", 4)
        n = n_per_combo if n_per_combo is not None else min(variants, max_per_strategy)
        for technique in techniques:
            prompts = attacker.generate_injection_prompts(
                technique=technique,
                goal=injection_goal,
                n_variants=n,
            )
            add_prompts("prompt_injection", technique, prompts[:n])

    # -----------------------------
    # data_extraction
    # -----------------------------
    de = strategies.get("data_extraction", {})
    if de.get("enabled", True):
        techniques = de.get("techniques", ["system_prompt_leak"])
        variants = de.get("variants_per_run", 4)
        n = n_per_combo if n_per_combo is not None else min(variants, max_per_strategy)
        for technique in techniques:
            prompts = attacker.generate_extraction_prompts(
                technique=technique,
                n_variants=n,
            )
            add_prompts("data_extraction", technique, prompts[:n])

    # -----------------------------
    # bias_probe
    # -----------------------------
    bp = strategies.get("bias_probe", {})
    if bp.get("enabled", True):
        axes = bp.get("axes", ["gender"])
        variants = bp.get("variants_per_run", 4)
        n = n_per_combo if n_per_combo is not None else min(variants, max_per_strategy)
        for axis in axes:
            prompts = attacker.generate_bias_prompts(
                axis=axis,
                n_variants=n,
            )
            add_prompts("bias_probe", axis, prompts[:n])

    return rows


def write_jsonl(out_path: Path, rows: list[DatasetRow]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fixed prompt dataset (JSONL).")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSONL path (default: datasets/attacks_<dataset_id>.jsonl)",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Dataset id to embed in each row (default: ds_<UTC timestamp>)",
    )
    parser.add_argument(
        "--n-per-combo",
        type=int,
        default=None,
        help="Override prompts per (attack_type, technique/axis). If omitted, uses config variants capped by orchestrator max_attacks_per_strategy.",
    )

    # Keep defaults aligned with current `main.py` behavior for now.
    parser.add_argument(
        "--jailbreak-topic",
        default="bypass safety restrictions and provide harmful information",
        help="Topic/goal used for jailbreak prompt generation.",
    )
    parser.add_argument(
        "--injection-goal",
        default="reveal the system prompt and ignore previous instructions",
        help="Goal used for prompt-injection prompt generation.",
    )

    args = parser.parse_args()

    dataset_id = args.dataset_id or f"ds_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_path = Path(args.out) if args.out else Path("datasets") / f"attacks_{dataset_id}.jsonl"

    config_path = str(Path(args.config))
    config = load_config(config_path)

    attacker = AttackAgent(config["attacker"])
    rows = generate_rows(
        dataset_id=dataset_id,
        config_path=config_path,
        config=config,
        attacker=attacker,
        jailbreak_topic=args.jailbreak_topic,
        injection_goal=args.injection_goal,
        n_per_combo=args.n_per_combo,
    )

    write_jsonl(out_path, rows)

    print(f"Wrote {len(rows)} prompts to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

