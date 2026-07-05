import argparse
import json
import random
from typing import List, Dict, Any


Record = Dict[str, Any]


def load_dataset(path: str) -> List[Record]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset invalide dans {path}: JSON doit etre une liste")
    return data


def sample_records(data: List[Record], count: int, allow_oversample: bool) -> List[Record]:
    if count <= 0:
        return []
    if not allow_oversample and count > len(data):
        raise ValueError(
            f"Impossible de prendre {count} elements sans oversampling (taille={len(data)})"
        )
    if allow_oversample and count > len(data):
        return random.choices(data, k=count)
    return random.sample(data, k=count)


def compute_targets(
    quality_size: int,
    explore_size: int,
    quality_ratio: float,
    total: int | None,
    allow_oversample: bool,
) -> tuple[int, int, int]:
    if not (0.0 <= quality_ratio <= 1.0):
        raise ValueError("quality_ratio doit etre entre 0.0 et 1.0")

    if total is not None:
        if total <= 0:
            raise ValueError("total doit etre > 0")
        q_target = int(round(total * quality_ratio))
        e_target = total - q_target

        if not allow_oversample:
            if q_target > quality_size:
                raise ValueError(
                    f"quality trop petit: besoin {q_target}, disponible {quality_size}"
                )
            if e_target > explore_size:
                raise ValueError(
                    f"explore trop petit: besoin {e_target}, disponible {explore_size}"
                )
        return total, q_target, e_target

    if quality_ratio == 1.0:
        max_total = quality_size
        return max_total, max_total, 0

    if quality_ratio == 0.0:
        max_total = explore_size
        return max_total, 0, max_total

    if allow_oversample:
        max_total = quality_size + explore_size
    else:
        max_total = int(min(quality_size / quality_ratio, explore_size / (1.0 - quality_ratio)))

    q_target = int(round(max_total * quality_ratio))
    e_target = max_total - q_target

    return max_total, q_target, e_target


def parse_args():
    parser = argparse.ArgumentParser(description="Fusionne dataset quality et explore")
    parser.add_argument("--quality", default="dataset_quality.json", help="Chemin dataset quality")
    parser.add_argument("--explore", default="dataset_explore.json", help="Chemin dataset explore")
    parser.add_argument("--output", default="dataset_train.json", help="Fichier de sortie")
    parser.add_argument(
        "--quality-ratio",
        type=float,
        default=0.7,
        help="Part quality dans le dataset final (0.0 a 1.0)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="Taille totale voulue. Si absent, prend le maximum possible selon le ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed aleatoire pour la reproductibilite",
    )
    parser.add_argument(
        "--allow-oversample",
        action="store_true",
        help="Autorise le tirage avec remise si un dataset est trop petit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    quality_data = load_dataset(args.quality)
    explore_data = load_dataset(args.explore)

    total, q_target, e_target = compute_targets(
        quality_size=len(quality_data),
        explore_size=len(explore_data),
        quality_ratio=args.quality_ratio,
        total=args.total,
        allow_oversample=args.allow_oversample,
    )

    merged = []
    merged.extend(sample_records(quality_data, q_target, args.allow_oversample))
    merged.extend(sample_records(explore_data, e_target, args.allow_oversample))
    random.shuffle(merged)

    with open(args.output, "w") as f:
        json.dump(merged, f)

    print("Fusion terminee")
    print(f"quality source: {len(quality_data)}")
    print(f"explore source: {len(explore_data)}")
    print(f"total output : {total}")
    print(f"quality pris : {q_target}")
    print(f"explore pris : {e_target}")
    print(f"fichier      : {args.output}")


if __name__ == "__main__":
    main()
