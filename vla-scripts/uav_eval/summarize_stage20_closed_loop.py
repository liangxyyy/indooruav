#!/usr/bin/env python3
"""Summarize matched Stage20 condition-selection and fixed-branch runs."""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def distance(left, right):
    return math.sqrt(sum((float(left[i]) - float(right[i])) ** 2 for i in range(3)))


def mean(values):
    return statistics.fmean(values) if values else float("nan")


def load_run(directory):
    directory = Path(directory)
    result_path = directory / "final_results.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing completed-run summary: {result_path}")
    run_results = json.loads(result_path.read_text())

    episodes = {}
    for path in directory.glob("*.json"):
        if path.name == "final_results.json":
            continue
        data = json.loads(path.read_text())
        key = data["episode_key"]
        metadata = run_results.get(key, {})
        trajectory = data["trajectory"]
        target = data["end_coords"]
        position_errors = [distance(pose, target) for pose in trajectory]
        path_length = sum(
            distance(trajectory[index - 1], trajectory[index])
            for index in range(1, len(trajectory))
        )
        start_error = distance(data["start_coords"], target)

        future_decisions = [
            decision for decision in data.get("decisions", [])
            if decision.get("plan_step", 0) > 0
        ]
        margins = []
        for decision in future_decisions:
            similarities = decision.get("condition_similarities")
            if similarities and len(similarities) >= 2:
                ordered = sorted(map(float, similarities), reverse=True)
                margins.append(ordered[0] - ordered[1])

        episodes[key] = {
            "success": bool(data["success"]),
            "steps": int(data["steps"]),
            "difficulty": metadata.get("difficulty", "unknown"),
            "action_type": metadata.get("action_type", []),
            "start_position_error": start_error,
            "final_position_error": float(data["final_position_error"]),
            "min_position_error": min(position_errors),
            "final_yaw_error": float(data["final_yaw_error"]),
            "position_progress": start_error - float(data["final_position_error"]),
            "normalized_position_progress": (
                (start_error - float(data["final_position_error"])) / start_error
                if start_error > 0.0 else 0.0
            ),
            "path_length": path_length,
            "future_branches": [
                int(decision["selected_branch"])
                for decision in future_decisions
                if decision.get("selected_branch") is not None
            ],
            "condition_margins": margins,
        }
    if set(episodes) != set(run_results):
        missing = sorted(set(run_results) - set(episodes))
        raise RuntimeError(f"Missing trajectory JSON for episodes: {missing}")
    return episodes


def summarize(label, episodes):
    rows = list(episodes.values())
    return {
        "label": label,
        "episodes": len(rows),
        "success_rate": mean([float(row["success"]) for row in rows]),
        "final_position_error": mean([row["final_position_error"] for row in rows]),
        "min_position_error": mean([row["min_position_error"] for row in rows]),
        "final_yaw_error": mean([row["final_yaw_error"] for row in rows]),
        "position_progress": mean([row["position_progress"] for row in rows]),
        "normalized_position_progress": mean(
            [row["normalized_position_progress"] for row in rows]
        ),
        "path_length": mean([row["path_length"] for row in rows]),
    }


def print_summary(summary):
    print(
        f"{summary['label']}: n={summary['episodes']}, "
        f"success={summary['success_rate']:.1%}, "
        f"final_pos={summary['final_position_error']:.4f} m, "
        f"min_pos={summary['min_position_error']:.4f} m, "
        f"final_yaw={summary['final_yaw_error']:.4f} rad, "
        f"progress={summary['position_progress']:.4f} m "
        f"({summary['normalized_position_progress']:.1%}), "
        f"path={summary['path_length']:.4f} m"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition_dir")
    parser.add_argument("fixed_dir")
    args = parser.parse_args()

    condition = load_run(args.condition_dir)
    fixed = load_run(args.fixed_dir)
    if set(condition) != set(fixed):
        raise RuntimeError("Runs do not contain the same episode keys")

    print_summary(summarize("condition", condition))
    print_summary(summarize("fixed", fixed))

    keys = sorted(condition)
    position_gains = [
        fixed[key]["final_position_error"] - condition[key]["final_position_error"]
        for key in keys
    ]
    minimum_position_gains = [
        fixed[key]["min_position_error"] - condition[key]["min_position_error"]
        for key in keys
    ]
    yaw_gains = [
        fixed[key]["final_yaw_error"] - condition[key]["final_yaw_error"]
        for key in keys
    ]
    condition_wins = sum(gain > 0.0 for gain in position_gains)
    fixed_wins = sum(gain < 0.0 for gain in position_gains)
    ties = len(keys) - condition_wins - fixed_wins
    print(
        f"paired: condition position wins/ties/fixed wins="
        f"{condition_wins}/{ties}/{fixed_wins}, "
        f"mean position gain={mean(position_gains):+.4f} m, "
        f"mean minimum-position gain={mean(minimum_position_gains):+.4f} m, "
        f"mean yaw gain={mean(yaw_gains):+.4f} rad"
    )

    by_difficulty = defaultdict(list)
    for key in keys:
        by_difficulty[condition[key]["difficulty"]].append(key)
    for difficulty, group in sorted(by_difficulty.items()):
        gains = [
            fixed[key]["final_position_error"] - condition[key]["final_position_error"]
            for key in group
        ]
        print(
            f"difficulty={difficulty}: n={len(group)}, "
            f"condition/fixed success="
            f"{mean([float(condition[key]['success']) for key in group]):.1%}/"
            f"{mean([float(fixed[key]['success']) for key in group]):.1%}, "
            f"mean position gain={mean(gains):+.4f} m"
        )

    branches = Counter(
        branch for row in condition.values() for branch in row["future_branches"]
    )
    margins = [margin for row in condition.values() for margin in row["condition_margins"]]
    print(
        f"condition future branches={dict(sorted(branches.items()))}, "
        f"mean top1-top2 margin={mean(margins):.6f}"
    )


if __name__ == "__main__":
    main()
