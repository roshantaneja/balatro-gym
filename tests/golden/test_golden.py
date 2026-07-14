"""Golden scoring tests driven by YAML cases.

Each case is a full setup plus hand-derived expected numbers (derivation in
the case's `comment`). Any regression fails with the rendered audit table.
"""

from pathlib import Path

import pytest
import yaml

from balatro_gym.engine import HandType, score_play

from tests.helpers import cards, joker, make_run

CASES_DIR = Path(__file__).parent / "cases"
CASES = sorted(CASES_DIR.glob("*.yaml"))


def load_case(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("case_path", CASES, ids=lambda p: p.stem)
def test_golden(case_path: Path):
    case = load_case(case_path)

    jokers = []
    for spec in case.get("jokers", []):
        if isinstance(spec, str):
            jokers.append(joker(spec))
        else:
            jokers.append(
                joker(spec["id"], edition=spec.get("edition", "none"), state=spec.get("state"))
            )

    hand_levels = {
        HandType[name]: lvl for name, lvl in case.get("hand_levels", {}).items()
    }
    run = make_run(jokers=jokers, hand_levels=hand_levels, seed=case.get("seed", "golden"))

    played = cards(*case["played"])
    held = cards(*case.get("held", []))
    res = score_play(run, played, held=held)

    exp = case["expected"]
    failures = []
    if "hand_type" in exp and res.hand_result.hand_type != HandType[exp["hand_type"]]:
        failures.append(f"hand_type: got {res.hand_result.hand_type.name}, want {exp['hand_type']}")
    if "chips" in exp and res.chips != exp["chips"]:
        failures.append(f"chips: got {res.chips}, want {exp['chips']}")
    if "mult" in exp and res.mult != exp["mult"]:
        failures.append(f"mult: got {res.mult}, want {exp['mult']}")
    if "score" in exp and res.score != exp["score"]:
        failures.append(f"score: got {res.score}, want {exp['score']}")
    if "money_delta" in exp and res.money_delta != exp["money_delta"]:
        failures.append(f"money_delta: got {res.money_delta}, want {exp['money_delta']}")

    if failures:
        pytest.fail(
            f"{case['name']}: " + "; ".join(failures) + "\n\nAudit log:\n" + res.explain()
        )
