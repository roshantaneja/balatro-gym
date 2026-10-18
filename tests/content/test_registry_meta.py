"""Meta-test: every registered joker id must be exercised by at least one
test under tests/content/ (grep of test sources). Enforces the 1:1
content-to-test coverage rule for all future jokers."""

from __future__ import annotations

from pathlib import Path

import tests.helpers  # noqa: F401  (loads content registries)
from balatro_gym.engine.effects import JOKERS

CONTENT_TEST_DIR = Path(__file__).parent


def test_every_joker_id_appears_in_content_tests():
    sources = []
    for path in sorted(CONTENT_TEST_DIR.rglob("test_*.py")):
        if path.name == "test_registry_meta.py":
            continue
        sources.append(path.read_text())
    blob = "\n".join(sources)

    missing = [jid for jid in sorted(JOKERS) if jid not in blob]
    assert not missing, (
        "registered jokers with no test mention under tests/content/: "
        f"{missing} — add at least one test that exercises each."
    )
