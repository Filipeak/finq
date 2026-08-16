"""The skill is a contract with the model; these tests keep it honest."""
import re
from pathlib import Path

SKILL = Path(__file__).parents[1] / "skills" / "portfolio-quant" / "SKILL.md"


def test_skill_file_exists_with_frontmatter():
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert re.search(r"^name:\s*portfolio-quant\s*$", text, re.M)
    assert re.search(r"^description:\s*\S", text, re.M)


def test_skill_documents_every_public_entry_point():
    text = SKILL.read_text(encoding="utf-8")
    for symbol in ("portfolio.load", "data.prices", "data.fx", "returns.aligned",
                   "covariance.estimate", "risk.risk_contributions",
                   "risk.exceedance_correlation", "liquidity.adv",
                   "optimize.compare"):
        assert symbol in text, f"{symbol} missing from SKILL.md"


def test_skill_states_the_mandatory_interpretation_rules():
    text = SKILL.read_text(encoding="utf-8").lower()
    for rule in ("q =", "delta", "equal weight", "dispersion"):
        assert rule in text


def test_skill_warns_about_the_silent_failure_modes():
    text = SKILL.read_text(encoding="utf-8").lower()
    for pitfall in ("simple returns", "252", "forward-fill", "sklearn"):
        assert pitfall in text
