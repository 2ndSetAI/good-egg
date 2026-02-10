"""Good Egg - GitHub contributor trust scoring."""

from __future__ import annotations

from importlib.metadata import version

from good_egg.config import GoodEggConfig
from good_egg.exceptions import GoodEggError
from good_egg.models import TrustLevel, TrustScore
from good_egg.scorer import TrustScorer, score_pr_author

__version__ = version("good-egg")

__all__ = [
    "GoodEggConfig",
    "GoodEggError",
    "TrustLevel",
    "TrustScore",
    "TrustScorer",
    "__version__",
    "score_pr_author",
]
