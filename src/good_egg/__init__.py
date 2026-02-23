"""Good Egg - GitHub contributor trust scoring."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from good_egg.config import GoodEggConfig, V2Config
from good_egg.exceptions import GoodEggError
from good_egg.models import TrustLevel, TrustScore
from good_egg.scorer import TrustScorer, score_pr_author

try:
    __version__ = version("good-egg")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "GoodEggConfig",
    "GoodEggError",
    "TrustLevel",
    "TrustScore",
    "TrustScorer",
    "V2Config",
    "__version__",
    "score_pr_author",
]
