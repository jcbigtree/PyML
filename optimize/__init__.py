"""Various methods for function optimization
"""

from .line_search import line_search
from .ecnp.de import DifferentialEvolution
from .ecnp.es import OnePlusOneES
from .ecnp.es import OnePlusLambdaES
from .ecnp.nsgaii import NsgaII

__all__ = [
    "line_search",
    "DifferentialEvolution",
    "OnePlusOneES",
    "OnePlusLambdaES",
    "NsgaII"
]