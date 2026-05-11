"""
Pytest configuration & shared fixtures
"""

import sys
from pathlib import Path

# Add fusion/ to sys.path so that ``import gazeshop`` and ``import apps``
# both resolve correctly when pytest is invoked from the project root.
_fusion = Path(__file__).resolve().parent / "fusion"
if str(_fusion) not in sys.path:
    sys.path.insert(0, str(_fusion))
