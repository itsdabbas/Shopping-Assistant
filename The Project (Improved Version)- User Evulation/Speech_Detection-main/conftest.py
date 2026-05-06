"""
Pytest configuration & shared fixtures
"""

import sys
from pathlib import Path

# Ensure the project root (Speech_Detection-main/) is on sys.path so that
# ``import gazeshop`` and ``import apps`` both work when pytest is invoked
# from any working directory.
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
