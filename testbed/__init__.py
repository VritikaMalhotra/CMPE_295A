"""CMPE 295A testbed (M1 walking skeleton)."""
import os
import sys

# testbed/grammar.py imports the vendored llama.cpp converter by module name
_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spikes", "vendor")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, os.path.abspath(_VENDOR))
