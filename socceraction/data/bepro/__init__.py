import os, sys
home_dir = os.path.expanduser("~")
target_dir = os.path.join(home_dir, "socceraction")
sys.path.insert(0, target_dir)

from .loader import BeproLoader