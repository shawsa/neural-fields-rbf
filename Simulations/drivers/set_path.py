"""
We assume that the rbf module is in the parent directory of these drivers.
We will add the parent directory to the path.
"""

from os.path import join, dirname, realpath
import sys

rbf_path = join(dirname(dirname(realpath(__file__))))
sys.path.append(rbf_path)
print(f"Added to PYTHONPATH: {rbf_path}")
