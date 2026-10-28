"""
Milky Way (Galactic) dust correction

Correction (Cardelli, Clayton & Mathis 1989, R_V = 3.1, at OBSERVED
wavelengths, E(B-V) from the IRSA table, Schlafly & Finkbeiner 2011):
"""

# Dependencies
import os
import shutil
import numpy as np
import pandas as pd

LN10 = np.log(10.0)