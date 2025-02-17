"""
Main simulation coordinator for ant colony simulation.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

from ant_colony.environment import World
from ant_colony.colony import Colony
from ant_colony.visualization import ColonyVisualizer
from ant_colony.utils.data_collection import DataCollector

# ... existing code ... 