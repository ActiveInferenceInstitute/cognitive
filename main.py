"""
Main entry point for the ant colony simulation.
"""

import argparse
import yaml
import numpy as np
from ant_colony.visualization.renderer import SimulationRenderer
from ant_colony.agents.nestmate import Nestmate, Position, TaskType
from dataclasses import dataclass
from typing import List

# ... existing code ... 