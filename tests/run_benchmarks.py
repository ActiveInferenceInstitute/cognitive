import os
import sys
import json
import logging
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Configure plotting style
sns.set_style('whitegrid')

from Things.Generic_Thing.core import GenericThing
from Things.Generic_Thing.message_passing import Message, MessagePassing
from Things.Generic_Thing.markov_blanket import MarkovBlanket
from Things.Generic_Thing.free_energy import FreeEnergy
from Things.Generic_Thing.inference import FederatedInference

# ... existing code ... 