
import numpy as np
from typing import NamedTuple
from rsarl.networks import Network

class Observation(NamedTuple):
    request: NamedTuple
    net: Network
