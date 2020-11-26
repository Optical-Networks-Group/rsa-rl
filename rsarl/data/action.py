
from typing import NamedTuple

class Action(NamedTuple):
    path: list
    slot_idx: int
    n_slot: int
    duration: float
