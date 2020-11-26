
from typing import NamedTuple

class Experience(NamedTuple):
    request_id: int
    source: int
    destination: int
    bandwidth: int
    duration: float
    path: str
    slot_index: int
    n_slot: int
    is_success: bool
    reward: float
    network: str
    slot_utilization: float

