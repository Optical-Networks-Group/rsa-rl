
from typing import NamedTuple

class Request(NamedTuple):
    source: int
    destination: int
    bandwidth: int
    duration: float
    