
from .domain import ODD, DomainSpec
class Environment:
    def __init__(self, name: str):
        self.name = name
        self.data = self.get_fixed_data()
        self.index = 0

    def get_domain(self) -> DomainSpec:
        raise NotImplementedError

    def step(self, timestamp) -> ODD:
        raise NotImplementedError
    
    def load(self, source: str):
        raise NotImplementedError