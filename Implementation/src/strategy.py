


from .domain import DomainSpec, Parameter, ODD, Configuration,ConstrainedDomain

class Strategy:
    last_opt_status: bool
    domain_spec: DomainSpec
    """Base class for different strategies."""


    """ A strategy defines how a particular approach achieves its optimization goals.
    It encapsulates the logic for selecting design parameters, applying constraints,
    and evaluating configurations within a given domain.
    """

    def __init__(self, name: str):
        self.name = name

    def execute(self, odd: ODD, ds: ConstrainedDomain):
        raise NotImplementedError("Execute method must be implemented by subclasses.")


    def print_info(self):
        print(self.__str__())
    
    def __str__(self):
        raise NotImplementedError(" String method must be implemented by subclasses.")

    def get_last_optimization_status(self):
        return self.last_opt_status

    def get_explored_volume(self):
        raise NotImplementedError("get_explored_volume method must be implemented by subclasses.")

    def load(self, source: str):
        raise NotImplementedError("load method must be implemented by subclasses.")