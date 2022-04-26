from abc import ABC, abstractmethod


class Job(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass