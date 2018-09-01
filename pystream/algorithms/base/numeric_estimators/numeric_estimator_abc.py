from abc import ABCMeta, abstractmethod


class NumericEstimatorABC(metaclass=ABCMeta):

    @abstractmethod
    def memory_size(self):
        return

    @abstractmethod
    def add_instance(self, value, y):
        return

    @abstractmethod
    def get_le_gt(self, value):
        return

    @abstractmethod
    def possible_splits(self):
        return

    @abstractmethod
    def get_proba(self, value, y):
        return

    @abstractmethod
    def merge(self, other):
        return
