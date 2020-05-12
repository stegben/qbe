import abc
from typing import Mapping

from custom_types import VoiceType, VoiceIndex, QueryResult



class QBE(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def build(cls, voices: Mapping[VoiceIndex, VoiceType]) -> 'QBE':
        pass

    @abc.abstractmethod
    def query(self, voice: VoiceType) -> QueryResult:
        pass


class RandomSelect(QBE):
    pass


class ExactAllignmentSearch(QBE):
    pass


class RAILS(QBE):
    pass
