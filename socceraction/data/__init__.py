"""Implements serializers for the event data of various providers."""

__all__ = [
    "opta",
    "statsbomb",
    "wyscout",
    "bepro"
]

from . import opta, statsbomb, wyscout, bepro
