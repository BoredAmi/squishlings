"""Squishlings: evolve soft voxel creatures that crawl, on softsim's material cards and physics.

* :mod:`squishlings.genome`: the encoding (a labelled grid + one oscillator per muscle group) and its operators.
* :mod:`squishlings.build`: genome <-> ``locomotion`` scenario (a softsim ``voxels`` body, tissue zones, muscles).
* :mod:`squishlings.evolve`: the (mu + lambda) loop behind ``squishlings evolve``, cached and resumable.
* :mod:`squishlings.locomotion`: the ``locomotion`` scenario, registered with softsim (``scenario: locomotion``).
* :mod:`squishlings.replay`: GIF and HTML replays.
"""

from importlib.resources import files

from squishlings.evolve import Evolution
from squishlings.genome import Genome, GenomeSpace

__version__ = "0.1.0.dev0"
CREATURES = files(__name__) / "creatures"

__all__ = ["CREATURES", "Evolution", "Genome", "GenomeSpace", "__version__"]
