"""
SparkNet Explorer - Curiosity-Driven Neural Network

A neural network architecture that combines:
- Extrinsic rewards (task performance)
- Intrinsic rewards (curiosity/novelty)
- Homeostatic stability (parameter health)

This creates emergent exploratory behavior through the drive to resolve uncertainty.
"""

from .sparknet_explorer import SparkNetExplorer

__version__ = "0.1.0"
__all__ = ["SparkNetExplorer"]
