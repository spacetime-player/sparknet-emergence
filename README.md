# sparknet-emergence

Exploring self-organization and emergent behavior in neural networks.

## Most uptodate version is sparknet_alpha_v1.py

### Latest Updates (v1)
- **Interactive Parameter Editor**: Real-time editing of model parameters with visual feedback (blinking cursor, color highlighting)
- **Play/Pause/Stop Controls**: Full simulation control - parameters editable only when paused
- **Auto GPU/CPU Detection**: Automatically uses GPU if available, falls back to CPU
- **Parameter Validation**: Prevents invalid configurations (e.g., more sparks than neurons)

## Current Focus
- Investigating how architectural constraints influence internal representations
- Testing variations of recurrent network structures with memory mechanisms
- Researching self-organization and emergence in artificial systems
- Visual analysis of network geometry and internal dynamics

## Core Concept: Spark-Based Learning

The network uses mobile "sparks" that traverse neurons, strengthening connections as they move. Key mechanisms:

- **Sparks**: Mobile agents that activate neurons and strengthen connections along their paths
- **Memory Field**: Long-term bias system that influences spark movement - frequently visited areas become more attractive
- **Adaptive Movement**: Sparks choose paths based on connection weights + memory traces + occasional random exploration
- **Hebbian-like Learning**: Connections strengthen when used, with global decay to prevent saturation

## Repository Structure
This repository contains multiple version files implementing different architectural ideas and updates. All versions are preserved for research and learning purposes - each explores a specific hypothesis about network structure and self-organization.

## Key Learning: Sequential Bias Problem
Early versions showed a persistent **diagonal line in the weight matrix heatmap** - indicating sequential bias. The system was learning simple ordered patterns rather than complex structure. This revealed that achieving true emergence requires careful rule design and constraints. 

**This "failure" was actually progress** - each unsuccessful pattern provides higher resolution on what's blocking emergence. The diagonal artifact showed the system taking the path of least resistance rather than developing interesting structure.

## Visualization Approach
All experiments emphasize visual representation to observe the geometry of internal workings:

- **Neuron Activations**: Heatmaps showing firing patterns as sparks traverse the network
- **Weight Matrix**: Connection strength distributions showing learned pathways (diagonal = sequential bias, dispersed = emergent structure)
- **Memory Field**: Accumulated usage traces that bias future spark movement
- **2D Graph Layout (force physics)**: Neurons positioned by weight matrix relationships - stronger connections = closer proximity. Uses spring physics simulation.

![Example visualization](screenshot.png)

## Observations
The 2D graph reveals interesting dynamics: during training, the system behaves like fluid dynamics. When reaching equilibrium, neuron distributions resemble gas molecules in a container - spreading evenly. 

So far, no striking emergent structures have appeared, but this fluid-like behavior during collapse to equilibrium is intriguing. It suggests networks might follow entropy laws similar to gases when settling into stable states. Curious to see if more complex systems exhibit similar thermodynamic-like properties.

## Technical Stack
- PyTorch with automatic GPU/CPU detection
- Custom architecture implementations
- Matplotlib for real-time visualization with interactive controls
- Force-directed graph layouts

## Research Direction
This work investigates self-organization in neural networks - how structure emerges from randomness at small scale. Long-term aim is to understand internal representations in larger models and contribute to research on emergence, self-awareness, and the architectural requirements for consciousness in AI systems.

## Status
Early-stage research. Iterating on architectural designs (memory systems, spark movement strategies, learning rules) and visualization methods to understand what structures enable emergent capabilities.