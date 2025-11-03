# GNN-TAMP

Graph Neural Network for Task and Motion Planning for automated structure building with blocks.

## Task

Automate the construction of block structures (towers, pyramids, stacks) using a robot arm. Given a target configuration, the system determines the correct placement sequence and executes the motions.

## Approach

**High-level planning**: Graph convolutions learn spatial relationships between blocks and predict optimal placement order.

**Low-level execution**: KOMO (K-Order Markov Optimization) generates feasible robot trajectories for each placement.

The system replans incrementally after each block is placed, adapting to the current state.

## Architecture

```
Target Scene → Graph Representation → GNN Predictions → Placement Order → 
KOMO Motion Planning → Robot Execution → State Update → Replan → Repeat
```

## Structure

```
GNN-TAMP/
├── main.py              # Main execution script
├── model.py             # GNN model (CustomEdgeConv + GNNModel)
├── graph_processor.py  # Parse .g files, convert to graphs
├── planner.py           # Generate placement sequences
├── motion_planner.py    # KOMO trajectory optimization
├── generate_dataset.py  # Generate training data
├── trained_model.pth    # Pre-trained GNN weights
├── robot_free.g         # Robot configuration
└── target/              # Target structure files (.g format)
```

## Dependencies

- PyTorch
- torch-geometric
- robotic (rai) - for KOMO
- NetworkX
- NumPy

## Usage

### Generate Training Data

```bash
python generate_dataset.py
```

Creates diverse stacking patterns: random stacks, pyramids, and predefined patterns.

### Run Planning and Execution

```bash
python main.py
```

1. Loads trained GNN model
2. Processes target scene from `target/`
3. Predicts placement order via graph convolutions
4. For each block:
   - Plans motion trajectory with KOMO
   - Executes placement
   - Updates scene state
   - Replans remaining sequence

## Graph Representation

- **Nodes**: Blocks/objects
- **Edges**: Spatial relationships with relative positions
- **Edge direction**: Upward connections (z > 0)

The GNN uses custom edge convolution incorporating:
- Node features
- Relative positions (edge attributes)
- Directional vectors (target - source)

Output: Probability scores indicating which block to place next.

## File Format

Scene files (`.g` format):
```
object1: { X: [x, y, z, ...], shape: ssBox, size: [0.8, 0.8, 0.8, .01], color: [0.5, 0.5, 0.5]}
object2(object1): { Q: "t(dx dy dz) d(angle 0 0 1)", shape: ssBox, ...}
```

First line defines base object position. Subsequent lines define objects relative to parents.

