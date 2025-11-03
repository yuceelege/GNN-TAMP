# GNN-TAMP

Graph Neural Network for Task and Motion Planning.

## Overview

Uses a GNN to predict object placement order, then executes plans using KOMO motion planning.

## Structure

```
GNN-TAMP/
├── main.py              # Main execution script
├── model.py             # GNN model definitions
├── graph_processor.py    # Graph parsing and conversion
├── planner.py           # Planning logic
├── motion_planner.py    # KOMO motion planning
├── generate_dataset.py  # Dataset generation
├── trained_model.pth    # Pre-trained model weights
├── robot_free.g         # Robot configuration
└── target/              # Target scene files (.g format)
```

## Dependencies

- PyTorch
- torch-geometric
- robotic (rai)
- NetworkX
- NumPy

## Usage

### Generate Dataset

```bash
python generate_dataset.py
```

Generates training data in `dataset/` directory with multiple stacking patterns.

### Run Planning

```bash
python main.py
```

Places target configuration files in `target/` directory. The script will:
1. Load the trained model
2. Generate a placement order
3. Execute motions using KOMO
4. Replan incrementally after each placement

## File Format

Scene files use `.g` format:
```
object1: { X: [x, y, z, ...], shape: ssBox, size: [0.8, 0.8, 0.8, .01], color: [0.5, 0.5, 0.5]}
object2(object1): { Q: "t(dx dy dz) d(angle 0 0 1)", shape: ssBox, ...}
```

## Model

GNN with custom edge convolution that uses:
- Node features
- Relative positions (edge weights)
- Directional information

Outputs probability scores for next object to place.

