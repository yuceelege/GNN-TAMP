"""
Main execution script for GNN-based Task and Motion Planning (TAMP).
"""
import torch
import robotic as ry
import time
from model import GNNModel
from graph_processor import (
    process_all_g_files,
    convert_to_pyg_data,
    correct_graph_edge_indices
)
from planner import create_plan
from motion_planner import init_komo, define_optimization


def optimize(obj_index, config):
    """
    Optimize motion trajectory for placing a single object.
    
    Args:
        obj_index: Index of the object to place
        config: Robot configuration
    
    Returns:
        Tuple of (path, result, komo object)
    """
    komo = ry.KOMO(config, 3, 30, 1, False)
    define_optimization(config, obj_index, komo)
    ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
    q = komo.getPath()
    return q, ret, komo


def main():
    """Main execution function."""
    # Model parameters
    in_channels = 1
    hidden_channels = 32
    out_channels = 1
    
    # Initialize robot configurations
    config = ry.Config()
    visualization_config = ry.Config()
    
    # Load and initialize model
    model = GNNModel(in_channels, hidden_channels, out_channels)
    model_path = 'trained_model.pth'
    model.load_state_dict(torch.load(model_path))
    
    # Process target configuration
    target_folder = 'target'
    target_graph, pos = process_all_g_files(target_folder)
    correct_graph_edge_indices(target_graph)
    target_pyg = [convert_to_pyg_data(graph) for graph, positions in zip(target_graph, pos)][0]
    
    # Generate initial plan
    plan = create_plan(target_folder, model, option=1)
    building_order = plan[0]
    print("Building Order:", building_order)
    
    # Initialize robot scene
    object_number = len(building_order)
    init_komo(object_number, config, pos)
    
    # Execution loop
    num_objects = len(building_order)
    objective_realized = False
    last_state = None
    states = []
    checkpoint_counter = 0
    
    while not objective_realized:
        if checkpoint_counter == 0:
            obj_index = building_order[checkpoint_counter]
            q, ret, komo = optimize(obj_index, config)
        
        if ret.feasible:
            if checkpoint_counter != 0:
                config.setFrameState(last_state)
            
            checkpoint_counter += 1
            graph = plan[1]
            
            # Replan for remaining objects
            plan = create_plan(target_folder, model, graph=graph, option=2)
            building_order = plan[0]
            print(f"Updated Building Order: {building_order}")
            
            obj_index = building_order[0]
            q, ret, komo = optimize(obj_index, config)
            waypoints = komo.getPath_qAll()
            states.append(komo.getFrameState(len(waypoints) - 1))
            last_state = komo.getFrameState(len(waypoints) - 1)
            
            # Check if all objects are placed
            if checkpoint_counter == num_objects - 1:
                init_komo(object_number, visualization_config, pos)
                for s in states:
                    visualization_config.setFrameState(s)
                    time.sleep(0.5)
                    visualization_config.view(True)
                print("Task completed successfully!")
                objective_realized = True
                break
        else:
            print("Plan infeasible. Replanning...")
            if checkpoint_counter == 0:
                building_order = create_plan(target_folder, model, option=1)[0]
            else:
                building_order = create_plan(target_folder, model, graph=graph, option=2)[0]


if __name__ == "__main__":
    main()
