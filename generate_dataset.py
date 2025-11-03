"""
Dataset generation script for GNN-TAMP training data.

This script generates various patterns of object stacking configurations
to create training datasets for the graph neural network.
"""
import random
import os
from itertools import combinations, chain


# Constants
CUBE_SIZE = 0.8
DEFAULT_COLOR = [0.5, 0.5, 0.5]


def random_uniform_rounded(low, high, decimal_places=2):
    """Generate a random float rounded to specified decimal places."""
    return round(random.uniform(low, high), decimal_places)


def random_offset():
    """Generate a random offset value."""
    if random.choice([True, False]):
        return random_uniform_rounded(1, 1.5)
    else:
        return random_uniform_rounded(-1.5, -1)


def generate_unique_numbers(n, start, end):
    """Generate n unique numbers within range [start, end] with minimum spacing."""
    unique_numbers = set()
    while len(unique_numbers) < n:
        number = random.uniform(start, end)
        if all(abs(number - existing_number) >= 0.2 for existing_number in unique_numbers):
            unique_numbers.add(number)
    return list(unique_numbers)


def generate_random_rgb():
    """Generate random RGB color values."""
    return (random_uniform_rounded(0, 1) for _ in range(3))


def generate_instance(num_objects=None):
    """
    Generate random grouping instances for stacked objects.
    
    Args:
        num_objects: Number of objects (if None, randomly chosen between 2-6)
    
    Returns:
        List of tuples representing object groups
    """
    if num_objects is None:
        num_objects = random.randint(2, 6)
    
    objects = list(range(1, num_objects + 1))
    all_groupings = list(chain(*[combinations(objects, i) for i in range(1, num_objects + 1)]))
    
    sampled_groupings = []
    while objects:
        group = random.choice(all_groupings)
        sampled_groupings.append(group)
        for obj in group:
            objects.remove(obj)
            all_groupings = [g for g in all_groupings if obj not in g]
    
    return sampled_groupings


def write_object_base(file, obj_id, x_pos, y_pos, z_pos):
    """Write a base object (not stacked on another) to file."""
    line = (f'object{obj_id}: {{ X: [{x_pos}, {y_pos}, {z_pos}, 0.3, 0, 0, 0], '
            f'shape: ssBox, size: [{CUBE_SIZE}, {CUBE_SIZE}, {CUBE_SIZE}, .01], '
            f'color: [{DEFAULT_COLOR[0]}, {DEFAULT_COLOR[1]}, {DEFAULT_COLOR[2]}]}}')
    file.write(line + "\n")


def write_object_stacked(file, obj_id, base_obj_id, tx, ty, tz, angle=0):
    """Write a stacked object (on top of another) to file."""
    line = (f'object{obj_id}(object{base_obj_id}): {{ '
            f'Q: "t({tx} {ty} {tz}) d({angle} 0 0 1)", '
            f'shape: ssBox, size: [{CUBE_SIZE}, {CUBE_SIZE}, {CUBE_SIZE}, .01], '
            f'color: [{DEFAULT_COLOR[0]}, {DEFAULT_COLOR[1]}, {DEFAULT_COLOR[2]}]}}')
    file.write(line + "\n")


def generate_random_stacks_pattern(output_dir, num_files, start_index=1, min_objects=2, max_objects=6):
    """
    Generate random stack patterns with varying numbers of objects.
    
    This creates configurations where objects are randomly grouped into stacks.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        instances = generate_instance(random.randint(min_objects, max_objects))
        file_name = os.path.join(output_dir, f"sample_{start_index + i}.g")
        
        x_offsets = generate_unique_numbers(len(instances), 3, -3)
        y_offsets = generate_unique_numbers(len(instances), 3, -3)
        
        with open(file_name, "w") as file:
            for idx, instance in enumerate(instances):
                k = len(instance)
                x_pos = x_offsets[idx]
                y_pos = y_offsets[idx]
                
                # Write base object
                write_object_base(file, instance[0], x_pos, y_pos, CUBE_SIZE / 2)
                
                # Write stacked objects
                for j in range(1, k):
                    angle_random = random.randint(0, 180)
                    tx = random.uniform(0, 0.1)
                    ty = random.uniform(0, 0.1)
                    tz = CUBE_SIZE
                    write_object_stacked(
                        file, instance[j], instance[j - 1], tx, ty, tz, angle_random
                    )
    
    print(f"Generated {num_files} random stack pattern files in {output_dir}")


def generate_pyramid_pattern(output_dir, num_files, start_index=1, min_objects=2, max_objects=5):
    """
    Generate pyramid-shaped stacking patterns.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        file_name = os.path.join(output_dir, f"sample_{start_index + i}.g")
        num_objects = random.randint(min_objects, max_objects)
        object_count = 0
        
        with open(file_name, "w") as file:
            layer = 0
            current_layer_objects = num_objects
            
            while current_layer_objects > 0:
                # Calculate x positions for this layer
                x_offsets = [j * 0.8 - (2 - 0.4 * layer) for j in range(1, current_layer_objects + 1)]
                for j in range(1, len(x_offsets)):
                    x_offsets[j] = x_offsets[j - 1] + random.random() * 0.2 + 0.8
                
                z_positions = [0.4 + 0.8 * layer] * current_layer_objects
                x_offsets = [round(x, 3) for x in x_offsets]
                z_positions = [round(z, 3) for z in z_positions]
                
                # Write objects for this layer
                for j in range(current_layer_objects):
                    object_count += 1
                    write_object_base(file, object_count, x_offsets[j], 0, z_positions[j])
                
                current_layer_objects -= 1
                layer += 1
    
    print(f"Generated {num_files} pyramid pattern files in {output_dir}")


def generate_pattern1(output_dir, num_files, start_index=1):
    """
    Pattern 1: Two base objects, two stacked on top, one on top of those.
    6 objects total.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        file_name = os.path.join(output_dir, f"sample_{start_index + i}.g")
        x_offsets = random.uniform(3, -3)
        y_offsets = random.uniform(3, -3)
        
        with open(file_name, "w") as file:
            # Base objects
            write_object_base(file, 1, x_offsets, y_offsets, 0)
            write_object_base(file, 2, x_offsets + random.uniform(0.5, 0) + CUBE_SIZE, y_offsets, 0)
            write_object_base(file, 3, x_offsets + random.uniform(0.5, 0) + CUBE_SIZE, y_offsets, 0)
            
            # Stack on object 1
            tx = CUBE_SIZE / 2
            tz = CUBE_SIZE + random.uniform(0.5, 0)
            write_object_stacked(file, 4, 1, tx, 0, tz, 0)
            
            # Stack on object 2
            write_object_stacked(file, 5, 2, tx, 0, tz, 0)
            
            # Stack on object 4
            write_object_stacked(file, 6, 4, tx, 0, tz, 0)
    
    print(f"Generated {num_files} pattern1 files in {output_dir}")


def generate_pattern2(output_dir, num_files, start_index=1):
    """
    Pattern 2: Two base objects with two objects stacked on top.
    4 objects total.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        file_name = os.path.join(output_dir, f"sample_{start_index + i}.g")
        x_offsets = random.uniform(3, -3)
        y_offsets = random.uniform(3, -3)
        
        with open(file_name, "w") as file:
            # Base objects
            write_object_base(file, 1, x_offsets, y_offsets, 0)
            write_object_base(file, 2, x_offsets + random.uniform(0.5, 0) + CUBE_SIZE, y_offsets, 0)
            
            # Stack on object 1
            tx = -CUBE_SIZE / 2
            tz = CUBE_SIZE + random.uniform(0.5, 0)
            write_object_stacked(file, 3, 1, tx, 0, tz, 0)
            
            # Stack on object 2
            write_object_stacked(file, 4, 2, tx, 0, tz, 0)
    
    print(f"Generated {num_files} pattern2 files in {output_dir}")


def generate_pattern3(output_dir, num_files, start_index=1):
    """
    Pattern 3: Two base objects, two stacked on top, one more on one stack.
    5 objects total.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        file_name = os.path.join(output_dir, f"sample_{start_index + i}.g")
        x_offsets = random.uniform(3, -3)
        y_offsets = random.uniform(3, -3)
        
        with open(file_name, "w") as file:
            # Base objects
            write_object_base(file, 1, x_offsets, y_offsets, 0)
            write_object_base(file, 2, x_offsets + random.uniform(0.5, 0) + CUBE_SIZE, y_offsets, 0)
            
            # Stack on object 1
            tx = CUBE_SIZE / 2
            tz = CUBE_SIZE + random.uniform(0.5, 0)
            write_object_stacked(file, 3, 1, tx, 0, tz, 0)
            
            # Stack on object 2
            write_object_stacked(file, 4, 2, tx, 0, tz, 0)
            
            # Stack on object 3
            write_object_stacked(file, 5, 3, tx, 0, tz, 0)
    
    print(f"Generated {num_files} pattern3 files in {output_dir}")


def generate_pattern4(output_dir, num_files, start_index=1):
    """
    Pattern 4: Two base objects, one stacked on first, one on top of that.
    4 objects total.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        file_name = os.path.join(output_dir, f"sample_{start_index + i}.g")
        x_offsets = random.uniform(3, -3)
        y_offsets = random.uniform(3, -3)
        
        with open(file_name, "w") as file:
            # Base objects
            write_object_base(file, 1, x_offsets, y_offsets, 0)
            write_object_base(file, 2, x_offsets + random.uniform(0.5, 0) + CUBE_SIZE, y_offsets, 0)
            
            # Stack on object 1
            tx = CUBE_SIZE / 2
            tz = CUBE_SIZE + random.uniform(0.5, 0)
            write_object_stacked(file, 3, 1, tx, 0, tz, 0)
            
            # Stack on object 3
            write_object_stacked(file, 4, 3, tx, 0, tz, 0)
    
    print(f"Generated {num_files} pattern4 files in {output_dir}")


def main():
    """Main function to generate all dataset patterns."""
    # Configuration
    OUTPUT_BASE_DIR = "dataset"
    
    # Pattern generation parameters
    patterns_config = [
        {
            "name": "random_stacks",
            "func": generate_random_stacks_pattern,
            "output_dir": os.path.join(OUTPUT_BASE_DIR, "random_stacks"),
            "num_files": 20000,
            "start_index": 1,
            "kwargs": {"min_objects": 2, "max_objects": 6}
        },
        {
            "name": "pyramid",
            "func": generate_pyramid_pattern,
            "output_dir": os.path.join(OUTPUT_BASE_DIR, "pyramid"),
            "num_files": 10000,
            "start_index": 1,
            "kwargs": {"min_objects": 2, "max_objects": 5}
        },
        {
            "name": "pattern1",
            "func": generate_pattern1,
            "output_dir": os.path.join(OUTPUT_BASE_DIR, "pattern1"),
            "num_files": 1000,
            "start_index": 20001,
            "kwargs": {}
        },
        {
            "name": "pattern2",
            "func": generate_pattern2,
            "output_dir": os.path.join(OUTPUT_BASE_DIR, "pattern2"),
            "num_files": 1000,
            "start_index": 21001,
            "kwargs": {}
        },
        {
            "name": "pattern3",
            "func": generate_pattern3,
            "output_dir": os.path.join(OUTPUT_BASE_DIR, "pattern3"),
            "num_files": 1000,
            "start_index": 22001,
            "kwargs": {}
        },
        {
            "name": "pattern4",
            "func": generate_pattern4,
            "output_dir": os.path.join(OUTPUT_BASE_DIR, "pattern4"),
            "num_files": 1000,
            "start_index": 23001,
            "kwargs": {}
        },
    ]
    
    print("Starting dataset generation...")
    print(f"Total patterns to generate: {len(patterns_config)}")
    print("-" * 50)
    
    for config in patterns_config:
        print(f"\nGenerating {config['name']}...")
        config["func"](
            config["output_dir"],
            config["num_files"],
            config["start_index"],
            **config["kwargs"]
        )
    
    print("\n" + "=" * 50)
    print("Dataset generation completed!")
    total_files = sum(c["num_files"] for c in patterns_config)
    print(f"Total files generated: {total_files}")


if __name__ == "__main__":
    main()

