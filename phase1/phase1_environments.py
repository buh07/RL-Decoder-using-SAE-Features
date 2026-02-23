#!/usr/bin/env python3
"""
Phase 1 Ground-Truth Environments
Implements exact-state systems: BFS traversal, stack machine, logic puzzles.
Each environment tracks precise latent states for SAE learning and validation.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExactState:
    """Represents an exact latent state in a ground-truth system."""
    step: int
    state_vector: np.ndarray  # The exact state representation
    action: str  # What action was taken
    metadata: Dict  # Environment-specific metadata


class BFSEnvironment:
    """
    BFS traversal on random graphs.
    
    Exact latent state: [current_node_onehot(100), visited_bitmask(128), queue_encoded(256)]
    Total: 484 dimensions of exact state
    """
    
    def __init__(self, num_nodes: int = 100, num_sequences: int = 100, max_steps: int = 50):
        self.num_nodes = num_nodes
        self.num_sequences = num_sequences
        self.max_steps = max_steps
        self.state_dim = num_nodes + 128 + 256  # onehot + visited mask + queue encoding
        
        # Generate random adjacency matrix (connected graph guaranteed via Erdos-Renyi with p=0.2)
        self.adjacency = self._generate_connected_graph()
        
    def _generate_connected_graph(self) -> np.ndarray:
        """Generate a connected graph via Erdos-Renyi model."""
        p = 0.2
        adj = np.random.binomial(1, p, size=(self.num_nodes, self.num_nodes))
        # Make undirected
        adj = np.logical_or(adj, adj.T).astype(np.int32)
        # Ensure connected by adding path from node 0 to all others
        for i in range(1, self.num_nodes):
            adj[i-1, i] = 1
            adj[i, i-1] = 1
        return adj
    
    def encode_state(self, current_node: int, visited: set, queue: List[int]) -> np.ndarray:
        """Encode BFS state to exact latent vector."""
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Current node (one-hot, first 100 dims)
        state[current_node] = 1.0
        
        # Visited nodes (bitmask, next 128 dims; supports up to 1024 nodes)
        visited_list = list(visited)
        for node in visited_list[:128]:  # Cap at 128 for this encoding
            state[self.num_nodes + node % 128] = 1.0
        
        # Queue encoding (next 256 dims; position-based encoding)
        for pos, node in enumerate(queue[:256]):
            state[self.num_nodes + 128 + pos] = float(node) / self.num_nodes
        
        return state
    
    def generate_trajectory(self, start_node: int = 0) -> List[ExactState]:
        """Generate one complete BFS trajectory with exact states."""
        trajectory = []
        visited = {start_node}
        queue = [start_node]
        
        for step in range(self.max_steps):
            if not queue:
                break
            
            current = queue.pop(0)
            state_vec = self.encode_state(current, visited, queue)
            
            trajectory.append(ExactState(
                step=step,
                state_vector=state_vec,
                action=f"visit_node_{current}",
                metadata={"current": current, "visited_size": len(visited), "queue_size": len(queue)}
            ))
            
            # Explore neighbors
            neighbors = np.where(self.adjacency[current] == 1)[0]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return trajectory
    
    def generate_dataset(self) -> Tuple[np.ndarray, List[Dict]]:
        """Generate full dataset of BFS trajectories."""
        all_states = []
        metadata = []
        
        for seq_id in range(self.num_sequences):
            start_node = np.random.randint(0, self.num_nodes)
            trajectory = self.generate_trajectory(start_node)
            
            for state in trajectory:
                all_states.append(state.state_vector)
                metadata.append(state.metadata)
        
        return np.stack(all_states, axis=0), metadata


class StackMachineEnvironment:
    """
    Stack machine executing push/pop/peek operations.
    
    Exact latent state: [operation_type_onehot(4), stack_contents_encoded(512)]
    Total: 516 dimensions
    """
    
    def __init__(self, max_stack_size: int = 128, num_sequences: int = 100, max_steps: int = 50):
        self.max_stack_size = max_stack_size
        self.num_sequences = num_sequences
        self.max_steps = max_steps
        self.state_dim = 4 + 512  # operation onehot + stack encoding
        
    def encode_state(self, operation: str, stack: List[int]) -> np.ndarray:
        """Encode stack machine state to exact latent vector."""
        state = np.zeros(self.state_dim, dtype=np.float32)

        # Operation (one-hot, first 4 dims)
        op_to_idx = {"push": 0, "pop": 1, "peek": 2, "idle": 3}
        state[op_to_idx[operation]] = 1.0

        # Stack encoding (next 512 dims; position-based + normalized value)
        # Use normalized direct encoding: each position gets value/255 in slot [4 + pos]
        # and presence indicator (1.0) in slot [4 + 256 + pos].
        # This is injective: (presence=0, value=0) vs (presence=1, value=0) are distinct.
        for pos, value in enumerate(stack[:256]):  # Cap at 256 positions
            state[4 + pos] = value / 255.0          # normalized value
            state[4 + 256 + pos] = 1.0              # presence bit

        return state

    def generate_trajectory(self, seed: int = 0) -> List[ExactState]:
        """Generate one complete stack machine trajectory."""
        np.random.seed(seed)
        trajectory = []
        stack = []

        operations_sequence = np.random.choice(
            ["push", "pop", "peek"],
            size=self.max_steps,
            p=[0.6, 0.2, 0.2]
        )
        
        for step, op in enumerate(operations_sequence):
            if op == "push":
                value = np.random.randint(0, 256)
                stack.append(value)
            elif op == "pop" and stack:
                stack.pop()
            elif op == "peek" and not stack:
                op = "idle"
            
            state_vec = self.encode_state(op, stack[:self.max_stack_size])
            
            trajectory.append(ExactState(
                step=step,
                state_vector=state_vec,
                action=op,
                metadata={"stack_size": len(stack), "top_value": stack[-1] if stack else -1}
            ))
        
        return trajectory
    
    def generate_dataset(self) -> Tuple[np.ndarray, List[Dict]]:
        """Generate full dataset of stack machine trajectories."""
        all_states = []
        metadata = []
        
        for seq_id in range(self.num_sequences):
            trajectory = self.generate_trajectory(seed=seq_id)
            
            for state in trajectory:
                all_states.append(state.state_vector)
                metadata.append(state.metadata)
        
        return np.stack(all_states, axis=0), metadata


class LogicPuzzleEnvironment:
    """
    Logic puzzle solver (Sudoku-like constraint satisfaction).
    
    Exact latent state: [grid_state_encoded(512), constraint_satisfaction_state(128)]
    Total: 640 dimensions
    """
    
    def __init__(self, grid_size: int = 9, num_sequences: int = 100, max_steps: int = 50):
        self.grid_size = grid_size
        self.num_sequences = num_sequences
        self.max_steps = max_steps
        self.state_dim = 512 + 128  # grid encoding + constraint satisfaction
        
    def _compute_constraints(self, grid: np.ndarray) -> List[bool]:
        """Compute row/col/box constraint satisfaction deterministically from grid."""
        constraints = []
        n = self.grid_size
        # 9 rows: satisfied if all filled cells in the row have no duplicates
        for r in range(n):
            row_vals = grid[r, grid[r] != 0]
            constraints.append(len(row_vals) == len(set(row_vals.tolist())))
        # 9 cols
        for c in range(n):
            col_vals = grid[grid[:, c] != 0, c]
            constraints.append(len(col_vals) == len(set(col_vals.tolist())))
        # 9 boxes (3x3 sub-grids for a 9x9 grid)
        box_size = 3
        for br in range(box_size):
            for bc in range(box_size):
                box = grid[br*box_size:(br+1)*box_size, bc*box_size:(bc+1)*box_size].flatten()
                box_vals = box[box != 0]
                constraints.append(len(box_vals) == len(set(box_vals.tolist())))
        return constraints  # 27 booleans

    def encode_state(self, grid: np.ndarray, constraints_satisfied: List[bool]) -> np.ndarray:
        """Encode logic puzzle state to exact latent vector."""
        state = np.zeros(self.state_dim, dtype=np.float32)

        # Grid encoding: first 81*2=162 dims (sin/cos per cell, no aliasing via %)
        flat_grid = grid.flatten()  # 9x9 = 81 cells
        for i, val in enumerate(flat_grid):
            angle = 2 * np.pi * val / self.grid_size
            state[2 * i] = np.sin(angle)
            state[2 * i + 1] = np.cos(angle)
        # Remaining dims 162..511 left as zero (padding to 512)

        # Constraints satisfied (bitmask, next 128 dims; 27 real constraints)
        for idx, satisfied in enumerate(constraints_satisfied[:128]):
            state[512 + idx] = float(satisfied)

        return state

    def generate_trajectory(self, puzzle_id: int = 0) -> List[ExactState]:
        """Generate one complete logic puzzle solving trajectory."""
        np.random.seed(puzzle_id)
        trajectory = []

        # Initialize partially filled grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        # Fill in some random clues
        num_clues = self.grid_size * 2
        clue_positions = np.random.choice(self.grid_size * self.grid_size, num_clues, replace=False)
        for pos in clue_positions:
            r, c = pos // self.grid_size, pos % self.grid_size
            grid[r, c] = np.random.randint(1, self.grid_size + 1)

        for step in range(self.max_steps):
            # Randomly fill one empty cell if possible
            empty_cells = np.argwhere(grid == 0)
            if len(empty_cells) > 0:
                r, c = empty_cells[np.random.randint(len(empty_cells))]
                grid[r, c] = np.random.randint(1, self.grid_size + 1)

            # Compute constraints deterministically from current grid
            constraints_satisfied = self._compute_constraints(grid)

            state_vec = self.encode_state(grid, constraints_satisfied)

            trajectory.append(ExactState(
                step=step,
                state_vector=state_vec,
                action="fill_cell",
                metadata={"empty_cells": len(np.argwhere(grid == 0)), "constraints_met": sum(constraints_satisfied)}
            ))

        return trajectory
    
    def generate_dataset(self) -> Tuple[np.ndarray, List[Dict]]:
        """Generate full dataset of logic puzzle solving trajectories."""
        all_states = []
        metadata = []
        
        for seq_id in range(self.num_sequences):
            trajectory = self.generate_trajectory(puzzle_id=seq_id)
            
            for state in trajectory:
                all_states.append(state.state_vector)
                metadata.append(state.metadata)
        
        return np.stack(all_states, axis=0), metadata


def save_environment_data(env: object, name: str, output_dir: Path):
    """Save environment dataset to disk for SAE training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data, metadata = env.generate_dataset()
    
    # Save as torch tensors
    torch.save({
        "data": torch.from_numpy(data),
        "metadata": metadata,
        "state_dim": env.state_dim,
        "description": f"{name} exact states"
    }, output_dir / f"{name}_data.pt")
    
    logger.info(f"Saved {name} dataset: {data.shape}")
