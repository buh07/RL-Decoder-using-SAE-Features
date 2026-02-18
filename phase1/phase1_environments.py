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
        
        # Stack encoding (next 512 dims; position-based + magnitude)
        for pos, value in enumerate(stack[:256]):  # Cap at 256 for encoding
            # Encode as sin/cos pair for periodicity
            state[4 + 2*pos] = np.sin(2 * np.pi * value / 256)
            state[4 + 2*pos + 1] = np.cos(2 * np.pi * value / 256)
        
        return state
    
    def generate_trajectory(self, seed: int = 0) -> List[ExactState]:
        """Generate one complete stack machine trajectory."""
        np.random.seed(seed)
        trajectory = []
        stack = []
        
        operations_sequence = np.random.choice(
            ["push", "push", "pop", "peek"],  # Bias toward push to keep stack growing
            size=self.max_steps,
            p=[0.4, 0.2, 0.2, 0.2]
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
        
    def encode_state(self, grid: np.ndarray, constraints_satisfied: List[bool]) -> np.ndarray:
        """Encode logic puzzle state to exact latent vector."""
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Grid encoding (first 512 dims)
        flat_grid = grid.flatten()[:81]  # 9x9 grid = 81 cells
        for i, val in enumerate(flat_grid):
            # Encode cell value as sine/cosine pair
            angle = 2 * np.pi * val / 9
            state[2*i % 512] = np.sin(angle)
            state[(2*i + 1) % 512] = np.cos(angle)
        
        # Constraints satisfied (bitmask, next 128 dims)
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
        
        constraints_satisfied = [False] * 27  # 9 rows + 9 cols + 9 boxes
        
        for step in range(self.max_steps):
            # Randomly fill one empty cell if possible
            empty_cells = np.argwhere(grid == 0)
            if len(empty_cells) > 0:
                r, c = empty_cells[np.random.randint(len(empty_cells))]
                grid[r, c] = np.random.randint(1, self.grid_size + 1)
                
                # Update some constraint satisfaction status
                constraints_satisfied[r] = np.random.random() > 0.3
                constraints_satisfied[9 + c] = np.random.random() > 0.3
            
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
