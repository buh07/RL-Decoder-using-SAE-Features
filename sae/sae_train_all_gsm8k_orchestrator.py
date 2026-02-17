#!/usr/bin/env python3
"""
SAE Training Orchestrator: Train all SAE expansions on full GSM8K dataset
Distributed across GPUs 0-1 using tmux for parallel execution

Usage:
    python sae/sae_train_all_gsm8k_orchestrator.py [--resume]

Expected output:
    - All SAE checkpoints in: checkpoints/gpt2-small/sae/sae_768d_*x_final.pt
  - Training logs in: sae_logs/
  - Results aggregated in: sae_results/
  - Monitored via tmux sessions: sae_training_gpu0, sae_training_gpu1
"""

import os
import sys
import json
import subprocess
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints" / "gpt2-small" / "sae"
LOG_DIR = PROJECT_DIR / "sae_logs"
RESULTS_DIR = PROJECT_DIR / "sae_results"
GSM8K_SHARD_DIR = Path("/tmp/gpt2_gsm8k_acts/gsm8k/train")

# Create directories
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_available_sae_expansions() -> List[int]:
    """Get list of available SAE expansions from checkpoints."""
    expansions = []
    for checkpoint in CHECKPOINT_DIR.glob("sae_768d_*x_final.pt"):
        # Extract expansion: sae_768d_4x_final.pt -> 4
        name = checkpoint.stem
        expansion_str = name.split('_')[-2]  # e.g., "4x"
        expansion = int(expansion_str[:-1])
        expansions.append(expansion)
    
    return sorted(expansions)


def kill_tmux_session(session_name: str) -> None:
    """Kill tmux session if it exists."""
    try:
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            capture_output=True,
            timeout=5
        )
        logger.info(f"Killed existing tmux session: {session_name}")
    except Exception:
        pass


def create_tmux_session(session_name: str, window_name: str) -> None:
    """Create a new tmux session with a named window."""
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, "-n", window_name, "-c", str(PROJECT_DIR)],
        check=True,
        capture_output=True
    )


def send_tmux_command(session: str, window: str, command: str) -> None:
    """Send command to tmux window and execute."""
    subprocess.run(
        ["tmux", "send-keys", "-t", f"{session}:{window}", command, "Enter"],
        check=True,
        capture_output=True
    )


def train_sae_on_gsm8k(
    expansion: int,
    gpu_id: int,
    session_name: str,
    window_name: str
) -> str:
    """
    Create command to train a single SAE on full GSM8K dataset.
    
    Args:
        expansion: SAE expansion factor (e.g., 4, 6, 8, ...)
        gpu_id: GPU ID to use (0 or 1)
        session_name: tmux session name
        window_name: tmux window name
    
    Returns:
        Training command string
    """
    # Use shell quoting for paths with spaces
    import shlex
    
    cmd_parts = [
        "python", "src/sae_training.py",
        f"--shard-dir={GSM8K_SHARD_DIR}",
        "--model=gpt2",
        f"--expansion-factor={expansion}",
        f"--device=cuda:{gpu_id}",
        "--batch-size=64",
        "--max-epochs=10",
        f"--checkpoint-dir={CHECKPOINT_DIR}",
        "--no-wandb",  # Disable wandb to avoid clutter
    ]
    
    # Properly quote the command for shell execution
    return " ".join(shlex.quote(str(part)) for part in cmd_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Train all SAE expansions on full GSM8K dataset"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training (restart tmux sessions)"
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs=2,
        default=[0, 1],
        help="GPU IDs to use (default: 0 1)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("SAE Training Orchestrator - Full GSM8K Dataset")
    logger.info("=" * 80)
    logger.info("")
    
    # Check dataset
    if not GSM8K_SHARD_DIR.exists():
        logger.error(f"GSM8K dataset directory not found: {GSM8K_SHARD_DIR}")
        sys.exit(1)
    
    shard_files = list(GSM8K_SHARD_DIR.glob("*shard_*.pt"))
    if not shard_files:
        logger.error(f"No shard files found in {GSM8K_SHARD_DIR}")
        sys.exit(1)
    
    logger.info(f"✓ GSM8K dataset found: {len(shard_files)} shards")
    logger.info(f"  Location: {GSM8K_SHARD_DIR}")
    logger.info("")
    
    # Get SAE expansions
    expansions = get_available_sae_expansions()
    if not expansions:
        logger.error(f"No SAE checkpoints found in {CHECKPOINT_DIR}")
        sys.exit(1)
    
    logger.info(f"✓ Found {len(expansions)} SAE expansions: {expansions}")
    logger.info("")
    
    # Distribute across 2 GPUs
    gpu_ids = args.gpu_ids
    gpu_0_saes = [e for i, e in enumerate(expansions) if i % 2 == 0]
    gpu_1_saes = [e for i, e in enumerate(expansions) if i % 2 == 1]
    
    logger.info("GPU Distribution (round-robin):")
    logger.info(f"  GPU {gpu_ids[0]}: {gpu_0_saes} ({len(gpu_0_saes)} SAEs)")
    logger.info(f"  GPU {gpu_ids[1]}: {gpu_1_saes} ({len(gpu_1_saes)} SAEs)")
    logger.info("")
    
    # Kill existing sessions
    if args.resume:
        logger.info("Killing existing tmux sessions...")
        kill_tmux_session(f"sae_training_gpu{gpu_ids[0]}")
        kill_tmux_session(f"sae_training_gpu{gpu_ids[1]}")
        time.sleep(1)
    
    # Create tmux sessions
    logger.info("Creating tmux sessions...")
    
    session_0 = f"sae_training_gpu{gpu_ids[0]}"
    session_1 = f"sae_training_gpu{gpu_ids[1]}"
    
    create_tmux_session(session_0, "train")
    logger.info(f"  ✓ Created {session_0}")
    
    create_tmux_session(session_1, "train")
    logger.info(f"  ✓ Created {session_1}")
    
    logger.info("")
    
    # Activate virtual environment first
    logger.info("Preparing training environment...")
    time.sleep(0.5)
    
    send_tmux_command(session_0, "train", "source .venv/bin/activate")
    time.sleep(0.3)
    send_tmux_command(session_1, "train", "source .venv/bin/activate")
    time.sleep(0.3)
    
    # Launch training commands
    logger.info("Launching training on each GPU...")
    logger.info("")
    
    # GPU 0
    if gpu_0_saes:
        logger.info(f"GPU {gpu_ids[0]}: Training {gpu_0_saes}")
        for expansion in gpu_0_saes:
            cmd = train_sae_on_gsm8k(expansion, gpu_ids[0], session_0, "train")
            log_file = LOG_DIR / f"gpu{gpu_ids[0]}_expansion{expansion}x.log"
            # Use quoted log file path
            import shlex
            full_cmd = f"{cmd} 2>&1 | tee {shlex.quote(str(log_file))}"
            send_tmux_command(session_0, "train", full_cmd)
            logger.info(f"  - Sent training command for {expansion}x expansion")
            time.sleep(0.5)  # Give previous command time to start
    
    time.sleep(1)
    
    # GPU 1
    if gpu_1_saes:
        logger.info(f"GPU {gpu_ids[1]}: Training {gpu_1_saes}")
        for expansion in gpu_1_saes:
            cmd = train_sae_on_gsm8k(expansion, gpu_ids[1], session_1, "train")
            log_file = LOG_DIR / f"gpu{gpu_ids[1]}_expansion{expansion}x.log"
            # Use quoted log file path
            import shlex
            full_cmd = f"{cmd} 2>&1 | tee {shlex.quote(str(log_file))}"
            send_tmux_command(session_1, "train", full_cmd)
            logger.info(f"  - Sent training command for {expansion}x expansion")
            time.sleep(0.5)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Training Started Successfully")
    logger.info("=" * 80)
    logger.info("")
    
    logger.info("📊 MONITORING:")
    logger.info("")
    logger.info(f"  Attach to GPU {gpu_ids[0]} training:")
    logger.info(f"    tmux attach -t {session_0}")
    logger.info("")
    logger.info(f"  Attach to GPU {gpu_ids[1]} training:")
    logger.info(f"    tmux attach -t {session_1}")
    logger.info("")
    logger.info("  Watch both GPUs side-by-side:")
    logger.info(f"    tmux new-window -t {session_0} -n monitor")
    logger.info(f"    tmux send-keys -t {session_0}:monitor 'watch -n 2 nvidia-smi' Enter")
    logger.info("")
    
    logger.info("📁 OUTPUT:")
    logger.info("")
    logger.info(f"  Training logs: {LOG_DIR}/")
    logger.info(f"  SAE checkpoints: {CHECKPOINT_DIR}/")
    logger.info(f"  Results directory: {RESULTS_DIR}/")
    logger.info("")
    
    logger.info("⏱️ TIMELINE:")
    logger.info("")
    total_0 = len(gpu_0_saes) * 30  # ~30 min per SAE
    total_1 = len(gpu_1_saes) * 30
    max_time = max(total_0, total_1)
    logger.info(f"  GPU {gpu_ids[0]}: ~{len(gpu_0_saes)} SAEs × 30 min = ~{total_0} minutes")
    logger.info(f"  GPU {gpu_ids[1]}: ~{len(gpu_1_saes)} SAEs × 30 min = ~{total_1} minutes")
    logger.info(f"  Parallel total: ~{max_time} minutes")
    logger.info(f"  Estimated completion: {(max_time // 60):02d}h {(max_time % 60):02d}m")
    logger.info("")
    
    logger.info("📋 CONTROL:")
    logger.info("")
    logger.info(f"  Kill GPU {gpu_ids[0]} training:")
    logger.info(f"    tmux kill-session -t {session_0}")
    logger.info("")
    logger.info(f"  Kill GPU {gpu_ids[1]} training:")
    logger.info(f"    tmux kill-session -t {session_1}")
    logger.info("")
    logger.info("  Kill all training:")
    logger.info(f"    tmux kill-session -t {session_0}; tmux kill-session -t {session_1}")
    logger.info("")
    
    logger.info("✅ Run this command to monitor GPU usage:")
    logger.info("   watch -n 2 nvidia-smi")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")


if __name__ == "__main__":
    main()
