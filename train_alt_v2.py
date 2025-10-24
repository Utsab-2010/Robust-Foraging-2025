# train_alt_v2.py - Alternating environment training script with proper checkpoint handling

import subprocess
import os
import time
from pathlib import Path
import glob
import replace
import pandas as pd
import argparse

# ─── Helper functions ────────────────────────────────────────────────
def get_next_session_number(network_name, results_dir="./results"):
    """Get the next session number for a given network by checking existing session folders."""
    os.makedirs(results_dir, exist_ok=True)
    pattern = os.path.join(results_dir, f"{network_name}_AltTrain_*")
    existing_sessions = glob.glob(pattern)
    if not existing_sessions:
        return 1
    
    session_numbers = []
    for session_path in existing_sessions:
        try:
            # Extract session number from folder name like "trans_v6_AltTrain_3"
            session_num = int(session_path.split('_')[-1])
            session_numbers.append(session_num)
        except (ValueError, IndexError):
            continue
    return max(session_numbers) + 1 if session_numbers else 1

def get_next_run_number(base_name, results_dir="./results"):
    """Get the next run number for a given base name by checking existing results."""
    os.makedirs(results_dir, exist_ok=True)
    pattern = os.path.join(results_dir, f"{base_name}_*")
    existing_runs = glob.glob(pattern)
    if not existing_runs:
        return 1
    run_numbers = []
    for run_path in existing_runs:
        try:
            run_num = int(run_path.split('_')[-1])
            run_numbers.append(run_num)
        except (ValueError, IndexError):
            continue
    return max(run_numbers) + 1 if run_numbers else 1

def summarize_log(log_path: str):
    """
    Reads the Unity log at log_path, then prints:
      • Overall success rate (%)
      • Success rate per trial type
      • Max target distance (units)
    """
    try:
        if not Path(log_path).exists():
            print(f"⚠️  Log file not found: {log_path}")
            return
            
        df = pd.read_csv(
            log_path,
            sep=r'\s+',
            comment='#',
            header=None,
            names=['SessionTime','EventType','x','y','z','r','extra'],
            usecols=[0,1,2,4,5],
            engine='python'
        )
        df = df[df.EventType.isin(['n','t','s','h','f'])].reset_index(drop=True)

        new_trial_idxs = df.index[df.EventType=='n'].tolist()
        successes = []
        by_type = {}
        distances = []

        for ti, start_idx in enumerate(new_trial_idxs):
            end_idx = new_trial_idxs[ti+1] if ti+1 < len(new_trial_idxs) else len(df)
            trial = df.iloc[start_idx:end_idx]

            ttype = int(trial.loc[trial.EventType=='s','x'].iat[0])

            trow = trial[trial.EventType=='t']
            if trow.empty:
                continue
            dx, dz = float(trow.x.iat[0]), float(trial.loc[trow.index,'z'].iat[0])
            distances.append(dx)

            hit = 1 if ('h' in trial.EventType.values) else 0
            successes.append(hit)
            by_type.setdefault(ttype, []).append(hit)

        if successes:
            overall = sum(successes)/len(successes)*100
            print(f"\n=== EVALUATION RESULTS ===")
            print(f"Overall success rate: {overall:.1f}% ({sum(successes)}/{len(successes)})")
            for ttype, hits in by_type.items():
                rate = sum(hits)/len(hits)*100
                print(f"  * Trial type {ttype}: {rate:.1f}% ({sum(hits)}/{len(hits)})")
        if distances:
            print(f"Max target distance reached: {max(distances):.3f}/5.00")
        print("==========================\n")
    except Exception as e:
        print(f"Error reading log file {log_path}: {e}")

# ─── Checkpoint management functions ────────────────────────────────────────────────
def find_latest_checkpoint(run_id, results_base_dir):
    """Find the latest checkpoint for a given run_id within results directory."""
    results_dir = results_base_dir / run_id
    
    if not results_dir.exists():
        print(f"   🔍 No results directory found: {results_dir}")
        return None
    
    behavior_dir = results_dir / "My Behavior"
    if not behavior_dir.exists():
        print(f"   🔍 No behavior directory found: {behavior_dir}")
        return None
    
    # Look for checkpoint files in order of preference
    checkpoint_files = []
    
    # 1. Standard checkpoint.pt
    standard_checkpoint = behavior_dir / "checkpoint.pt"
    if standard_checkpoint.exists():
        checkpoint_files.append(standard_checkpoint)
    
    # 2. Numbered checkpoints (e.g., My Behavior-85.pt)
    numbered_checkpoints = list(behavior_dir.glob("My Behavior-*.pt"))
    checkpoint_files.extend(numbered_checkpoints)
    
    # 3. Any other .pt files
    other_checkpoints = [f for f in behavior_dir.glob("*.pt") if f not in checkpoint_files]
    checkpoint_files.extend(other_checkpoints)
    
    if not checkpoint_files:
        print(f"   🔍 No checkpoint files found in: {behavior_dir}")
        return None
    
    # Return the most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    print(f"   🔍 Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def cleanup_session_data(session_name):
    """Cleanup ML-Agents run data for a session."""
    import shutil
    
    # Clean session directory
    session_results_dir = Path("./results") / session_name
    if session_results_dir.exists():
        print(f"   🧹 Removing session directory: {session_results_dir}")
        try:
            shutil.rmtree(session_results_dir)
        except Exception as e:
            print(f"   ⚠️  Could not remove {session_results_dir}: {e}")
    
    # Clean any tensorboard logs (summaries directory)
    summaries_session_dir = Path("./summaries") / session_name
    if summaries_session_dir.exists():
        print(f"   🧹 Removing summaries directory: {summaries_session_dir}")
        try:
            shutil.rmtree(summaries_session_dir)
        except Exception as e:
            print(f"   ⚠️  Could not remove {summaries_session_dir}: {e}")

# ─── Main alternating training function ────────────────────────────────────────────────
def train_alternating_v2(network_name, total_runs, config_path, clean_start=False, use_gpu=False):
    """
    Train alternating between NormalTrain and FogTrain environments.
    Each run builds upon the previous one using checkpoints.
    All runs are organized under a session directory.
    """
    # Create session-based naming (but use default ML-Agents structure)
    session_number = get_next_session_number(network_name)
    session_name = f"{network_name}_AltTrain_{session_number}"
    
    print(f"🔄 ALTERNATING TRAINING V2 SESSION: {session_name}")
    print(f"   Network: {network_name}")
    print(f"   Session Run ID: {session_name}")
    print(f"   Total runs: {total_runs}")
    print(f"   Will alternate: NormalTrain ↔ FogTrain")
    print(f"   Model continuity: Each run loads from previous checkpoint")
    if use_gpu:
        print(f"   🚀 GPU acceleration: ENABLED")
    else:
        print(f"   💻 GPU acceleration: DISABLED (CPU only)")
    
    # Handle clean start
    if clean_start:
        print(f"   🧹 Clean start requested")
        cleanup_session_data(session_name)
        time.sleep(2)
    
    # Replace encoder if it's a custom network
    if network_name not in ["nature_cnn", "simple", "resnet", "fully_connected"]:
        print(f"   🔧 Installing custom encoder: {network_name}")
        replace.replace_nature_visual_encoder(
            "D:/miniconda/envs/mouse/Lib/site-packages/mlagents/trainers/torch/encoders.py",
            f"./Encoders/{network_name}.py"
        )
    
    run_results = []
    
    for run_num in range(1, total_runs + 1):
        # Determine environment for this run
        if run_num % 2 == 1:  # Odd runs (1, 3, 5, ...) use NormalTrain
            env_name = "NormalTrain"
            env_path = f"./Builds/NormalTrain"
        else:  # Even runs (2, 4, 6, ...) use FogTrain
            env_name = "FogTrain"
            env_path = f"./Builds/FogTrain"
        
        # Create unique run ID for each environment to avoid conflicts
        current_run_id = f"run{run_num}_{env_name}"
        session_results_dir = f"./results/{session_name}"
        
        print(f"\n🎯 === RUN {run_num}/{total_runs} ===")
        print(f"   Environment: {env_name}")
        print(f"   Current Run ID: {current_run_id}")
        print(f"   Results will be saved to: {session_results_dir}/{current_run_id}")
        
        # Set up logging
        log_filename = f"{session_name}_{current_run_id}_train.txt"
        log_path = os.path.join(env_path, "2D go to target v1_Data", "StreamingAssets", "currentLog.txt")
        
        with open(log_path, "w") as f:
            f.write(log_filename)
        
        time.sleep(1)
        
        # Build base command using custom results directory
        cmd = [
            "mlagents-learn",
            config_path,
            "--env", str(Path(env_path) / "2D go to target v1.exe"),
            "--run-id", current_run_id,  # Unique run ID for each environment
            "--results-dir", session_results_dir,  # Session-specific results directory
            "--env-args", "--screen-width=155", "--screen-height=86",
            "--force"  # Always use force to overwrite any existing data
        ]
        
        # Add GPU support if requested
        if use_gpu:
            cmd.extend(["--torch-device", "cuda"])
            print(f"   🚀 Using GPU (CUDA) for training")
        
        # Handle model continuity using --initialize-from
        if run_num == 1:
            # First run - start fresh
            print(f"   🆕 First run - starting fresh")
        else:
            # Subsequent runs - initialize from previous run
            previous_run_num = run_num - 1
            previous_env_name = "FogTrain" if previous_run_num % 2 == 0 else "NormalTrain"
            previous_run_id = f"run{previous_run_num}_{previous_env_name}"
            
            # Check if previous run exists in session directory
            session_results_path = Path(session_results_dir)
            previous_checkpoint = find_latest_checkpoint(previous_run_id, session_results_path)
            
            if previous_checkpoint:
                print(f"   🔄 Initializing from previous run: {previous_run_id}")
                cmd.extend(["--initialize-from", previous_run_id])
            else:
                print(f"   ⚠️  No checkpoint found for previous run {previous_run_id}, starting fresh")
        
        print(f"   🚀 Command: {' '.join(cmd)}")
        
        # Execute training
        try:
            # Save the exact command used for debugging
            try:
                with open("last_mlagents_command.txt", "w", encoding="utf-8") as cf:
                    cf.write(" ".join(cmd))
            except Exception as _:
                pass

            print(f"   ▶️  Starting training...")
            subprocess.run(cmd, check=True)
            print(f"   ✅ Completed run {run_num}: {current_run_id} ({env_name})")
            run_results.append((f"{session_name}_{current_run_id}", env_name))
            
            # Brief pause between runs
            time.sleep(3)
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Training failed for run {run_num}: {e}")
            print(f"   💡 Troubleshooting:")
            print(f"      • Try: python train_alt_v2.py --network {network_name} --runs {total_runs} --clean-start")
            print(f"      • Check if Unity executable is accessible")
            print(f"      • Verify config file: {config_path}")
            print(f"      • Check results directory: ./results/{current_run_id}")
            break
    
    return run_results, session_name

# ─── Network configuration ────────────────────────────────────────────────
def get_config_path(network_name):
    """Get the appropriate config file for a network."""
    config_map = {
        "fully_connected": "./Config/fc.yaml",
        "simple": "./Config/simple.yaml", 
        "resnet": "./Config/resnet.yaml",
        "nature_cnn": "./Config/nature.yaml"
    }
    
    # Default to nature config for custom networks
    return config_map.get(network_name, "./Config/nature.yaml")

# ─── Main CLI interface ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Alternating Environment Training Script V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train trans_v6 for 6 alternating runs
  python train_alt_v2.py --network trans_v6 --runs 6
  
  # Clean start with custom network
  python train_alt_v2.py --network trans_v5 --runs 8 --clean-start
  
  # Train built-in network  
  python train_alt_v2.py --network simple --runs 4
        """
    )
    
    parser.add_argument("--network", type=str, required=True,
                        help="Network name (e.g., trans_v6, nature_cnn, simple)")
    parser.add_argument("--runs", type=int, required=True,
                        help="Total number of alternating runs")
    parser.add_argument("--config", type=str, default=None,
                        help="Config file path (auto-detected if not specified)")
    parser.add_argument("--clean-start", action="store_true",
                        help="Remove existing run data before starting")
    parser.add_argument("--no-summary", action="store_true",
                        help="Skip log file summaries at the end")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (CUDA) for training acceleration")
    
    args = parser.parse_args()
    
    # Determine config path
    config_path = args.config if args.config else get_config_path(args.network)
    
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        return 1
    
    print(f"🎮 ALTERNATING TRAINING SETUP")
    print(f"   Network: {args.network}")
    print(f"   Config: {config_path}")
    print(f"   Total Runs: {args.runs}")
    print(f"   Pattern: Odd runs → NormalTrain, Even runs → FogTrain")
    if args.clean_start:
        print(f"   🧹 Clean start: Will remove existing data")
    if args.gpu:
        print(f"   🚀 GPU acceleration: ENABLED")
        # Check if CUDA is available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   📱 GPU detected: {gpu_name}")
            else:
                print(f"   ⚠️  GPU requested but CUDA not available - will use CPU")
                args.gpu = False  # Disable GPU if not available
        except ImportError:
            print(f"   ⚠️  PyTorch not available for GPU detection")
    else:
        print(f"   💻 GPU acceleration: DISABLED (CPU only)")
    
    # Execute training
    run_results, session_name = train_alternating_v2(
        network_name=args.network,
        total_runs=args.runs,
        config_path=config_path,
        clean_start=args.clean_start,
        use_gpu=args.gpu
    )
    
    # Summarize results
    if not args.no_summary and run_results:
        print(f"\n📊 TRAINING SUMMARY")
        print(f"   Session: {session_name}")
        print(f"   Completed {len(run_results)}/{args.runs} runs")
        
        logs_dir = Path("./logfiles")
        logs_dir.mkdir(exist_ok=True)
        
        for run_id, env_name in run_results:
            log_file = logs_dir / f"{run_id}_train.txt"
            print(f"\n=== Summary for {run_id} ({env_name}) ===")
            summarize_log(str(log_file))
    
    print(f"\n🏁 Training complete!")
    return 0

if __name__ == "__main__":
    exit(main())