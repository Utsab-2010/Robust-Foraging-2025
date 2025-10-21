# train_alt.py - Modified version with alternating environment support and model continuity

import subprocess
import os
import time
from pathlib import Path
import glob
import replace
import pandas as pd

# ─── Helper functions (copied from original) ────────────────────────────────────────────────
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
        trial_type_idx = df.index[df.EventType=='s'].tolist()

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

# ─── New functions for alternating training with model continuity ────────────────

def find_latest_checkpoint(run_id):
    """Find the latest checkpoint (.pt file) for a given run_id."""
    results_dir = Path("./results")
    if not results_dir.exists():
        return None
    
    # Look for directories matching the run_id pattern
    matching_dirs = list(results_dir.glob(f"{run_id}"))
    
    if not matching_dirs:
        return None
    
    # Find the most recent checkpoint file in the run directory
    run_dir = matching_dirs[0]
    checkpoint_files = list(run_dir.glob("*.pt"))
    
    if not checkpoint_files:
        return None
    
    # Get the most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    return latest_checkpoint

def train_alternating(run_id, base_env_path, config_path, total_runs=5, log_name=None):
    """
    Train a single network alternating between NormalTrain and FogTrain environments.
    Maintains model continuity by loading from checkpoints.
    """
    print(f"🔄 ALTERNATING TRAINING MODE: {run_id}")
    print(f"   Will alternate between NormalTrain and FogTrain every run")
    print(f"   Total runs: {total_runs}")
    print(f"   🔗 Model continuity: Each run loads from previous checkpoint")
    
    next_run = get_next_run_number(run_id)
    run_id_list = []
    
    for i in range(total_runs):
        current_run_num = next_run + i
        current = f"{run_id}_{current_run_num}"
        
        # Determine environment for this run (alternate each run)
        if i % 2 == 0:  # Even runs (0, 2, 4, ...) use NormalTrain
            env_name = "NormalTrain"
            env_path = base_env_path.replace("AltTrain", "NormalTrain")
        else:  # Odd runs (1, 3, 5, ...) use FogTrain
            env_name = "FogTrain"
            env_path = base_env_path.replace("AltTrain", "FogTrain")
        
        print(f"\n🎯 Starting training: {current} (Environment: {env_name})")
        
        # Set up logging
        fn = f"{(log_name if log_name else run_id)}_{current_run_num}_train.txt"
        sa = os.path.join(env_path, "2D go to target v1_Data", "StreamingAssets", "currentLog.txt")
        
        with open(sa, "w") as f:
            f.write(fn)
        
        time.sleep(1)
        
        # Build command
        cmd = [
            "mlagents-learn",
            config_path,
            "--env", str(Path(env_path) / "2D go to target v1.exe"),
            "--run-id", current,
            "--env-args", "--screen-width=155", "--screen-height=86",
        ]
        
        # Handle model continuity for runs after the first
        if i > 0:
            # Look for the previous run's checkpoint
            prev_run_id = f"{run_id}_{current_run_num - 1}"
            checkpoint = find_latest_checkpoint(prev_run_id)
            
            if checkpoint:
                print(f"   📁 Loading from checkpoint: {checkpoint}")
                cmd.extend(["--resume", str(checkpoint)])
            else:
                print(f"   ⚠️  No checkpoint found for {prev_run_id}, starting fresh")
                cmd.append("--force")
        else:
            # First run - start fresh
            print(f"   🆕 First run - starting fresh")
            cmd.append("--force")
        
        print(f"   🚀 Command: {' '.join(cmd)}")
        
        # Run training
        try:
            subprocess.run(cmd, check=True)
            print(f"   ✅ Completed training: {current} (Environment: {env_name})")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Training failed for {current}: {e}")
            break
        
        time.sleep(5)
        run_id_list.append(current)
    
    return run_id_list

def train_solo_original(run_id, env_path, config_path, total_runs=5, log_name=None):
    """Original train_solo function for non-alternating environments."""
    next_run = get_next_run_number(run_id)
    run_id_list = []
    for i in range(total_runs):
        current = f"{run_id}_{next_run + i}"
        print(f"Starting training: {current}")
        
        fn = f"{(log_name if log_name else run_id)}_{next_run + i}_train.txt"
        sa = os.path.join(env_path, "2D go to target v1_Data", "StreamingAssets", "currentLog.txt")
        
        with open(sa, "w") as f:
            f.write(fn)

        time.sleep(1)
        cmd = [
            "mlagents-learn",
            config_path,
            "--env", str(Path(env_path) / "2D go to target v1.exe"),
            "--run-id", current,
            "--force",
            "--env-args", "--screen-width=155", "--screen-height=86",
        ]
        subprocess.run(cmd, check=True)
        print(f"Completed training: {current}")
        time.sleep(5)
        run_id_list.append(current)
    return run_id_list

def train_multiple_networks_alt(networks, env_path, runs_per_network=2, log_name=None, env='NormalTrain'):
    """
    Modified version that supports alternating environments with model continuity.
    """
    run_id_list2 = []
    
    # Check if we're using alternating mode
    is_alternating = env == 'AltTrain'
    
    for network in networks:
        print(f"\n🔧 Setting up network: {network}")
        
        # Set up config path
        if network == "fully_connected":
            config_path = "./Config/fc.yaml"
        elif network == "simple":
            config_path = "./Config/simple.yaml"
        elif network == "resnet":
            config_path = "./Config/resnet.yaml"
        else:
            config_path = "./Config/nature.yaml"
            if network != "nature_cnn":
                print(f"Warning: Unrecognized network '{network}', defaulting to nature_cnn")
                replace.replace_nature_visual_encoder("D:/miniconda/envs/mouse/Lib/site-packages/mlagents/trainers/torch/encoders.py", "./Encoders/" + network + ".py")
        
        # Train the network
        if is_alternating:
            run_ids = train_alternating(
                run_id=f"{network}_{env}",
                base_env_path=env_path,
                config_path=config_path,
                total_runs=runs_per_network,
                log_name=log_name
            )
        else:
            # Use original training method for non-alternating environments
            run_ids = train_solo_original(
                run_id=f"{network}_{env}",
                env_path=env_path,
                config_path=config_path,
                total_runs=runs_per_network,
                log_name=log_name
            )
        
        run_id_list2.extend(run_ids)
    
    return run_id_list2

# ─── Main CLI entrypoint ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train multiple networks on MouseVsAI with alternating environment support")
    parser.add_argument("--env", type=str, default="NormalTrain",
                        help="Environment: 'AltTrain' for alternating NormalTrain/FogTrain, or specific folder name")
    parser.add_argument("--runs-per-network", type=int, default=2,
                        help="How many runs per network variant")
    parser.add_argument("--networks", type=str, default="nature_cnn,simple,resnet",
                        help="Comma-separated list of network names")
    parser.add_argument("--log-name", type=str, default=None,
                        help="Optional prefix for all log files")
    args = parser.parse_args()

    # Handle environment setup
    if args.env == "AltTrain":
        env_folder = f"./Builds/AltTrain"  # Base path - will be modified dynamically
        print("🔄 ALTERNATING ENVIRONMENT MODE ENABLED:")
        print("   Even runs (0,2,4...) → NormalTrain environment")
        print("   Odd runs (1,3,5...)  → FogTrain environment")
        print("   🔗 Model continuity: Each run loads from previous checkpoint")
        print("   📋 Same network trains across both environments!")
    else:
        env_folder = f"./Builds/{args.env}"
        print(f"📁 Single environment mode: {args.env}")
    
    nets = [n.strip() for n in args.networks.split(",")]
    print(f"🧠 Networks to train: {nets}")
    print(f"🔢 Runs per network: {args.runs_per_network}")
    
    run_ids = train_multiple_networks_alt(nets, env_folder, args.runs_per_network, args.log_name, args.env)

    # Summarize each run
    logs_dir = Path("./logfiles")
    logs_dir.mkdir(exist_ok=True)
    for rid in run_ids:
        summary_file = logs_dir / f"{(args.log_name if args.log_name else rid)}_train.txt"
        print(f"\n=== Summary for {rid} ===")
        if summary_file.exists():
            summarize_log(str(summary_file))
        else:
            print(f"⚠️  Log file not found: {summary_file}")
