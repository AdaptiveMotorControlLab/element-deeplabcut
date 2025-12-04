#!/usr/bin/env python
"""
Real fine-tuning of SuperAnimal-Quadruped on your own data
==========================================================

This script supports **both**:
- TensorFlow SuperAnimal ("classic" DLC SuperAnimal)
- PyTorch SuperAnimal ModelZoo (HRNet-W32 + Faster R-CNN)

You choose the engine with:  --engine tensorflow   OR   --engine pytorch

End-to-end workflow (exact order)
---------------------------------
1. Create project + run SuperAnimal + extract frames
   - `--create-project --run-superanimal --extract-frames`
   - For TF: uses create_pretrained_project(model="superanimal_quadruped")
   - For PT: uses create_new_project(engine=pytorch) + ModelZoo (hrnet_w32)

2. Open GUI with refine_labels → correct + save
   - `dlc.refine_labels(config)` or `--refine-labels`
   - Load SuperAnimal predictions + frames in GUI
   - Correct keypoints interactively
   - SAVE → creates CollectedData_*.csv/.h5  (your training labels)

3. Create training dataset
   - `--create-dataset`
   - Uses CollectedData_*.csv/.h5 from step 2
   - For PyTorch: creates a top-down HRNet-W32 + Faster R-CNN dataset
     with SuperAnimal-Quadruped weights (transfer learning).

4. Train the model
   - `--train`
   - For TF: standard DLC training (maxiters-based)
   - For PT: DLC training with epoch-based PyTorch API
     and SuperAnimal transfer learning (ModelZoo weights)
"""

import argparse
import sys
from pathlib import Path

import deeplabcut as dlc


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real fine-tuning of SuperAnimal-Quadruped (TF or PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------

# 1) Create project + run SuperAnimal + extract frames (one shot)
python real_quadruped_training_example.py \\
    --engine tensorflow \\
    --project-name quad_superanimal \\
    --experimenter mariia \\
    --videos /path/to/your_quadruped_video.mp4 \\
    --create-project \\
    --run-superanimal \\
    --extract-frames

# Same but with PyTorch ModelZoo backend
python real_quadruped_training_example.py \\
    --engine pytorch \\
    --project-name quad_superanimal \\
    --experimenter mariia \\
    --videos /path/to/your_quadruped_video.mp4 \\
    --create-project \\
    --run-superanimal \\
    --extract-frames

# 2) Refine labels in GUI (after predictions & frames exist)
python real_quadruped_training_example.py \\
    --engine pytorch \\
    --project-dir /path/to/quad_superanimal-mariia-YYYY-MM-DD \\
    --refine-labels

# 3) Create dataset from refined labels
python real_quadruped_training_example.py \\
    --engine pytorch \\
    --project-dir /path/to/quad_superanimal-mariia-YYYY-MM-DD \\
    --create-dataset

# 4) Train (fine-tune) the model
python real_quadruped_training_example.py \\
    --engine pytorch \\
    --project-dir /path/to/quad_superanimal-mariia-YYYY-MM-DD \\
    --train \\
    --epochs 50 \\
    --save-epochs 10
""",
    )

    # Engine selection
    parser.add_argument(
        "--engine",
        type=str,
        choices=["tensorflow", "pytorch"],
        default="pytorch",
        help="Backend engine to use: 'tensorflow' (classic) or 'pytorch' (ModelZoo). Default: pytorch.",
    )

    # Project / config location
    parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Existing DLC project directory (if omitted and --create-project is used, a new project is created).",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="quad_superanimal",
        help="Project name (default: quad_superanimal).",
    )
    parser.add_argument(
        "--experimenter",
        type=str,
        default="mariia",
        help="Experimenter name (default: mariia).",
    )
    parser.add_argument(
        "--videos",
        type=str,
        nargs="+",
        default=None,
        help="One or more video paths for project creation (required for --create-project).",
    )

    # Workflow flags
    parser.add_argument(
        "--create-project",
        action="store_true",
        help="Create a new project (TF: create_pretrained_project; PT: create_new_project).",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Run deeplabcut.extract_frames on the config.",
    )
    parser.add_argument(
        "--run-superanimal",
        action="store_true",
        help="Run SuperAnimal inference on project videos (auto-predict).",
    )
    parser.add_argument(
        "--refine-labels",
        action="store_true",
        help="Open DLC refine_labels GUI to correct predictions into labels.",
    )
    parser.add_argument(
        "--create-dataset",
        action="store_true",
        help="Run deeplabcut.create_training_dataset on the config.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run deeplabcut.train_network (real training / fine-tuning).",
    )

    # SuperAnimal / GPU parameters (inference)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for SuperAnimal inference (where supported). Default: 16.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (where supported by DLC). Default: 0.",
    )
    
    # PyTorch inference parameters (for better detection of tiny animals in 4K video)
    # Note: --detector-batch-size is defined in training section below (shared for both inference and training)
    parser.add_argument(
        "--bbox-threshold",
        type=float,
        default=0.1,
        help="Bounding box threshold for detector (lower = less strict). Default: 0.1.",
    )
    parser.add_argument(
        "--pcutoff",
        type=float,
        default=0.05,
        help="P-cutoff threshold for pose estimation (lower = less strict). Default: 0.05.",
    )
    parser.add_argument(
        "--pseudo-threshold",
        type=float,
        default=0.05,
        help="Pseudo threshold for pose estimation. Default: 0.05.",
    )
    parser.add_argument(
        "--scale-list",
        type=int,
        nargs="+",
        default=[300, 400, 500, 600, 700, 800],
        help="Scale list for multi-scale detection (better for tiny animals). Default: [300, 400, 500, 600, 700, 800].",
    )
    parser.add_argument(
        "--video-adapt",
        action="store_true",
        help="Enable video adaptation for better tracking (default: True).",
    )
    parser.add_argument(
        "--no-video-adapt",
        dest="video_adapt",
        action="store_false",
        help="Disable video adaptation.",
    )
    # Set default after adding arguments (before parsing)
    parser.set_defaults(video_adapt=True)
    parser.add_argument(
        "--adapt-iterations",
        type=int,
        default=1500,
        help="Number of iterations for video adaptation. Default: 1500.",
    )
    parser.add_argument(
        "--detector-epochs-inference",
        type=int,
        default=10,
        help="Detector epochs for inference (if supported). Default: 10.",
    )
    parser.add_argument(
        "--pose-epochs-inference",
        type=int,
        default=10,
        help="Pose epochs for inference (if supported). Default: 10.",
    )

    # Training dataset / training hyperparameters
    parser.add_argument(
        "--shuffle",
        type=int,
        default=1,
        help="Shuffle index for training (default: 1).",
    )
    parser.add_argument(
        "--trainingsetindex",
        type=int,
        default=0,
        help="Training set index (default: 0).",
    )

    # TF-style training params (used only if engine == tensorflow)
    parser.add_argument(
        "--maxiters",
        type=int,
        default=50000,
        help="Max iterations for TF train_network (default: 50000).",
    )
    parser.add_argument(
        "--displayiters",
        type=int,
        default=100,
        help="Display iterations for TF train_network (default: 100).",
    )
    parser.add_argument(
        "--saveiters",
        type=int,
        default=1000,
        help="Save iterations for TF train_network (default: 1000).",
    )

    # PyTorch-style training params (used only if engine == pytorch)
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs for PyTorch training (default: 50).",
    )
    parser.add_argument(
        "--save-epochs",
        type=int,
        default=10,
        help="Save snapshot every N epochs for PyTorch (default: 10).",
    )
    parser.add_argument(
        "--detector-epochs",
        type=int,
        default=0,
        help="Detector epochs for PyTorch top-down SuperAnimal (default: 0 = only pose head).",
    )
    parser.add_argument(
        "--detector-save-epochs",
        type=int,
        default=None,
        help="Detector save interval in epochs (PyTorch). If None, use pytorch_config.yaml.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        help="Batch size for PyTorch training (pose model). Default: 16.",
    )
    parser.add_argument(
        "--detector-batch-size",
        type=int,
        default=1,
        help="Detector batch size for PyTorch training. Default: 1.",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Config / project handling
# ----------------------------------------------------------------------


def get_config_path(args: argparse.Namespace) -> Path:
    """
    Resolve or create the DLC config.yaml, depending on engine.

    For TensorFlow engine:
      - --create-project uses create_pretrained_project(model="superanimal_quadruped")
    For PyTorch engine:
      - --create-project uses create_new_project(..., engine="pytorch")
        and sets engine: pytorch in config.yaml
    """
    if args.create_project:
        if not args.videos:
            raise ValueError("--create-project requires at least one --videos path.")

        video_paths = [str(Path(v).expanduser().resolve()) for v in args.videos]

        print("\n=== Creating project ===")
        print(f"  engine       : {args.engine}")
        print(f"  project_name : {args.project_name}")
        print(f"  experimenter : {args.experimenter}")
        print(f"  videos       : {video_paths}")

        if args.engine == "tensorflow":
            # Classic TF SuperAnimal (dlcrnet-based) via create_pretrained_project
            config_path = dlc.create_pretrained_project(
                args.project_name,
                args.experimenter,
                video_paths,
                model="superanimal_quadruped",
            )
            config_path = Path(config_path)
        else:
            # PyTorch engine: normal project, then we use ModelZoo weights later
            project_path = dlc.create_new_project(
                args.project_name,
                args.experimenter,
                video_paths,
                copy_videos=False,
            )
            # create_new_project returns the config.yaml path directly (as a string)
            config_path = Path(project_path)

            # Verify it's actually a config.yaml file
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")
            if config_path.name != "config.yaml":
                raise ValueError(f"Expected config.yaml, got {config_path.name}")

            # Ensure engine is set to pytorch in config
            import yaml

            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            old_engine = cfg.get("engine", "not set")
            cfg["engine"] = "pytorch"
            with open(config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            print(f"  ✅ Set engine: {old_engine} → pytorch in config.yaml")

        print(f"\n✅ Project created. Config: {config_path}")
        return config_path

    # Reuse existing project
    if args.project_dir:
        project_dir = Path(args.project_dir).expanduser().resolve()
    else:
        # Auto-discover latest project matching pattern <project_name>-<experimenter>-*
        search_root = Path.cwd()
        pattern = f"{args.project_name}-{args.experimenter}-*"
        candidates = list(search_root.glob(pattern))

        if not candidates:
            raise FileNotFoundError(
                "No existing project found and --create-project was not used.\n"
                f"Looked for directories matching '{pattern}' in: {search_root}\n\n"
                "Either run with --create-project first, or pass --project-dir explicitly.\n"
            )
        project_dir = max(candidates, key=lambda p: p.stat().st_mtime)

    config_path = project_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")

    print(f"\nUsing existing project: {project_dir}")
    print(f"Config: {config_path}")
    return config_path


# ----------------------------------------------------------------------
# Optional: minimal bodyparts override
# ----------------------------------------------------------------------


def enforce_minimal_bodyparts(config_path: Path, bodyparts=None):
    """
    Optionally override the project to use a minimal set of bodyparts.

    This is useful if you only care about a couple of keypoints
    (e.g. head, tailbase) and want a "minimal pose" model.
    """
    import yaml

    if bodyparts is None:
        bodyparts = ["head", "tailbase"]

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["bodyparts"] = bodyparts

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"✅ Minimal bodyparts set written to {config_path}: {bodyparts}")


# ----------------------------------------------------------------------
# Steps: extract frames, run SuperAnimal, refine, dataset, train
# ----------------------------------------------------------------------


def step_extract_frames(config_path: Path):
    """Run DLC frame extraction."""
    print("\n=== Step: extract_frames ===")
    dlc.extract_frames(
        str(config_path),
        mode="automatic",
        algo="kmeans",
        crop=False,
    )
    print("✅ Frames extracted.")
    print("👉 Next: run --run-superanimal (if not done yet), then refine labels in the GUI.")


def step_run_superanimal_inference_tf(config_path: Path, args: argparse.Namespace):
    """
    TensorFlow SuperAnimal inference (classic).

    - Uses: superanimal_quadruped (TF dlcrnet model)
    - API:  video_inference_superanimal(videos, superanimal_name=..., ...)
    """
    from deeplabcut.utils import auxiliaryfunctions as aux

    print("\n=== Step: video_inference_superanimal (TensorFlow) ===")
    cfg = aux.read_config(str(config_path))

    if args.videos:
        videos = [str(Path(v).expanduser().resolve()) for v in args.videos]
    else:
        videos = list(cfg.get("video_sets", {}).keys())

    if not videos:
        raise ValueError("No videos found. Pass --videos or ensure video_sets is defined in config.yaml.")

    dest_folder = Path(config_path).parent / "superanimal_predictions_tf"
    dest_folder.mkdir(parents=True, exist_ok=True)

    print(f"  engine          : tensorflow")
    print(f"  superanimal_name: superanimal_quadruped")
    print(f"  dest_folder     : {dest_folder}")
    print(f"  gputouse        : {args.gpu}")
    print(f"  batch_size      : {args.batch_size} (used if DLC TF API supports it)")

    # Check what parameters the function accepts
    import inspect

    sig = inspect.signature(dlc.video_inference_superanimal)
    param_names = list(sig.parameters.keys())

    kwargs = {
        "superanimal_name": "superanimal_quadruped",
        "dest_folder": str(dest_folder),
    }

    # Use device parameter if available (PyTorch), otherwise try gputouse (TensorFlow)
    if "device" in param_names:
        device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
        kwargs["device"] = device
        print(f"  device          : {device}")
    elif "gputouse" in param_names:
        kwargs["gputouse"] = args.gpu
        print(f"  gputouse        : {args.gpu}")

    if "batch_size" in param_names:
        kwargs["batch_size"] = args.batch_size

    dlc.video_inference_superanimal(
        videos,
        **kwargs,
    )

    print("✅ SuperAnimal (TF) predictions saved.")
    print("👉 Next: open DLC refine GUI to correct predictions into labels.")


def step_run_superanimal_inference_pt(config_path: Path, args: argparse.Namespace):
    """
    PyTorch SuperAnimal inference using ModelZoo.

    - superanimal_name: "superanimal_quadruped"
    - model_name     : "hrnet_w32"
    - detector_name  : "fasterrcnn_resnet50_fpn_v2"
    """
    import os
    from deeplabcut.utils import auxiliaryfunctions as aux

    print("\n=== Step: video_inference_superanimal (PyTorch ModelZoo) ===")

    # NOTE: deeplabcut is already imported at module level.
    # We still set CUDA_VISIBLE_DEVICES here for downstream torch usage.
    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"🔧 Set CUDA_VISIBLE_DEVICES={args.gpu}")

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            print(f"🔧 Set PyTorch default device to GPU {args.gpu}")
            print(f"🔧 GPU available: {torch.cuda.get_device_name(args.gpu)}")
        else:
            print(f"⚠️  CUDA not available in PyTorch")
    except ImportError:
        print(f"⚠️  PyTorch not available")
    except Exception as e:
        print(f"⚠️  Error setting PyTorch device: {e}")

    cfg = aux.read_config(str(config_path))

    if args.videos:
        videos = [str(Path(v).expanduser().resolve()) for v in args.videos]
    else:
        videos = list(cfg.get("video_sets", {}).keys())

    if not videos:
        raise ValueError("No videos found. Pass --videos or ensure video_sets is defined in config.yaml.")

    dest_folder = Path(config_path).parent / "superanimal_predictions_pt"
    dest_folder.mkdir(parents=True, exist_ok=True)

    superanimal_name = "superanimal_quadruped"
    model_name = "hrnet_w32"
    detector_name = "fasterrcnn_resnet50_fpn_v2"

    print(f"  engine          : pytorch")
    print(f"  superanimal_name: {superanimal_name}")
    print(f"  model_name      : {model_name}")
    print(f"  detector_name   : {detector_name}")
    print(f"  dest_folder     : {dest_folder}")
    print(f"  device          : cuda:{args.gpu} (GPU {args.gpu})")

    # Use device parameter instead of gputouse (PyTorch API)
    device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"

    # Check what parameters the function accepts
    import inspect

    sig = inspect.signature(dlc.video_inference_superanimal)
    param_names = list(sig.parameters.keys())
    print(f"  🔍 video_inference_superanimal accepts: {param_names}")

    # Build kwargs - keep memory under control
    safe_batch_size = min(args.batch_size, 4)
    print(f"  batch_size      : {safe_batch_size} (capped from {args.batch_size} to prevent OOM)")

    # Build call_kwargs with all PyTorch inference parameters
    call_kwargs = {
        "model_name": model_name,
        "detector_name": detector_name,
        "dest_folder": str(dest_folder),
        "batch_size": safe_batch_size,
        "detector_batch_size": args.detector_batch_size,
        
        # Make detector less strict (better for tiny animals)
        "bbox_threshold": args.bbox_threshold,
        "pcutoff": args.pcutoff,
        "pseudo_threshold": args.pseudo_threshold,
        
        # Better for tiny mice in 4K video
        "scale_list": args.scale_list,
        
        # Improve video adaptation
        "video_adapt": args.video_adapt,
        "adapt_iterations": args.adapt_iterations,
        "detector_epochs": args.detector_epochs_inference,
        "pose_epochs": args.pose_epochs_inference,
    }

    # Add device parameter
    if "device" in param_names:
        call_kwargs["device"] = device
        print(f"  ✅ Passing device={device} (GPU {args.gpu})")
    elif "gputouse" in param_names:
        call_kwargs["gputouse"] = args.gpu
        print(f"  ✅ Passing gputouse={args.gpu} (GPU {args.gpu})")
    else:
        print(f"  ⚠️  Function doesn't accept device/gputouse parameter")
    
    # Filter to only include parameters that the function accepts
    final_kwargs = {k: v for k, v in call_kwargs.items() if k in param_names}
    
    # Print all parameters being used
    print(f"\n  🔍 PyTorch inference parameters:")
    print(f"     batch_size: {final_kwargs.get('batch_size', 'N/A')}")
    print(f"     detector_batch_size: {final_kwargs.get('detector_batch_size', 'N/A')}")
    print(f"     bbox_threshold: {final_kwargs.get('bbox_threshold', 'N/A')}")
    print(f"     pcutoff: {final_kwargs.get('pcutoff', 'N/A')}")
    print(f"     pseudo_threshold: {final_kwargs.get('pseudo_threshold', 'N/A')}")
    print(f"     scale_list: {final_kwargs.get('scale_list', 'N/A')}")
    print(f"     video_adapt: {final_kwargs.get('video_adapt', 'N/A')}")
    print(f"     adapt_iterations: {final_kwargs.get('adapt_iterations', 'N/A')}")
    print(f"     detector_epochs: {final_kwargs.get('detector_epochs', 'N/A')}")
    print(f"     pose_epochs: {final_kwargs.get('pose_epochs', 'N/A')}")

    # Show which parameters were skipped (not accepted by function)
    skipped = {k: v for k, v in call_kwargs.items() if k not in param_names}
    if skipped:
        print(f"\n  ⚠️  Parameters not accepted by function (skipped):")
        for key, value in skipped.items():
            print(f"     {key}: {value}")

    dlc.video_inference_superanimal(
        videos,
        superanimal_name,
        **final_kwargs,
    )

    print("✅ SuperAnimal (PyTorch) predictions saved.")
    print("👉 Next: open DLC refine GUI to correct predictions into labels.")

    # Restore CUDA_VISIBLE_DEVICES
    if original_cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]


def step_refine_labels_gui(config_path: Path):
    """
    Launch DLC's refine_labels GUI (single-animal refinement).

    Workflow:
      1. SuperAnimal predictions (H5/CSV) are NOT training labels
      2. This GUI lets you load predictions + frames, correct keypoints
      3. When you SAVE in the GUI, it creates CollectedData_*.csv/.h5 files
      4. Those CollectedData files ARE the training labels you need
    """
    print("\n=== Step: refine_labels (GUI) ===")
    print("⚠️ SuperAnimal predictions ≠ training labels.")
    print("   This GUI converts predictions → labels (CollectedData_*.csv/.h5).")
    print("   You MUST save your refined labels in the GUI.")

    project_dir = config_path.parent
    labeled_dir = project_dir / "labeled-data"
    if not labeled_dir.exists():
        print("⚠️ No 'labeled-data' directory found.")
        print("   You probably need to run --extract-frames and --run-superanimal first.")

    dlc.refine_labels(str(config_path))
    print("✅ Refine GUI closed.")

    collected = list(labeled_dir.rglob("CollectedData_*.csv")) + list(
        labeled_dir.rglob("CollectedData_*.h5")
    )
    if collected:
        print(f"✅ Found {len(collected)} label file(s) - you can now run --create-dataset.")
    else:
        print("⚠️ No CollectedData_*.csv/.h5 files found. Did you SAVE in the GUI?")


def step_check_labels(config_path: Path):
    """Check labels (optional but recommended)."""
    print("\n=== Step: check_labels ===")
    dlc.check_labels(str(config_path))
    print("✅ Labels plot created (inspect figures).")


def step_create_training_dataset(config_path: Path, engine: str):
    """
    Create the training dataset (.mat) used for training.

    For PyTorch:
      - Uses ModelZoo weight initialization (SuperAnimal Quadruped)
      - Creates a top-down HRNet-W32 + Faster R-CNN dataset.
    """
    print("\n=== Step: create_training_dataset ===")

    project_dir = config_path.parent
    labeled_data_dir = project_dir / "labeled-data"

    if not labeled_data_dir.exists():
        print("❌ Error: No 'labeled-data' directory found!")
        print(f"   Expected: {labeled_data_dir}")
        print("   You need to refine predictions first to create labels.")
        sys.exit(1)

    collected = list(labeled_data_dir.rglob("CollectedData_*.csv")) + list(
        labeled_data_dir.rglob("CollectedData_*.h5")
    )
    if not collected:
        print("❌ Error: No CollectedData_*.csv or CollectedData_*.h5 files found!")
        print(f"   Searched in: {labeled_data_dir}")
        print("   SuperAnimal predictions are NOT the same as training labels.")
        print("   Refine in GUI and SAVE to create CollectedData files.")
        sys.exit(1)

    print(f"✅ Found {len(collected)} label file(s) - proceeding with dataset creation")

    if engine == "pytorch":
        # Use ModelZoo weight initialization for transfer learning
        from deeplabcut.modelzoo import build_weight_init

        super_animal = "superanimal_quadruped"
        model_name = "hrnet_w32"
        detector_name = "fasterrcnn_resnet50_fpn_v2"

        print("Using PyTorch ModelZoo weight_init for training dataset (SuperAnimal transfer learning)...")
        print(f"  super_animal : {super_animal}")
        print(f"  model_name   : {model_name}")
        print(f"  detector_name: {detector_name}")

        # Note: cfg is passed as string path; DLC handles reading internally.
        weight_init = build_weight_init(
            cfg=str(config_path),
            super_animal=super_animal,
            model_name=model_name,
            detector_name=detector_name,
            with_decoder=False,  # transfer learning (new decoder), as in docs
        )

        dlc.create_training_dataset(
            str(config_path),
            num_shuffles=1,
            net_type=f"top_down_{model_name}",
            detector_type=detector_name,
            weight_init=weight_init,
            userfeedback=False,
        )
    else:
        # Classic TF path (ImageNet or SuperAnimal via superanimal_name in config)
        dlc.create_training_dataset(str(config_path), num_shuffles=1)

    print("✅ Training dataset created.")


def step_train_network(config_path: Path, args: argparse.Namespace):
    """
    Run real training / fine-tuning.

    - For TensorFlow: uses maxiters/saveiters/displayiters
    - For PyTorch: uses epoch-based API (epochs, save_epochs, device, batch sizes)
    """
    print("\n=== Step: train_network ===")
    if args.engine == "tensorflow":
        print(
            f"[TF] shuffle={args.shuffle}, trainingsetindex={args.trainingsetindex}, "
            f"maxiters={args.maxiters}, displayiters={args.displayiters}, saveiters={args.saveiters}"
        )
        dlc.train_network(
            str(config_path),
            shuffle=args.shuffle,
            trainingsetindex=args.trainingsetindex,
            maxiters=args.maxiters,
            displayiters=args.displayiters,
            saveiters=args.saveiters,
            allow_growth=True,
        )
    else:
        # PyTorch engine: use epoch-based interface
        device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
        print(
            f"[PyTorch] shuffle={args.shuffle}, trainingsetindex={args.trainingsetindex}, "
            f"epochs={args.epochs}, save_epochs={args.save_epochs}, "
            f"batch_size={args.train_batch_size}, detector_batch_size={args.detector_batch_size}, "
            f"detector_epochs={args.detector_epochs}, detector_save_epochs={args.detector_save_epochs}, "
            f"device={device}"
        )

        dlc.train_network(
            str(config_path),
            shuffle=args.shuffle,
            trainingsetindex=args.trainingsetindex,
            device=device,
            batch_size=args.train_batch_size,
            detector_batch_size=args.detector_batch_size,
            epochs=args.epochs,
            save_epochs=args.save_epochs,
            detector_epochs=args.detector_epochs,
            detector_save_epochs=args.detector_save_epochs,
        )

    print("✅ Training finished. Snapshots saved under dlc-models / dlc-models-pytorch.")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    args = parse_args()
    config_path = get_config_path(args)

    # Uncomment if you want a minimal pose model (e.g. head + tailbase only)
    # enforce_minimal_bodyparts(config_path, ["head", "tailbase"])

    # Run SuperAnimal inference
    if args.run_superanimal:
        if args.engine == "tensorflow":
            step_run_superanimal_inference_tf(config_path, args)
        else:
            step_run_superanimal_inference_pt(config_path, args)

    # Extract frames
    if args.extract_frames:
        step_extract_frames(config_path)
        print(
            "\n👉 Next: refine predictions / label frames in the DLC GUI,"
            " then rerun this script with --create-dataset and/or --train."
        )

    # Refine labels in GUI
    if args.refine_labels:
        step_refine_labels_gui(config_path)

    # Create dataset from refined labels
    if args.create_dataset:
        step_check_labels(config_path)
        step_create_training_dataset(config_path, engine=args.engine)

    # Train network
    if args.train:
        project_dir = config_path.parent
        training_datasets_dir = project_dir / "training-datasets"
        if not training_datasets_dir.exists():
            print("❌ Error: Training dataset not found!")
            print(f"   Expected: {training_datasets_dir}")
            print("   Run with --create-dataset first to create the training dataset.")
            sys.exit(1)

        mat_files = list(training_datasets_dir.rglob("*.mat"))
        if not mat_files:
            print("❌ Error: No .mat files found in training-datasets directory!")
            print(f"   Directory: {training_datasets_dir}")
            print("   Run with --create-dataset first to create the training dataset.")
            sys.exit(1)

        step_train_network(config_path, args)

    if not (
        args.extract_frames
        or args.create_dataset
        or args.train
        or args.run_superanimal
        or args.refine_labels
    ):
        print(
            "\nNothing to do: specify at least one of "
            "--run-superanimal, --refine-labels, --extract-frames, "
            "--create-dataset, or --train.\n"
        )


if __name__ == "__main__":
    main()
