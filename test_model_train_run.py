#!/usr/bin/env python
"""Simple script to test trained model workflow (training + inference) with video files.

This script automatically:
    1. Creates a new DLC project
    2. Generates mock labeled data (no manual labeling required!)
    3. Mocks model training (functional/integration test - no actual training occurs)
    4. Runs inference on test videos

Usage:
    1. Configure database (see below)
    2. Put your video file(s) in ./test_videos/ directory (or set DLC_ROOT_DATA_DIR)
       - In Docker: videos should be in ./data/ directory (mounted to /app/data)
    3. Run: python test_model_train_run.py
       - In Docker: make test-trained

    Examples:
        python test_model_train_run.py  # Full workflow: create project, train, infer
        python test_model_train_run.py --skip-training  # Use existing trained model
        python test_model_train_run.py --skip-inference  # Only train, don't infer
        python test_model_train_run.py --model-name my_model  # Custom model name

Database Configuration:
    The script will look for database configuration in this order:
    1. dj_local_conf.json file in the project root
    2. Environment variables: DJ_HOST, DJ_USER, DJ_PASS
    3. Default DataJoint configuration
    
    Example dj_local_conf.json:
    {
        "database.host": "localhost",
        "database.user": "root",
        "database.password": "your_password",
        "database.port": 3306,
        "custom": {
            "database.prefix": "test_"
        }
    }
"""
import os
import sys
import importlib.util
import logging
import argparse
from pathlib import Path
from unittest.mock import patch

import datajoint as dj

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Simple status printer for user-facing messages
class StatusPrinter:
    """Simple status printer for step-by-step progress."""

    def __init__(self, total_steps=10):
        self.total_steps = total_steps
        self.current_step = 0

    def step(self, message, status="info"):
        """Print a step message with status indicator."""
        self.current_step += 1
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "skip": "⏭️",
        }
        icon = icons.get(status, "•")
        print(f"\n[{self.current_step}/{self.total_steps}] {icon} {message}")

    def sub(self, message, indent=3, icon=""):
        """Print a sub-message with indentation."""
        prefix = f"{icon} " if icon else ""
        print(" " * indent + prefix + message)

    def header(self, title):
        """Print a section header."""
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)


# Configure database connection
if Path("./dj_local_conf.json").exists():
    dj.config.load("./dj_local_conf.json")
    logger.info("✅ Loaded database configuration from dj_local_conf.json")
else:
    logger.info("⚠️  No dj_local_conf.json found, using environment variables or defaults")
    logger.info("   Set DJ_HOST, DJ_USER, DJ_PASS environment variables if needed")

# Update config from environment variables
dj.config.update(
    {
        "safemode": False,
        "database.host": os.environ.get("DJ_HOST")
        or dj.config.get("database.host", "localhost"),
        "database.user": os.environ.get("DJ_USER")
        or dj.config.get("database.user", "root"),
        "database.password": os.environ.get("DJ_PASS")
        or dj.config.get("database.password", ""),
    }
)

# Set database prefix for tests
if "custom" not in dj.config:
    dj.config["custom"] = {}
dj.config["custom"]["database.prefix"] = os.environ.get(
    "DATABASE_PREFIX", dj.config["custom"].get("database.prefix", "test_")
)

# Set DLC root data directory if not already set
# In Docker, prefer /app/test_videos (from project mount), otherwise /app/data
# Check for Docker: /.dockerenv exists OR we're in /app directory (Docker working dir)
is_docker = os.path.exists("/.dockerenv") or (
    os.getcwd() == "/app" and os.path.exists("/app")
)
if is_docker:
    # Prefer /app/test_videos (from project mount .:/app) since videos are in ./test_videos
    test_videos_path = Path("/app/test_videos")
    if test_videos_path.exists():
        default_video_dir = "/app/test_videos"
    else:
        default_video_dir = "/app/data"
else:
    default_video_dir = "./test_videos"

video_dir = Path(os.getenv("DLC_ROOT_DATA_DIR", default_video_dir))
# CRITICAL: Set dlc_root_data_dir in DataJoint config to match where videos actually are
# This is used by element-deeplabcut to find video files
if "dlc_root_data_dir" not in dj.config.get("custom", {}) or not dj.config[
    "custom"
].get("dlc_root_data_dir"):
    dj.config["custom"]["dlc_root_data_dir"] = str(video_dir.absolute())
    logger.info(f"📁 Set DLC_ROOT_DATA_DIR to: {video_dir.absolute()}")
    if is_docker:
        logger.info("🐳 Running in Docker mode")

# Get the root directory for making relative paths (ensure it's absolute)
dlc_root_dir = Path(dj.config["custom"].get("dlc_root_data_dir", str(video_dir.absolute())))
if not dlc_root_dir.is_absolute():
    dlc_root_dir = dlc_root_dir.resolve()

logger.info(
    f"📊 Database: {dj.config['database.host']} (prefix: {dj.config['custom']['database.prefix']})"
)
logger.info(f"📁 DLC Root: {dlc_root_dir}")

from element_deeplabcut import model, train
from tests import tutorial_pipeline as pipeline


def check_database_connection():
    """Verify database connection is working."""
    try:
        dj.conn()
        return True
    except Exception as e:
        logger.error(f"\n❌ Database connection failed: {e}")
        logger.error("\nPlease configure your database:")
        logger.error("  1. Create dj_local_conf.json with database credentials")
        logger.error("  2. Or set environment variables: DJ_HOST, DJ_USER, DJ_PASS")
        logger.error(
            "  3. Or ensure database is running (docker compose -f docker-compose-db.yaml up -d)"
        )
        return False


def check_dlc_installation():
    """Check if DeepLabCut is installed and available."""
    try:
        import deeplabcut  # noqa: F401

        return True, None
    except (ImportError, Exception) as e:
        return False, str(e)


def create_dlc_project(project_name, experimenter, video_files, project_dir=None):
    """Create a new DLC project programmatically."""
    import deeplabcut

    if project_dir is None:
        project_dir = dlc_root_dir / project_name

    # Create project directory if it doesn't exist
    project_dir.mkdir(parents=True, exist_ok=True)

    # Convert video files to absolute paths
    video_paths = [str(Path(v).resolve()) for v in video_files]

    # Create DLC project
    config_path = deeplabcut.create_new_project(
        project_name,
        experimenter,
        video_paths,
        working_directory=str(project_dir.parent),
        copy_videos=False,  # Don't copy videos, just reference them
    )

    return Path(config_path).parent  # Return project directory


def create_mock_labeled_data(
    config_path, num_frames=20, bodyparts=None, use_existing_frames=False
):
    """
    Create mock labeled data with images and CSV files for testing.

    IMPORTANT for DLC 3.x:
    - We ONLY create the CSV here.
    - The DataFrame index MUST be a string path like:
      "labeled-data/<video_name>/img000.png"
    - We do NOT create the H5 file ourselves; DLC will call convertcsv2h5
      and build the MultiIndex / tuples internally.
    """
    import pandas as pd
    import numpy as np
    import yaml
    from PIL import Image

    # --- Read config and basic info ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if bodyparts is None:
        bodyparts = config.get("bodyparts", ["nose", "tailbase", "head"])

    # DLC uses "scorer" (DLC 3) or "experimenter" (older)
    scorer = config.get("scorer") or config.get("experimenter") or "experimenter"

    project_path = Path(config_path).parent
    labeled_data_dir = project_path / "labeled-data"
    labeled_data_dir.mkdir(parents=True, exist_ok=True)

    # Choose video name
    video_sets = config.get("video_sets", {})
    if video_sets:
        video_name = Path(list(video_sets.keys())[0]).stem
    else:
        video_name = "test_video"

    video_labeled_dir = labeled_data_dir / video_name
    video_labeled_dir.mkdir(parents=True, exist_ok=True)

    # Image size
    img_width = config.get("im_width", 640)
    img_height = config.get("im_height", 480)

    # --- Create / reuse images ---
    existing_images = sorted(
        list(video_labeled_dir.glob("*.png")) + list(video_labeled_dir.glob("*.jpg"))
    )

    if use_existing_frames and existing_images:
        logger.info(f"Using {len(existing_images)} existing frame(s) extracted by DLC")
        num_frames = min(num_frames, len(existing_images))
        img_files = existing_images[:num_frames]
    else:
        img_files = []
        for frame_idx in range(num_frames):
            img = Image.new(
                "RGB",
                (img_width, img_height),
                color=(
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                ),
            )
            img_path = video_labeled_dir / f"img{frame_idx:03d}.png"
            img.save(img_path)
            img_files.append(img_path)

        logger.info(f"Created {len(img_files)} mock image files in {video_labeled_dir}")

    # --- Build DataFrame in DLC format ---

    # MultiIndex columns: (scorer, bodypart, coord)
    columns = []
    for bp in bodyparts:
        columns.append((scorer, bp, "x"))
        columns.append((scorer, bp, "y"))
        columns.append((scorer, bp, "likelihood"))

    data = []
    index_strings = []

    for img_path in img_files:
        row = []
        for bp in bodyparts:
            x = np.random.uniform(50, img_width - 50)
            y = np.random.uniform(50, img_height - 50)
            likelihood = np.random.uniform(0.8, 1.0)
            row.extend([x, y, likelihood])
        data.append(row)

        # IMPORTANT:
        # DLC 3.x will later split this string into path components and
        # build its own MultiIndex. Here we just give it a clean relative path.
        rel_str = f"labeled-data/{video_name}/{img_path.name}"
        index_strings.append(rel_str)

    df = pd.DataFrame(
        data,
        columns=pd.MultiIndex.from_tuples(
            columns, names=["scorer", "bodyparts", "coords"]
        ),
        index=index_strings,
    )

    csv_filename = f"CollectedData_{scorer}.csv"
    csv_path = video_labeled_dir / csv_filename

    # Save CSV with index: index contains the relative image path as string
    df.to_csv(csv_path, index=True)
    logger.info(f"Created CSV file: {csv_path}")

    # DO NOT create .h5 here. Let DLC handle convertcsv2h5 internally.
    # If an old .h5 exists from previous runs, it can confuse DLC, so we remove it:
    h5_path = video_labeled_dir / csv_filename.replace(".csv", ".h5")
    if h5_path.exists():
        logger.info(f"Removing old H5 file (will be regenerated by DLC): {h5_path}")
        h5_path.unlink()

    # Sanity checks
    if not csv_path.exists():
        raise FileNotFoundError(f"Failed to create CSV file: {csv_path}")

    final_imgs = list(video_labeled_dir.glob("img*.png"))
    if len(final_imgs) == 0:
        raise FileNotFoundError(
            f"No image files found in {video_labeled_dir} after mock data creation."
        )

    logger.info(f"Created mock labeled data: {csv_path}")
    logger.info(f"  - Directory: {video_labeled_dir}")
    logger.info(f"  - {len(final_imgs)} frames")
    logger.info(f"  - {len(bodyparts)} body parts: {bodyparts}")
    logger.info(f"  - Scorer: {scorer}")
    logger.info(f"  - Video name: {video_name}")

    return csv_path


def main():
    training_was_mocked = False
    status = StatusPrinter(total_steps=12)
    status.header("Testing Trained Model Workflow (Training + Inference)")

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Test trained DeepLabCut workflow with video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dlc_project",
        nargs="?",
        default=None,
        help="Path to DLC project directory or config.yaml file",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training step and use existing trained model",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference step (only train the model)",
    )
    parser.add_argument(
        "--mock-results-on-failure",
        action="store_true",
        help="If inference fails or no animals detected, insert mock pose estimation results instead of failing. Useful for testing when videos don't contain detectable animals.",
    )
    parser.add_argument(
        "--model-name",
        default="test_trained_model",
        help="Name for the trained model (default: test_trained_model)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (default: 0). Use -1 for CPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Batch size for pose estimation inference (default: 4). Reduce if you get CUDA OOM errors. "
            "Increase if you have more GPU memory."
        ),
    )

    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES for PyTorch/TensorFlow
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"🔧 Set CUDA_VISIBLE_DEVICES={args.gpu}")

        # Configure PyTorch device if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_device(args.gpu)
                print(f"🔧 Set PyTorch default device to GPU {args.gpu}")
                print(f"🔧 GPU available: {torch.cuda.get_device_name(args.gpu)}")
            else:
                print("⚠️  CUDA not available in PyTorch")
        except ImportError:
            pass  # PyTorch not available, that's OK

    # 0. Check database connection
    status.step("Checking database connection")
    if not check_database_connection():
        sys.exit(1)
    status.sub("Database connection successful", indent=3)

    # 0.5. Clean up database (remove test data from previous runs)
    status.step("Cleaning up database (removing test data from previous runs)")
    cleanup_count = 0

    # Delete PoseEstimation entries (and their parts)
    pose_estimation_rel = pipeline.model.PoseEstimation
    pose_keys = pose_estimation_rel.fetch("KEY")
    pose_count = len(pose_keys)
    if pose_count > 0:
        pose_estimation_rel.delete()
        cleanup_count += pose_count
        status.sub(
            f"Deleted {pose_count} PoseEstimation entry/entries",
            icon="🗑️",
            indent=3,
        )

    # Delete PoseEstimationTask entries
    task_rel = pipeline.model.PoseEstimationTask
    task_keys = task_rel.fetch("KEY")
    task_count = len(task_keys)
    if task_count > 0:
        task_rel.delete()
        cleanup_count += task_count
        status.sub(
            f"Deleted {task_count} PoseEstimationTask entry/entries",
            icon="🗑️",
            indent=3,
        )

    # Delete test models (models with names starting with "test_")
    test_models_rel = model.Model & "model_name LIKE 'test_%'"
    test_model_keys = test_models_rel.fetch("KEY")
    test_model_count = len(test_model_keys)
    if test_model_count > 0:
        test_models_rel.delete()
        cleanup_count += test_model_count
        status.sub(f"Deleted {test_model_count} test model(s)", icon="🗑️", indent=3)

    # Delete test training tasks (optional - comment out if you want to keep training history)
    training_rel = pipeline.train.ModelTraining
    training_keys = training_rel.fetch("KEY")
    training_count = len(training_keys)
    if training_count > 0:
        training_rel.delete()
        cleanup_count += training_count
        status.sub(
            f"Deleted {training_count} ModelTraining entry/entries",
            icon="🗑️",
            indent=3,
        )

    if cleanup_count > 0:
        status.sub(f"Total: {cleanup_count} entry/entries cleaned", icon="✅", indent=3)
    else:
        status.sub("No test data found to clean", icon="ℹ️", indent=3)

    # 1. Check DeepLabCut installation
    status.step("Checking DeepLabCut installation")
    dlc_available, dlc_error = check_dlc_installation()
    if not dlc_available:
        status.sub(
            f"DeepLabCut is not installed or not importable: {dlc_error}",
            icon="❌",
            indent=3,
        )
        status.sub("Install with: pip install 'deeplabcut[superanimal]'", indent=5)
        sys.exit(1)
    else:
        import deeplabcut

        status.sub(
            f"DeepLabCut is available (version {deeplabcut.__version__})",
            icon="✅",
            indent=3,
        )

    # 2. Create or find DLC project
    status.step("Creating DLC project")

    # Find video files for project creation (use first available video or create dummy)
    is_docker_local = os.path.exists("/.dockerenv") or (
        os.getcwd() == "/app" and os.path.exists("/app")
    )
    if is_docker_local:
        test_videos_path = Path("/app/test_videos")
        if test_videos_path.exists():
            default_video_dir_local = "/app/test_videos"
        else:
            default_video_dir_local = "/app/data"
    else:
        default_video_dir_local = "./test_videos"

    video_dir_local = Path(os.getenv("DLC_ROOT_DATA_DIR", default_video_dir_local))
    video_files = list(video_dir_local.glob("*.mp4")) + list(
        video_dir_local.glob("*.avi")
    ) + list(video_dir_local.glob("*.mov"))

    if not video_files:
        # Create a dummy video file path (DLC will handle missing videos gracefully)
        video_files = [str(video_dir_local / "dummy_video.mp4")]
        status.sub(
            "No videos found - will create project with dummy video path",
            icon="⚠️",
            indent=3,
        )
        status.sub("(You can add real videos later)", icon="ℹ️", indent=5)

    # Create project name
    project_name = f"test_training_project_{args.model_name}"
    experimenter = "test_experimenter"

    # Check if project already exists
    project_dir = dlc_root_dir / project_name
    config_file = project_dir / "config.yaml"

    if config_file.exists() and not args.skip_training:
        status.sub(f"Project already exists: {project_dir}", icon="ℹ️", indent=3)
        status.sub("Using existing project (delete it to recreate)", icon="ℹ️", indent=5)
        dlc_project_path = project_dir
    else:
        # Create new project
        status.sub(f"Creating new DLC project: {project_name}", icon="ℹ️", indent=3)
        try:
            dlc_project_path = create_dlc_project(
                project_name,
                experimenter,
                video_files[:1],  # Use first video for project creation
                project_dir=project_dir,
            )
            status.sub(f"Project created: {dlc_project_path}", icon="✅", indent=5)
        except Exception as e:
            status.sub(f"Error creating project: {e}", icon="❌", indent=3)
            raise

    config_file = dlc_project_path / "config.yaml"
    if not config_file.exists():
        status.sub(f"Config file not found: {config_file}", icon="❌", indent=3)
        sys.exit(1)

    # Make path relative to dlc_root_dir
    try:
        dlc_project_rel = Path(dlc_project_path).relative_to(dlc_root_dir)
    except ValueError:
        dlc_project_rel = Path(dlc_project_path)
        status.sub(
            "Warning: Project path not under DLC_ROOT_DATA_DIR, using absolute path",
            icon="⚠️",
            indent=3,
        )

    config_file_rel = dlc_project_rel / "config.yaml"
    status.sub(f"Project path: {dlc_project_path}", icon="✅", indent=3)
    status.sub(f"Config file: {config_file_rel}", icon="ℹ️", indent=5)

    # 3. Create mock labeled data (skip manual labeling)
    if not args.skip_training:
        status.step("Creating mock labeled data")
        labeled_data_dir = dlc_project_path / "labeled-data"

        # Check if labeled data already exists
        if labeled_data_dir.exists() and any(labeled_data_dir.iterdir()):
            status.sub(
                "Labeled data already exists, skipping mock data creation",
                icon="ℹ️",
                indent=3,
            )
        else:
            import yaml

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            bodyparts = config.get("bodyparts", ["nose", "tailbase", "head"])
            status.sub(
                f"Creating mock labeled data with {len(bodyparts)} body parts",
                icon="ℹ️",
                indent=3,
            )
            status.sub(f"Body parts: {bodyparts}", icon="ℹ️", indent=5)

            try:
                create_mock_labeled_data(
                    config_file, num_frames=20, bodyparts=bodyparts
                )
                status.sub("Mock labeled data created successfully", icon="✅", indent=3)
            except Exception as e:
                status.sub(f"Error creating mock labeled data: {e}", icon="❌", indent=3)
                raise

    # 4. Setup test data (subject, session, recordings)
    status.step("Setting up test data")
    base_key = {
        "subject": "test1",
        "session_datetime": "2024-01-01 12:00:00",
    }

    pipeline.subject.Subject.insert1(
        {
            "subject": "test1",
            "sex": "F",
            "subject_birth_date": "2020-01-01",
            "subject_description": "Test subject for trained model workflow",
        },
        skip_duplicates=True,
    )

    pipeline.session.Session.insert1(
        {
            "subject": "test1",
            "session_datetime": "2024-01-01 12:00:00",
        },
        skip_duplicates=True,
    )

    # Find video files for inference (reuse from earlier or find again)
    if not video_files:
        video_files = list(video_dir_local.glob("*.mp4")) + list(
            video_dir_local.glob("*.avi")
        ) + list(video_dir_local.glob("*.mov"))
    elif video_files and len(video_files) > 0:
        first_video = str(video_files[0])
        if "dummy_video.mp4" in first_video:
            video_files = list(video_dir_local.glob("*.mp4")) + list(
                video_dir_local.glob("*.avi")
            ) + list(video_dir_local.glob("*.mov"))

    if not video_files and not args.skip_inference:
        status.sub(f"No video files found in {video_dir_local}", icon="⚠️", indent=3)
        status.sub("Supported formats: .mp4, .avi, .mov", icon="ℹ️", indent=3)
        status.sub(
            "Set DLC_ROOT_DATA_DIR environment variable to point to video directory",
            indent=3,
        )
        if args.skip_training:
            sys.exit(1)
        else:
            status.sub(
                "Continuing with training only (no inference videos)", icon="ℹ️", indent=3
            )

    # Create recordings for inference videos
    recording_keys = []
    if video_files:
        for idx, video_file in enumerate(video_files):
            recording_key = {
                **base_key,
                "recording_id": idx + 1,
            }
            recording_keys.append(recording_key)

            # Insert or update recording
            existing_rec = pipeline.model.VideoRecording & recording_key
            if len(existing_rec):
                existing_device = existing_rec.fetch1("device")
                if existing_device != "Camera1":
                    existing_rec.delete()
                    pipeline.model.VideoRecording.insert1(
                        {**recording_key, "device": "Camera1"}
                    )
                    status.sub(
                        f"Updated existing recording {recording_key['recording_id']} (device changed)",
                        icon="🔄",
                        indent=5,
                    )
                else:
                    status.sub(
                        f"Recording {recording_key['recording_id']} already exists with correct device",
                        icon="✅",
                        indent=5,
                    )
            else:
                pipeline.model.VideoRecording.insert1(
                    {**recording_key, "device": "Camera1"}
                )
                status.sub(
                    f"Created new recording {recording_key['recording_id']}",
                    icon="✅",
                    indent=5,
                )

            video_file_abs = Path(video_file).resolve()
            try:
                relative_path = video_file_abs.relative_to(dlc_root_dir)
            except ValueError:
                relative_path = Path(video_file.name)

            # Insert or update file entry
            file_key = {**recording_key, "file_id": 0}
            existing_file = pipeline.model.VideoRecording.File & file_key
            if len(existing_file):
                existing_path = existing_file.fetch1("file_path")
                if existing_path != str(relative_path):
                    file_keys = (
                        pipeline.model.VideoRecording.File & recording_key
                    ).fetch("KEY", as_dict=True)
                    all_files = []
                    for fk in file_keys:
                        file_data = (
                            pipeline.model.VideoRecording.File & fk
                        ).fetch1()
                        all_files.append(file_data)
                    (pipeline.model.VideoRecording & recording_key).delete()
                    pipeline.model.VideoRecording.insert1(
                        {**recording_key, "device": "Camera1"}
                    )
                    for file_entry in all_files:
                        if file_entry["file_id"] == 0:
                            pipeline.model.VideoRecording.File.insert1(
                                {
                                    **recording_key,
                                    "file_id": 0,
                                    "file_path": str(relative_path),
                                }
                            )
                        else:
                            pipeline.model.VideoRecording.File.insert1(file_entry)
                    status.sub(
                        f"Updated file path for recording {recording_key['recording_id']}: {existing_path} -> {video_file_abs.name}",
                        icon="🔄",
                        indent=5,
                    )
                else:
                    status.sub(
                        f"File path for recording {recording_key['recording_id']} is already correct: {video_file_abs.name}",
                        icon="✅",
                        indent=5,
                    )
            else:
                pipeline.model.VideoRecording.File.insert1(
                    {**file_key, "file_path": str(relative_path)}
                )
                status.sub(
                    f"Created file entry for recording {recording_key['recording_id']}: {video_file_abs.name}",
                    icon="✅",
                    indent=5,
                )

    status.sub(
        f"Created {len(recording_keys)} recording(s) for inference", icon="✅", indent=3
    )

    # 5. Extract video metadata (if videos exist)
    if recording_keys:
        status.step("Extracting video metadata")
        try:
            pipeline.model.RecordingInfo.populate()
            for rec_key in recording_keys:
                rec_info = (pipeline.model.RecordingInfo & rec_key).fetch1()
                status.sub(
                    f"Recording {rec_key['recording_id']}: {rec_info['px_width']}x{rec_info['px_height']}, "
                    f"{rec_info['nframes']} frames, {rec_info['fps']:.1f} fps",
                    icon="✅",
                    indent=5,
                )
        except ModuleNotFoundError as e:
            if "cv2" in str(e):
                status.sub(
                    "OpenCV (cv2) is required for video metadata extraction",
                    icon="⚠️",
                    indent=3,
                )
                status.sub("Install with: pip install opencv-python", indent=5)
            else:
                raise

    # 6. Training workflow (if not skipped)
    model_name = args.model_name

    if not args.skip_training:
        status.step("Setting up training workflow")

        # Check if training has already been done
        if len(model.Model & {"model_name": model_name}):
            status.sub(f"Model '{model_name}' already exists", icon="ℹ️", indent=3)
            status.sub(
                "Use --skip-training to use existing model, or choose different --model-name",
                indent=5,
            )
            if not args.skip_inference:
                status.sub(
                    "Skipping training, proceeding to inference...",
                    icon="⏭️",
                    indent=3,
                )
                args.skip_training = True
        else:
            import yaml

            with open(config_file, "r") as f:
                dlc_config = yaml.safe_load(f)

            # Truncate Task field early to fit varchar(32) constraint
            if "Task" in dlc_config and len(dlc_config["Task"]) > 32:
                original_task = dlc_config["Task"]
                dlc_config["Task"] = original_task[:32]
                status.sub(
                    f"Truncated Task field from {len(original_task)} to 32 chars: '{dlc_config['Task']}'",
                    icon="ℹ️",
                    indent=3,
                )
                with open(config_file, "w") as f:
                    yaml.dump(dlc_config, f, default_flow_style=False)
                status.sub("Config file updated with truncated Task", icon="✅", indent=5)

            # Check if project has labeled data
            labeled_data_dir = dlc_project_path / "labeled-data"

            # Remove old H5 files - will be regenerated by DLC
            old_h5_files = (
                list(labeled_data_dir.rglob("*.h5"))
                if labeled_data_dir.exists()
                else []
            )
            if old_h5_files:
                status.sub(
                    f"Removing {len(old_h5_files)} old H5 file(s) (will be regenerated by DLC)",
                    icon="⚠️",
                    indent=3,
                )
                for h5_file in old_h5_files:
                    h5_file.unlink()
                status.sub("Old H5 files removed", icon="✅", indent=5)

            labeled_files = (
                list(labeled_data_dir.rglob("*.csv"))
                if labeled_data_dir.exists()
                else []
            )

            if len(labeled_files) == 0:
                status.sub(
                    "No labeled data found - creating mock labeled data",
                    icon="⚠️",
                    indent=3,
                )
                try:
                    csv_path = create_mock_labeled_data(
                        config_file, num_frames=20, use_existing_frames=False
                    )
                    status.sub(
                        f"Mock labeled data created: {csv_path}", icon="✅", indent=5
                    )
                    labeled_files = list(labeled_data_dir.rglob("*.csv"))
                    img_files = list(labeled_data_dir.rglob("*.png")) + list(
                        labeled_data_dir.rglob("*.jpg")
                    )
                    if len(labeled_files) == 0:
                        raise ValueError("Failed to create labeled data files")
                    status.sub(
                        f"Verified {len(labeled_files)} CSV file(s) and {len(img_files)} image file(s)",
                        icon="✅",
                        indent=5,
                    )
                except Exception as e:
                    status.sub(
                        f"Error creating mock labeled data: {e}", icon="❌", indent=3
                    )
                    import traceback

                    status.sub(traceback.format_exc(), indent=5)
                    raise
            else:
                status.sub(
                    f"Found {len(labeled_files)} existing CSV file(s) (H5 will be generated by DLC)",
                    icon="ℹ️",
                    indent=3,
                )

            # Remove old training artifacts
            training_datasets_dir = dlc_project_path / "training-datasets"
            if training_datasets_dir.exists():
                status.sub(
                    "Training dataset exists; deleting to regenerate with current DLC...",
                    icon="⚠️",
                    indent=3,
                )
                import shutil

                shutil.rmtree(training_datasets_dir)
                status.sub(
                    "Deleted old training-datasets directory", icon="✅", indent=5
                )

            dlc_models_dir = dlc_project_path / "dlc-models"
            if dlc_models_dir.exists():
                status.sub(
                    "Cleaning up old dlc-models directory...", icon="ℹ️", indent=5
                )
                import shutil

                shutil.rmtree(dlc_models_dir)
                status.sub("Deleted old dlc-models directory", icon="✅", indent=5)

            dlc_models_pytorch_dir = dlc_project_path / "dlc-models-pytorch"
            if dlc_models_pytorch_dir.exists():
                status.sub(
                    "Cleaning up old dlc-models-pytorch directory...",
                    icon="ℹ️",
                    indent=5,
                )
                import shutil

                shutil.rmtree(dlc_models_pytorch_dir)
                status.sub(
                    "Deleted old dlc-models-pytorch directory", icon="✅", indent=5
                )

            # Simple training parameters
            shuffle = 1
            trainingsetindex = 0

            status.sub(
                "Creating training dataset with deeplabcut.create_training_dataset",
                icon="ℹ️",
                indent=3,
            )
            with open(config_file, "r") as f:
                create_config = yaml.safe_load(f)

            # Ensure engine is pytorch in config
            old_engine = create_config.get("engine", "not set")
            create_config["engine"] = "pytorch"
            with open(config_file, "w") as f:
                yaml.dump(create_config, f, default_flow_style=False)

            status.sub(
                f"Config updated: engine={create_config.get('engine')} (was {old_engine})",
                icon="ℹ️",
                indent=5,
            )

            try:
                import deeplabcut
                import pandas as pd

                dlc_version = deeplabcut.__version__
                status.sub(f"Using DLC version: {dlc_version}", icon="ℹ️", indent=5)

                status.sub("Converting CSV to H5 format...", icon="ℹ️", indent=5)
                try:
                    deeplabcut.convertcsv2h5(str(config_file), userfeedback=False)

                    labeled_data_dir = Path(config_file).parent / "labeled-data"
                    h5_files = list(labeled_data_dir.rglob("CollectedData_*.h5"))

                    for h5_file in h5_files:
                        df = pd.read_hdf(h5_file, key="df_with_missing")

                        # Fix index to be strings if needed
                        if isinstance(df.index, pd.MultiIndex) or (
                            len(df.index) > 0 and isinstance(df.index[0], tuple)
                        ):
                            df.index = [
                                "/".join(str(x) for x in idx)
                                if isinstance(idx, tuple)
                                else str(idx)
                                for idx in df.index
                            ]

                        # Backup full file then create x,y-only version
                        h5_file_backup = (
                            h5_file.parent / f"{h5_file.stem}_full_backup.h5"
                        )
                        df.to_hdf(
                            h5_file_backup,
                            key="df_with_missing",
                            mode="w",
                            format="table",
                        )

                        if isinstance(df.columns, pd.MultiIndex):
                            xy_cols = [
                                col for col in df.columns if col[2] in ["x", "y"]
                            ]
                            if xy_cols:
                                df_xy = df[xy_cols].copy()
                                df_xy.to_hdf(
                                    h5_file,
                                    key="df_with_missing",
                                    mode="w",
                                    format="table",
                                )
                                status.sub(
                                    f"Created x, y only version for training dataset: {h5_file.name}",
                                    icon="✅",
                                    indent=7,
                                )

                    status.sub(
                        "CSV converted to H5 successfully (x, y only for training dataset)",
                        icon="✅",
                        indent=7,
                    )
                except Exception as e:
                    status.sub(f"convertcsv2h5 failed: {e}", icon="⚠️", indent=7)
                    status.sub(
                        "Attempting manual CSV to H5 conversion...", icon="ℹ️", indent=7
                    )
                    try:
                        import yaml

                        with open(config_file, "r") as f:
                            cfg = yaml.safe_load(f)

                        labeled_data_dir = Path(config_file).parent / "labeled-data"
                        csv_files = list(labeled_data_dir.rglob("CollectedData_*.csv"))

                        for csv_file in csv_files:
                            df = pd.read_csv(
                                csv_file, index_col=0, header=[0, 1, 2]
                            )

                            if isinstance(df.columns, pd.MultiIndex):
                                new_columns = [col for col in df.columns]
                                df.columns = pd.MultiIndex.from_tuples(
                                    new_columns,
                                    names=["scorer", "bodyparts", "coords"],
                                )
                            else:
                                raise ValueError(
                                    f"Expected MultiIndex columns, got {type(df.columns)}"
                                )

                            if not all(isinstance(idx, str) for idx in df.index):
                                df.index = [str(idx) for idx in df.index]

                            h5_file = csv_file.with_suffix(".h5")
                            df.to_hdf(
                                h5_file,
                                key="df_with_missing",
                                mode="w",
                                format="table",
                            )
                            status.sub(
                                f"Created {h5_file.name} with correct MultiIndex structure",
                                icon="✅",
                                indent=9,
                            )

                        status.sub(
                            "Manual CSV to H5 conversion completed", icon="✅", indent=7
                        )
                    except Exception as e2:
                        status.sub(
                            f"Manual conversion also failed: {e2}",
                            icon="⚠️",
                            indent=7,
                        )
                        import traceback

                        status.sub(traceback.format_exc(), indent=9)
                        status.sub(
                            "Continuing - DLC may handle conversion internally",
                            icon="ℹ️",
                            indent=7,
                        )

                status.sub(
                    "Calling deeplabcut.create_training_dataset(..., num_shuffles=1)",
                    icon="ℹ️",
                    indent=5,
                )

                deeplabcut.create_training_dataset(str(config_file), num_shuffles=1)
                status.sub(
                    "Training dataset created successfully by DLC", icon="✅", indent=5
                )

                # Restore full H5 files (with likelihood) from backup
                labeled_data_dir = Path(config_file).parent / "labeled-data"
                backup_files = list(labeled_data_dir.rglob("*_full_backup.h5"))
                for backup_file in backup_files:
                    original_file = backup_file.parent / backup_file.name.replace(
                        "_full_backup.h5", ".h5"
                    )
                    if original_file.exists():
                        df_full = pd.read_hdf(
                            backup_file, key="df_with_missing"
                        )
                        df_full.to_hdf(
                            original_file,
                            key="df_with_missing",
                            mode="w",
                            format="table",
                        )
                        backup_file.unlink()
                        status.sub(
                            f"Restored full H5 file (with likelihood): {original_file.name}",
                            icon="✅",
                            indent=7,
                        )

                metadata_files = list(training_datasets_dir.rglob("metadata.yaml"))
                if not metadata_files:
                    raise FileNotFoundError(
                        "Training dataset metadata file not found after create_training_dataset"
                    )
                status.sub(
                    f"Found metadata file: {metadata_files[0]}",
                    icon="ℹ️",
                    indent=5,
                )

            except Exception as e:
                error_str = str(e)
                if (
                    "all the input array dimensions" in error_str
                    and "size 4" in error_str
                    and "size 6" in error_str
                ):
                    status.sub(
                        "Known DLC 3.x bug detected in format_training_data",
                        icon="⚠️",
                        indent=3,
                    )
                status.sub(f"Error creating training dataset: {e}", icon="❌", indent=3)
                import traceback

                status.sub(traceback.format_exc(), indent=5)
                status.sub(
                    "This appears to be a bug in DLC 3.x's format_training_data function.",
                    indent=5,
                )
                sys.exit(1)

            # Ensure VideoSet exists
            video_set_key = {"video_set_id": 1}
            if not len(train.VideoSet & video_set_key):
                train.VideoSet.insert1(video_set_key, skip_duplicates=True)

            # TrainingParamSet
            paramset_key = {"paramset_idx": 0}
            if not len(train.TrainingParamSet & paramset_key):
                default_params = {
                    "shuffle": shuffle,
                    "trainingsetindex": trainingsetindex,
                    "maxiters": dlc_config.get("maxiters", 5000),
                    "displayiters": dlc_config.get("displayiters", 100),
                    "saveiters": dlc_config.get("saveiters", 1000),
                }
                train.TrainingParamSet.insert_new_params(
                    paramset_desc="Default training parameters",
                    params=default_params,
                    paramset_idx=0,
                )

            # TrainingTask
            training_id = 1
            training_task_key = {
                **video_set_key,
                **paramset_key,
                "training_id": training_id,
            }

            if len(train.TrainingTask & training_task_key):
                old_task = (train.TrainingTask & training_task_key).fetch1()
                if old_task["project_path"] != str(dlc_project_rel):
                    status.sub(
                        f"Deleting old training task with wrong project path: {old_task['project_path']}",
                        icon="⚠️",
                        indent=3,
                    )
                    (train.TrainingTask & training_task_key).delete()
                    status.sub("Old training task deleted", icon="✅", indent=5)

            if not len(train.TrainingTask & training_task_key):
                train.TrainingTask.insert1(
                    {
                        **training_task_key,
                        "model_prefix": "",
                        "project_path": str(dlc_project_rel),
                    },
                    skip_duplicates=True,
                )
                status.sub("Created training task", icon="✅", indent=3)
            else:
                status.sub(
                    "Training task already exists with correct project path",
                    icon="ℹ️",
                    indent=3,
                )

            # 7. Train the model (mocked for functional test)
            status.step("Training the model (mocked)")
            status.sub("Training is mocked for this functional test", icon="ℹ️", indent=3)
            status.sub("No actual model training will occur", icon="ℹ️", indent=3)

            training_was_mocked = True

            def create_mock_pytorch_config(config_path, project_path, dlc_config):
                """Create a mock pytorch_config.yaml file for inference."""
                import yaml

                bodyparts = dlc_config.get(
                    "bodyparts", ["bodypart1", "bodypart2", "bodypart3"]
                )
                unique_bodyparts = dlc_config.get("uniquebodyparts", [])
                individuals = dlc_config.get("individuals", [])
                multianimal = dlc_config.get("multianimalproject", False)

                pytorch_config = {
                    "data": {
                        "bbox_margin": 20,
                        "colormode": "RGB",
                        "inference": {"normalize_images": True},
                        "train": {
                            "affine": {
                                "p": 0.5,
                                "rotation": 30,
                                "scaling": [0.5, 1.25],
                                "translation": 0,
                            },
                            "crop_sampling": {
                                "width": 448,
                                "height": 448,
                                "max_shift": 0.1,
                                "method": "hybrid",
                            },
                            "gaussian_noise": 12.75,
                            "motion_blur": True,
                            "normalize_images": True,
                        },
                    },
                    "device": "auto",
                    "inference": {
                        "multithreading": {
                            "enabled": True,
                            "queue_length": 4,
                            "timeout": 30.0,
                        },
                        "compile": {"enabled": False, "backend": "inductor"},
                        "autocast": {"enabled": False},
                    },
                    "metadata": {
                        "project_path": str(project_path),
                        "pose_config_path": str(config_path),
                        "bodyparts": bodyparts,
                        "unique_bodyparts": unique_bodyparts,
                        "individuals": individuals if multianimal else [],
                        "with_identity": multianimal,
                    },
                    "method": "bu",
                    "model": {
                        "backbone": {
                            "type": "ResNet",
                            "model_name": "resnet50_gn",
                            "output_stride": 16,
                            "freeze_bn_stats": False,
                            "freeze_bn_weights": False,
                        },
                        "backbone_output_channels": 2048,
                        "heads": {
                            "bodypart": {
                                "type": "HeatmapHead",
                                "weight_init": "normal",
                                "predictor": {
                                    "type": "HeatmapPredictor",
                                    "apply_sigmoid": False,
                                    "clip_scores": True,
                                    "location_refinement": True,
                                    "locref_std": 7.2801,
                                },
                                "target_generator": {
                                    "type": "HeatmapGaussianGenerator",
                                    "num_heatmaps": len(bodyparts),
                                    "pos_dist_thresh": 17,
                                    "heatmap_mode": "KEYPOINT",
                                    "gradient_masking": False,
                                    "generate_locref": True,
                                    "locref_std": 7.2801,
                                },
                                "criterion": {
                                    "heatmap": {
                                        "type": "WeightedMSECriterion",
                                        "weight": 1.0,
                                    },
                                    "locref": {
                                        "type": "WeightedHuberCriterion",
                                        "weight": 0.05,
                                    },
                                },
                                "heatmap_config": {
                                    "channels": [2048, len(bodyparts)],
                                    "kernel_size": [3],
                                    "strides": [2],
                                },
                                "locref_config": {
                                    "channels": [2048, len(bodyparts) * 2],
                                    "kernel_size": [3],
                                    "strides": [2],
                                },
                            }
                        },
                    },
                    "net_type": "resnet_50",
                    "runner": {
                        "type": "PoseTrainingRunner",
                        "gpus": [],
                        "key_metric": "test.mAP",
                        "key_metric_asc": True,
                        "eval_interval": 10,
                        "optimizer": {
                            "type": "AdamW",
                            "params": {"lr": 0.0005},
                        },
                        "scheduler": {
                            "type": "LRListScheduler",
                            "params": {
                                "lr_list": [[0.0001], [1e-05]],
                                "milestones": [90, 120],
                            },
                        },
                        "snapshots": {
                            "max_snapshots": 5,
                            "save_epochs": 25,
                            "save_optimizer_state": False,
                        },
                    },
                    "train_settings": {
                        "batch_size": 8,
                        "dataloader_workers": 0,
                        "dataloader_pin_memory": False,
                        "display_iters": 500,
                        "epochs": 200,
                        "seed": 42,
                    },
                }

                with open(config_path, "w") as f:
                    yaml.dump(pytorch_config, f, default_flow_style=False, sort_keys=False)

            try:
                # Store original make method
                original_make = pipeline.train.ModelTraining.make

                def mocked_make(self, key):
                    """Mocked make() that skips training but creates snapshot file."""
                    from pathlib import Path
                    import yaml as _yaml
                    from element_interface.utils import find_full_path
                    from element_deeplabcut.train import (
                        get_dlc_root_data_dir,
                        TrainingTask,
                        TrainingParamSet,
                    )
                    from element_deeplabcut.readers import dlc_reader as _dlc_reader

                    project_path, model_prefix = (TrainingTask & key).fetch1(
                        "project_path", "model_prefix"
                    )
                    project_path = find_full_path(
                        get_dlc_root_data_dir(), project_path
                    )
                    project_path = Path(project_path)
                    if project_path.is_file():
                        project_path = project_path.parent

                    _, dlc_config_local = _dlc_reader.read_yaml(project_path)
                    training_params = (TrainingParamSet & key).fetch1("params")
                    shuffle_local = training_params.get(
                        "shuffle", dlc_config_local.get("shuffle", 1)
                    )
                    trainingsetindex_local = training_params.get(
                        "trainingsetindex", dlc_config_local.get("trainingsetindex", 0)
                    )

                    dlc_config_local["shuffle"] = int(shuffle_local)
                    dlc_config_local["trainingsetindex"] = int(trainingsetindex_local)
                    train_fraction = dlc_config_local["TrainingFraction"][
                        int(trainingsetindex_local)
                    ]
                    dlc_config_local["train_fraction"] = train_fraction
                    dlc_config_local["project_path"] = project_path.as_posix()
                    dlc_config_local["modelprefix"] = model_prefix

                    try:
                        from deeplabcut.utils.auxiliaryfunctions import (
                            get_model_folder,
                        )
                    except ImportError:
                        from deeplabcut.utils.auxiliaryfunctions import (
                            GetModelFolder as get_model_folder,
                        )

                    model_folder = get_model_folder(
                        trainFraction=train_fraction,
                        shuffle=int(shuffle_local),
                        cfg=dlc_config_local,
                        modelprefix=model_prefix,
                    )

                    engine = dlc_config_local.get("engine", "pytorch")
                    model_folder_str = str(model_folder)

                    if engine == "pytorch" and "dlc-models-pytorch" not in model_folder_str:
                        model_folder_str = model_folder_str.replace(
                            "dlc-models", "dlc-models-pytorch"
                        )
                    model_folder = model_folder_str

                    if isinstance(model_folder, str):
                        model_folder_path = Path(model_folder)
                    else:
                        model_folder_path = Path(model_folder)

                    if model_folder_path.is_absolute():
                        model_train_folder = model_folder_path / "train"
                    else:
                        model_train_folder = project_path / model_folder_path / "train"

                    model_train_folder = model_train_folder.resolve()
                    model_train_folder.mkdir(parents=True, exist_ok=True)

                    logger.info(f"Model train folder (absolute): {model_train_folder}")

                    snapshot_num = 1000
                    snapshot_file = model_train_folder / f"snapshot-{snapshot_num}.index"
                    snapshot_file.touch()
                    logger.info(
                        f"Created: {snapshot_file} (exists: {snapshot_file.exists()})"
                    )

                    snapshot_data = (
                        model_train_folder
                        / f"snapshot-{snapshot_num}.data-00000-of-00001"
                    )
                    snapshot_data.touch()
                    logger.info(
                        f"Created: {snapshot_data} (exists: {snapshot_data.exists()})"
                    )

                    snapshot_meta = model_train_folder / f"snapshot-{snapshot_num}.meta"
                    snapshot_meta.touch()
                    logger.info(
                        f"Created: {snapshot_meta} (exists: {snapshot_meta.exists()})"
                    )

                    if engine == "pytorch":
                        try:
                            import torch

                            snapshot_pth = (
                                model_train_folder / f"snapshot-{snapshot_num}.pth"
                            )
                            mock_checkpoint = {
                                "epoch": 0,
                                "state_dict": {},
                                "model": {},
                                "optimizer": None,
                            }
                            torch.save(mock_checkpoint, snapshot_pth)
                            logger.info(f"Created PyTorch snapshot: {snapshot_pth}")

                            snapshot_pth_alt = (
                                model_train_folder / f"snapshot_{snapshot_num}.pth"
                            )
                            if not snapshot_pth_alt.exists():
                                torch.save(mock_checkpoint, snapshot_pth_alt)
                                logger.info(
                                    f"Created alternate PyTorch snapshot: {snapshot_pth_alt}"
                                )
                        except ImportError:
                            logger.warning(
                                "PyTorch not available, creating empty .pth files"
                            )
                            snapshot_pth = (
                                model_train_folder / f"snapshot-{snapshot_num}.pth"
                            )
                            snapshot_pth.write_bytes(b"")
                            snapshot_pth_alt = (
                                model_train_folder / f"snapshot_{snapshot_num}.pth"
                            )
                            snapshot_pth_alt.write_bytes(b"")
                        except Exception as e:
                            logger.error(
                                f"Error creating PyTorch snapshots: {e}"
                            )
                            snapshot_pth = (
                                model_train_folder / f"snapshot-{snapshot_num}.pth"
                            )
                            snapshot_pth.write_bytes(b"")
                            snapshot_pth_alt = (
                                model_train_folder / f"snapshot_{snapshot_num}.pth"
                            )
                            snapshot_pth_alt.write_bytes(b"")

                        snapshot_pth = (
                            model_train_folder / f"snapshot-{snapshot_num}.pth"
                        )
                        if not snapshot_pth.exists():
                            logger.warning(
                                f"snapshot-{snapshot_num}.pth not found, creating empty file"
                            )
                            snapshot_pth.write_bytes(b"PK\x03\x04")

                        snapshot_pth_alt = (
                            model_train_folder / f"snapshot_{snapshot_num}.pth"
                        )
                        if not snapshot_pth_alt.exists():
                            snapshot_pth_alt.write_bytes(b"PK\x03\x04")

                        created_files = list(model_train_folder.glob("snapshot*"))
                        logger.info(
                            f"All snapshot files in directory: {[f.name for f in created_files]}"
                        )

                        pth_files = list(model_train_folder.glob("*.pth"))
                        logger.info(f".pth files found: {[f.name for f in pth_files]}")
                        if not pth_files:
                            raise FileNotFoundError(
                                f"No .pth files found in {model_train_folder} after creation attempt."
                            )

                        pytorch_config_path = model_train_folder / "pytorch_config.yaml"
                        create_mock_pytorch_config(
                            pytorch_config_path, project_path, dlc_config_local
                        )
                        logger.info(
                            f"Created mock pytorch_config.yaml: {pytorch_config_path}"
                        )

                    dlc_config_local["snapshotindex"] = 0
                    try:
                        from deeplabcut.utils.auxiliaryfunctions import edit_config

                        config_file_path = project_path / "config.yaml"
                        edit_config(str(config_file_path), {"snapshotindex": 0})
                    except ImportError:
                        pass

                    self.insert1(
                        {
                            **key,
                            "latest_snapshot": snapshot_num,
                            "config_template": dlc_config_local,
                        }
                    )

                pipeline.train.ModelTraining.make = mocked_make

                try:
                    pipeline.train.ModelTraining.populate()
                    status.sub(
                        "Model training completed (mocked)!", icon="✅", indent=3
                    )
                finally:
                    pipeline.train.ModelTraining.make = original_make

            except KeyboardInterrupt:
                status.sub("Training interrupted by user", icon="⚠️", indent=3)
                status.sub(
                    "Resume later with: pipeline.train.ModelTraining.populate()",
                    icon="ℹ️",
                    indent=5,
                )
                sys.exit(0)
            except Exception as e:
                status.sub(f"Error during training: {e}", icon="❌", indent=3)
                raise

            # 8. Insert trained model into Model table
            status.step("Inserting trained model into Model table")

            training_result = (
                pipeline.train.ModelTraining & training_task_key
            ).fetch1()
            latest_snapshot = training_result["latest_snapshot"]
            status.sub(f"Using snapshot: {latest_snapshot}", icon="ℹ️", indent=3)

            def _mock_get_scorer_name(*args, **kwargs):
                return "mock_scorer"

            import yaml

            with open(config_file, "r") as f:
                dlc_config_for_insert = yaml.safe_load(f)

            if "Task" in dlc_config_for_insert and len(
                dlc_config_for_insert["Task"]
            ) > 32:
                status.sub(
                    f"WARNING: Task field still too long ({len(dlc_config_for_insert['Task'])} chars), truncating now",
                    icon="⚠️",
                    indent=3,
                )
                dlc_config_for_insert["Task"] = dlc_config_for_insert["Task"][:32]
                with open(config_file, "w") as f:
                    yaml.dump(dlc_config_for_insert, f, default_flow_style=False)

            with patch(
                "deeplabcut.pose_estimation_pytorch.apis.utils.get_scorer_name",
                side_effect=_mock_get_scorer_name,
            ):
                model.Model.insert_new_model(
                    model_name=model_name,
                    dlc_config=str(config_file_rel),
                    shuffle=shuffle,
                    trainingsetindex=trainingsetindex,
                    model_description=f"Test trained model from {dlc_project_path.name}",
                    prompt=False,
                )
            status.sub(f"Inserted model: {model_name}", icon="✅", indent=3)
    else:
        status.step("Skipping training (using existing model)")
        if not len(model.Model & {"model_name": model_name}):
            status.sub(f"Model '{model_name}' not found", icon="❌", indent=3)
            status.sub("Available models:", indent=3)
            for m in model.Model.fetch("model_name"):
                status.sub(f"  - {m}", indent=5)
            sys.exit(1)
        status.sub(f"Using existing model: {model_name}", icon="✅", indent=3)

    # 9. Create pose estimation tasks (if videos exist)
    if recording_keys and not args.skip_inference:
        status.step("Creating pose estimation tasks")
        for rec_key in recording_keys:
            task_key = {**rec_key, "model_name": model_name}
            output_dir = pipeline.model.PoseEstimationTask.infer_output_dir(
                task_key, relative=False, mkdir=False
            )

            results_exist = False
            output_path = Path(output_dir)
            if output_path.exists():
                h5_files = list(output_path.glob("*.h5"))
                pickle_files = list(output_path.glob("*.pickle"))
                if h5_files or pickle_files:
                    results_exist = True
                    status.sub(
                        f"Results found for recording {rec_key['recording_id']}",
                        icon="✅",
                        indent=5,
                    )

            task_mode = "load" if results_exist else None

            pipeline.model.PoseEstimationTask.generate(
                rec_key,
                model_name=model_name,
                task_mode=task_mode,
                analyze_videos_params={
                    "videotype": ".mp4",
                    "gputouse": args.gpu if args.gpu >= 0 else None,
                    "device": f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu",
                    "batch_size": args.batch_size,
                    "batchsize": args.batch_size,
                    "save_as_csv": True,
                },
            )
            status.sub(
                f"Task created for recording {rec_key['recording_id']}",
                icon="✅",
                indent=5,
            )

    # 10. Run inference (if not skipped)
    if recording_keys and not args.skip_inference:
        status.step("Running inference or loading existing results")

        if training_was_mocked:
            status.sub(
                "Using mock trained model files for inference",
                icon="ℹ️",
                indent=3,
            )
            status.sub(
                "Note: Results will be based on mock model weights",
                icon="⚠️",
                indent=5,
            )

        all_in_load_mode = True
        for rec_key in recording_keys:
            task_key = {**rec_key, "model_name": model_name}
            try:
                task_mode = (
                    pipeline.model.PoseEstimationTask & task_key
                ).fetch1("task_mode")
                if task_mode != "load":
                    all_in_load_mode = False
            except Exception:
                all_in_load_mode = False

        if all_in_load_mode:
            status.sub(
                "All tasks in 'load' mode - using existing results",
                icon="ℹ️",
                indent=3,
            )
        else:
            status.sub("Running inference (this may take a while)", icon="⚠️", indent=3)
            status.sub("GPU is recommended for speed", icon="ℹ️", indent=5)

        def _mock_get_model_snapshots(*args, **kwargs):
            from pathlib import Path

            train_dir = None
            if args and len(args) >= 2:
                train_dir = args[1]
            elif "train_dir" in kwargs:
                train_dir = kwargs["train_dir"]
            elif args and len(args) >= 1:
                train_dir = args[0]

            if train_dir is None:
                for key in ["train_dir", "train_path", "model_dir", "snapshot_dir"]:
                    if key in kwargs:
                        train_dir = kwargs[key]
                        break

            if train_dir is None:
                raise ValueError(
                    f"Could not determine train_dir from args={args}, kwargs={kwargs}."
                )

            train_dir = Path(train_dir)
            train_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"[MOCK get_model_snapshots] train_dir={train_dir}, args={args}, kwargs={kwargs}"
            )

            base_name = "snapshot-1000"
            for ext in [".index", ".meta", ".data-00000-of-00001"]:
                f = train_dir / f"{base_name}{ext}"
                if not f.exists():
                    f.touch()
                    logger.info(f"[MOCK get_model_snapshots] Created: {f}")

            snapshot_pth = train_dir / f"{base_name}.pth"
            if not snapshot_pth.exists():
                try:
                    import torch

                    mock_checkpoint = {
                        "epoch": 0,
                        "state_dict": {},
                        "model": {},
                        "optimizer": None,
                    }
                    torch.save(mock_checkpoint, snapshot_pth)
                    logger.info(f"Created PyTorch snapshot: {snapshot_pth}")
                except ImportError:
                    logger.warning(
                        "PyTorch not available, creating minimal .pth file"
                    )
                    snapshot_pth.write_bytes(b"PK\x03\x04")
                except Exception as e:
                    logger.warning(
                        f"Could not create PyTorch snapshot with torch.save: {e}"
                    )
                    snapshot_pth.write_bytes(b"PK\x03\x04")

            class MockSnapshot:
                def __init__(self, path):
                    self.path = Path(path)
                    self.path_str = str(self.path)

                def __fspath__(self):
                    return str(self.path)

                def __str__(self):
                    return str(self.path)

                def __repr__(self):
                    return f"MockSnapshot({self.path!r})"

            snapshot_path_no_ext = train_dir / base_name

            if not snapshot_path_no_ext.exists():
                import shutil

                try:
                    snapshot_path_no_ext.symlink_to(snapshot_pth)
                    logger.info(
                        f"Created symlink: {snapshot_path_no_ext} -> {snapshot_pth}"
                    )
                except (OSError, NotImplementedError):
                    shutil.copy2(snapshot_pth, snapshot_path_no_ext)
                    logger.info(
                        f"Copied file: {snapshot_pth} -> {snapshot_path_no_ext}"
                    )

            return [MockSnapshot(snapshot_path_no_ext)]

        try:
            import torch.nn as nn
            from element_deeplabcut.readers import dlc_reader

            original_load_state_dict = nn.Module.load_state_dict
            original_pkl_property = dlc_reader.PoseEstimation.pkl
            original_pkl_fget = original_pkl_property.fget

            def mock_load_state_dict(self, state_dict, strict=True, *args, **kwargs):
                return original_load_state_dict(
                    self, state_dict, strict=False, *args, **kwargs
                )

            def lenient_pkl_wrapper(self):
                try:
                    return original_pkl_fget(self)
                except AssertionError as e:
                    if "Inconsistent DLC-model-config file used" in str(e):
                        logger.warning(
                            "Smoke test: Metadata validation failed (expected for mock models). "
                            "Returning minimal pkl structure."
                        )
                        return {
                            "nframes": 0,
                            "Scorer": "DLC_mock_scorer_shuffle1_1000",
                            "Task": "mock_task",
                            "date": "2024-01-01",
                            "iteration (active-learning)": 0,
                            "training set fraction": 0.95,
                        }
                    else:
                        raise

            dlc_reader.PoseEstimation.pkl = property(lenient_pkl_wrapper)

            with patch(
                "deeplabcut.pose_estimation_pytorch.apis.utils.get_model_snapshots",
                side_effect=_mock_get_model_snapshots,
            ), patch.object(nn.Module, "load_state_dict", mock_load_state_dict):
                # Store original make method for potential restoration
                PoseEst = pipeline.model.PoseEstimation
                original_pe_make = PoseEst.make
                
                # Define mock pose estimation make function
                def mock_pose_make(self, key):
                    """Mock PoseEstimation.make: skip DLC, directly insert fake results."""
                    import numpy as np
                    import datajoint as dj
                    
                    # Get body parts from model config if available
                    try:
                        model_key = {**key, "model_name": key.get("model_name", model_name)}
                        model_data = (pipeline.model.Model & model_key).fetch1()
                        config_template = model_data.get("config_template", {})
                        body_parts = config_template.get("bodyparts", ["nose", "tailbase", "head"])
                    except Exception:
                        # Fallback to default body parts
                        body_parts = ["nose", "tailbase", "head"]
                    
                    # Get video info to determine frame count
                    try:
                        from element_interface.utils import find_full_path
                        from element_deeplabcut.model import get_dlc_root_data_dir
                        
                        rec_key = {k: v for k, v in key.items() if k in ["device", "recording_id"]}
                        video_files = (pipeline.model.VideoRecording.File & rec_key).fetch("file_path")
                        if video_files:
                            # Try to get frame count from first video
                            try:
                                import cv2
                                video_path = find_full_path(get_dlc_root_data_dir(), video_files[0])
                                cap = cv2.VideoCapture(str(video_path))
                                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                cap.release()
                                if n_frames == 0:
                                    n_frames = 100  # Fallback
                            except Exception:
                                n_frames = 100  # Fallback
                        else:
                            n_frames = 100  # Fallback
                    except Exception:
                        n_frames = 100  # Fallback
                    
                    logger.info(
                        f"Mock PoseEstimation.make: inserting fake results for {len(body_parts)} body parts, "
                        f"{n_frames} frames"
                    )
                    
                    # Insert master row
                    self.insert1({**key, "pose_estimation_time": dj.now()})
                    
                    # Insert fake body part positions
                    for bp in body_parts:
                        x = np.random.uniform(0, 640, size=n_frames)
                        y = np.random.uniform(0, 480, size=n_frames)
                        likelihood = np.random.uniform(0.8, 1.0, size=n_frames)
                        
                        pipeline.model.PoseEstimation.BodyPartPosition.insert1(
                            {
                                **key,
                                "body_part": bp,
                                "x_pos": x,
                                "y_pos": y,
                                "likelihood": likelihood,
                            }
                        )
                
                # Try normal populate first
                try:
                    pipeline.model.PoseEstimation.populate()
                    status.sub("Inference completed!", icon="✅", indent=3)
                except Exception as populate_err:
                    # If populate fails and mock-results-on-failure is enabled, use mock
                    if args.mock_results_on_failure:
                        error_msg = str(populate_err)
                        if (
                            "need at least one array to stack" in error_msg
                            or "Shape of passed values is" in error_msg
                            or "No DLC output file" in error_msg
                            or "No animals detected" in error_msg.lower()
                        ):
                            status.sub(
                                "Inference failed (no animals detected or empty results). "
                                "Inserting mock results instead.",
                                icon="⚠️",
                                indent=3,
                            )
                            # Temporarily replace make method with mock
                            PoseEst.make = mock_pose_make
                            try:
                                # Populate with mock results for each recording
                                for rec_key in recording_keys:
                                    task_key = {**rec_key, "model_name": model_name}
                                    # Check if results already exist
                                    if len(PoseEst & task_key) == 0:
                                        try:
                                            mock_pose_make(PoseEst, task_key)
                                        except Exception as mock_err:
                                            logger.warning(
                                                f"Failed to insert mock results for {task_key}: {mock_err}"
                                            )
                                status.sub("Mock results inserted!", icon="✅", indent=3)
                            finally:
                                # Restore original make method
                                PoseEst.make = original_pe_make
                        else:
                            # Different error, re-raise
                            raise
                    else:
                        # Mock not enabled, re-raise original error
                        raise

            dlc_reader.PoseEstimation.pkl = original_pkl_property

        except Exception as e:
            status.sub(f"Error during inference: {e}", icon="❌", indent=3)
            raise

        # 11. Show results
        status.step("Results")
        for rec_key in recording_keys:
            status.sub(f"Recording {rec_key['recording_id']}:", icon="📹", indent=2)
            try:
                # Check if pose estimation results exist
                pose_estimation_query = (
                    pipeline.model.PoseEstimation
                    & rec_key
                    & {"model_name": model_name}
                )
                
                if len(pose_estimation_query) == 0:
                    status.sub(
                        "No pose estimation results found.",
                        icon="⚠️",
                        indent=4,
                    )
                    status.sub(
                        "This may indicate that:",
                        indent=5,
                    )
                    status.sub(
                        "1) No animals were detected in the video",
                        indent=6,
                    )
                    status.sub(
                        "2) Inference failed or was skipped",
                        indent=6,
                    )
                    status.sub(
                        "3) Results are still being processed",
                        indent=6,
                    )
                    # Check if output directory exists and has files
                    try:
                        task_key = {**rec_key, "model_name": model_name}
                        output_dir = pipeline.model.PoseEstimationTask.infer_output_dir(
                            task_key, relative=False, mkdir=False
                        )
                        output_path = Path(output_dir)
                        if output_path.exists():
                            result_files = list(output_path.rglob("*.h5")) + list(output_path.rglob("*.pickle"))
                            if result_files:
                                status.sub(
                                    f"Found {len(result_files)} result file(s) in {output_dir}",
                                    icon="ℹ️",
                                    indent=5,
                                )
                                status.sub(
                                    "Results may not have been inserted into database yet.",
                                    indent=5,
                                )
                            else:
                                status.sub(
                                    f"No result files found in {output_dir}",
                                    icon="ℹ️",
                                    indent=5,
                                )
                    except Exception as check_err:
                        logger.debug(f"Could not check output directory: {check_err}")
                    continue
                
                pose_estimation = pose_estimation_query.fetch1()

                status.sub(
                    f"Completed at: {pose_estimation['pose_estimation_time']}",
                    icon="✅",
                    indent=4,
                )

                body_parts = (
                    pipeline.model.PoseEstimation.BodyPartPosition
                    & rec_key
                    & {"model_name": model_name}
                ).fetch("body_part")

                unique_bp = sorted(set(body_parts))
                status.sub(
                    f"Detected body parts ({len(unique_bp)}): {unique_bp}",
                    icon="📊",
                    indent=4,
                )

                if unique_bp:
                    bp = unique_bp[0]
                    bp_data = (
                        pipeline.model.PoseEstimation.BodyPartPosition
                        & rec_key
                        & {"model_name": model_name, "body_part": bp}
                    ).fetch1()

                    x_pos = bp_data["x_pos"]
                    y_pos = bp_data["y_pos"]
                    likelihood = bp_data["likelihood"]

                    status.sub(
                        f"Example ({bp}): {len(x_pos)} frames, avg likelihood: {likelihood.mean():.3f}",
                        icon="📈",
                        indent=4,
                    )
            except Exception as e:
                error_msg = str(e)
                if "fetch1 requires exactly one tuple" in error_msg:
                    status.sub(
                        "No pose estimation results found in database.",
                        icon="⚠️",
                        indent=4,
                    )
                    status.sub(
                        "This likely means no animals were detected or inference was skipped.",
                        icon="ℹ️",
                        indent=5,
                    )
                else:
                    status.sub(f"Error fetching results: {e}", icon="❌", indent=4)

    status.header("Test completed!")
    status.sub("Next steps:", indent=2)
    if not args.skip_training:
        status.sub("- Check training results in DLC project directory", indent=4)
    if not args.skip_inference:
        status.sub(
            "- Check output directory for DLC inference results", indent=4
        )
        status.sub(
            "- Visualize results using DLC's plotting functions", indent=4
        )
        status.sub(
            "- Query PoseEstimation.BodyPartPosition for analysis", indent=4
        )


if __name__ == "__main__":
    main()
