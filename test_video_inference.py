#!/usr/bin/env python
"""Simple script to test pretrained inference with a video file.

Usage:
    1. Configure database (see below)
    2. Put your video file(s) in ./test_videos/ directory
       - In Docker: videos should be in ./data/ directory (mounted to /app/data)
    3. Run: python test_video_inference.py [model_name]
       - In Docker: make test-pretrained

    Available models:
        - superanimal_quadruped (default): For quadruped animals (mice, rats, etc.)
        - superanimal_topviewmouse: For top-view mouse pose estimation

    Examples:
        python test_video_inference.py
        python test_video_inference.py superanimal_quadruped
        python test_video_inference.py superanimal_topviewmouse

Or set DLC_ROOT_DATA_DIR environment variable to point to your video directory.

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

import datajoint as dj

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Simple status printer for user-facing messages
class StatusPrinter:
    """Simple status printer for step-by-step progress."""
    def __init__(self, total_steps=7):
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
            "skip": "⏭️"
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
dj.config.update({
    "safemode": False,
    "database.host": os.environ.get("DJ_HOST") or dj.config.get("database.host", "localhost"),
    "database.user": os.environ.get("DJ_USER") or dj.config.get("database.user", "root"),
    "database.password": os.environ.get("DJ_PASS") or dj.config.get("database.password", ""),
})

# Set database prefix for tests
if "custom" not in dj.config:
    dj.config["custom"] = {}
dj.config["custom"]["database.prefix"] = os.environ.get("DATABASE_PREFIX", dj.config["custom"].get("database.prefix", "test_"))

# Set DLC root data directory if not already set
# In Docker, prefer /app/test_videos (from project mount), otherwise /app/data
# Check for Docker: /.dockerenv exists OR we're in /app directory (Docker working dir)
is_docker = os.path.exists("/.dockerenv") or (os.getcwd() == "/app" and os.path.exists("/app"))
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
if "dlc_root_data_dir" not in dj.config.get("custom", {}) or not dj.config["custom"].get("dlc_root_data_dir"):
    dj.config["custom"]["dlc_root_data_dir"] = str(video_dir.absolute())
    logger.info(f"📁 Set DLC_ROOT_DATA_DIR to: {video_dir.absolute()}")
    if is_docker:
        logger.info("🐳 Running in Docker mode")

# Get the root directory for making relative paths (ensure it's absolute)
dlc_root_dir = Path(dj.config["custom"].get("dlc_root_data_dir", str(video_dir.absolute())))
if not dlc_root_dir.is_absolute():
    dlc_root_dir = dlc_root_dir.resolve()

logger.info(f"📊 Database: {dj.config['database.host']} (prefix: {dj.config['custom']['database.prefix']})")
logger.info(f"📁 DLC Root: {dlc_root_dir}")

from element_deeplabcut import model
from tests import tutorial_pipeline as pipeline

def check_database_connection():
    """Verify database connection is working."""
    try:
        # Try to connect by activating a schema
        test_schema = dj.schema("test_connection_check", create_schema=True, create_tables=False)
        test_schema.drop()
        return True
    except Exception as e:
        logger.error(f"\n❌ Database connection failed: {e}")
        logger.error("\nPlease configure your database:")
        logger.error("  1. Create dj_local_conf.json with database credentials")
        logger.error("  2. Or set environment variables: DJ_HOST, DJ_USER, DJ_PASS")
        logger.error("  3. Or ensure database is running (docker compose -f docker-compose-db.yaml up -d)")
        return False

def main():
    status = StatusPrinter(total_steps=8)
    status.header("Testing Pretrained Model Inference with Video")
    
    # 0. Check database connection
    status.step("Checking database connection")
    if not check_database_connection():
        return  # Exit gracefully
    status.sub("Database connection successful", indent=3)
    
    # 0.5. Clean up database (remove test data from previous runs)
    status.step("Cleaning up database (removing test data from previous runs)")
    cleanup_count = 0
    
    # Delete PoseEstimation entries (and their parts)
    pose_estimation_query = pipeline.model.PoseEstimation
    if pose_estimation_query:
        keys = pose_estimation_query.fetch("KEY")
        count = len(keys) if keys else 0
        if count > 0:
            pose_estimation_query.delete()
            cleanup_count += count
            status.sub(f"Deleted {count} PoseEstimation entry/entries", icon="🗑️", indent=3)
    
    # Delete PoseEstimationTask entries
    task_query = pipeline.model.PoseEstimationTask
    if task_query:
        keys = task_query.fetch("KEY")
        count = len(keys) if keys else 0
        if count > 0:
            task_query.delete()
            cleanup_count += count
            status.sub(f"Deleted {count} PoseEstimationTask entry/entries", icon="🗑️", indent=3)
    
    # Delete test models (models with names starting with "test_")
    test_models = model.Model & "model_name LIKE 'test_%'"
    if test_models:
        keys = test_models.fetch("KEY")
        count = len(keys) if keys else 0
        if count > 0:
            test_models.delete()
            cleanup_count += count
            status.sub(f"Deleted {count} test model(s)", icon="🗑️", indent=3)
    
    # Delete test recordings (optional - comment out if you want to keep recordings)
    # Uncomment if you want to clean recordings too:
    # if pipeline.model.VideoRecording & {"subject": "test1"}:
    #     count = len(pipeline.model.VideoRecording & {"subject": "test1"})
    #     (pipeline.model.VideoRecording & {"subject": "test1"}).delete()
    #     cleanup_count += count
    #     if count > 0:
    #         status.sub(f"Deleted {count} test recording(s)", icon="🗑️", indent=3)
    
    if cleanup_count > 0:
        status.sub(f"Total: {cleanup_count} entry/entries cleaned", icon="✅", indent=3)
    else:
        status.sub("No test data found to clean", icon="ℹ️", indent=3)
    
    # 1. Find video files
    status.step("Finding video files")
    # Use same Docker detection logic as at the top
    is_docker = os.path.exists("/.dockerenv") or (os.getcwd() == "/app" and os.path.exists("/app"))
    
    # In Docker, prefer /app/test_videos (from project mount .:/app) since videos are in ./test_videos
    if is_docker:
        # Check /app/test_videos first (from project mount)
        test_videos_path = Path("/app/test_videos")
        if test_videos_path.exists():
            default_video_dir = "/app/test_videos"
        else:
            default_video_dir = "/app/data"
    else:
        default_video_dir = "./test_videos"
    
    video_dir = Path(os.getenv("DLC_ROOT_DATA_DIR", default_video_dir))
    
    if not video_dir.exists():
        status.sub(f"Video directory not found: {video_dir}", indent=3)
        if is_docker:
            status.sub("In Docker: Videos should be in ./test_videos/ on host", indent=3)
            status.sub("(available at /app/test_videos via project mount)", indent=5)
        return  # Exit gracefully
    
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
    if not video_files:
        status.sub(f"No video files found in {video_dir}", indent=3)
        status.sub("Supported formats: .mp4, .avi, .mov", indent=3)
        if is_docker:
            status.sub("In Docker: Videos should be in ./test_videos/ on host", indent=3)
            status.sub("(available at /app/test_videos in container)", indent=5)
        return  # Exit gracefully
    
    status.sub(f"Found {len(video_files)} video file(s):", indent=3)
    for vf in video_files:
        status.sub(f"- {vf.name}", indent=5)
    
    # 2. Register pretrained model
    # Get model name from command line or use default
    pretrained_model_name = getattr(main, 'pretrained_model_name', 'superanimal_quadruped')
    
    status.step(f"Registering pretrained model: {pretrained_model_name}")
    model.PretrainedModel.populate_common_models([pretrained_model_name])
    status.sub(f"Registered: {pretrained_model_name}", icon="✅")
    
    # 3. Insert pretrained model instance
    status.step("Inserting pretrained model instance")
    # Use a model name that includes the pretrained model name for clarity
    model_name = f"test_video_inference_{pretrained_model_name.replace('superanimal_', '')}"
    
    # Check if model already exists and verify it's a pretrained model
    if model.Model & {"model_name": model_name}:
        existing_model = (model.Model & {"model_name": model_name}).fetch1()
        config_template = existing_model.get("config_template", {})
        is_pretrained = config_template.get("_pretrained_model_name") is not None
        
        if is_pretrained:
            existing_pretrained_name = config_template.get("_pretrained_model_name")
            if existing_pretrained_name == pretrained_model_name:
                status.sub(f"Model '{model_name}' already exists (pretrained: {pretrained_model_name}), skipping insertion", icon="✅")
            else:
                status.sub(f"Model '{model_name}' exists but uses different pretrained model: {existing_pretrained_name}", icon="⚠️")
                status.sub(f"Expected: {pretrained_model_name}", indent=5)
                status.sub("Deleting existing model and tasks, creating new one...", indent=5)
                # Delete any existing tasks that reference this model
                if pipeline.model.PoseEstimationTask & {"model_name": model_name}:
                    (pipeline.model.PoseEstimationTask & {"model_name": model_name}).delete()
                    status.sub("Deleted existing PoseEstimationTask entries", icon="✅", indent=7)
                # Delete the model
                (model.Model & {"model_name": model_name}).delete()
                # Insert the correct pretrained model
                model.Model.insert_pretrained_model(
                    model_name=model_name,
                    pretrained_model_name=pretrained_model_name,
                    model_description=f"Test model using {pretrained_model_name}",
                    prompt=False,
                )
                status.sub(f"Re-inserted model: {model_name}", icon="✅")
        else:
            status.sub(f"Model '{model_name}' exists but is a TRAINED model, not pretrained", icon="⚠️")
            status.sub("This script requires a PRETRAINED model. Deleting existing model and tasks...", indent=5)
            # Delete any existing tasks that reference this model
            if pipeline.model.PoseEstimationTask & {"model_name": model_name}:
                (pipeline.model.PoseEstimationTask & {"model_name": model_name}).delete()
                status.sub("Deleted existing PoseEstimationTask entries", icon="✅", indent=7)
            # Delete the model
            (model.Model & {"model_name": model_name}).delete()
            # Insert the correct pretrained model
            model.Model.insert_pretrained_model(
                model_name=model_name,
                pretrained_model_name=pretrained_model_name,
                model_description=f"Test model using {pretrained_model_name}",
                prompt=False,
            )
            status.sub(f"Inserted pretrained model: {model_name}", icon="✅")
    else:
        try:
            model.Model.insert_pretrained_model(
                model_name=model_name,
                pretrained_model_name=pretrained_model_name,
                model_description=f"Test model using {pretrained_model_name}",
                prompt=False,
            )
            status.sub(f"Inserted model: {model_name}", icon="✅")
        except dj.errors.DuplicateError as e:
            # If duplicate error occurs (e.g., same unique index), skip insertion
            status.sub(f"Model with similar configuration already exists, skipping insertion", icon="⚠️")
            status.sub(f"Error: {str(e)[:100]}...", indent=5)
    
    # 4. Setup test data (subject, session, recordings)
    status.step("Setting up test data")
    # Use shorter subject name (element-animal Subject table has limited varchar length)
    base_key = {
        "subject": "test1",
        "session_datetime": "2024-01-01 12:00:00",
    }
    
    pipeline.subject.Subject.insert1({
        "subject": "test1",  # Short name to fit database column
        "sex": "F",
        "subject_birth_date": "2020-01-01",
        "subject_description": "Test subject for video inference",
    }, skip_duplicates=True)
    
    pipeline.session.Session.insert1({
        "subject": "test1",
        "session_datetime": "2024-01-01 12:00:00",
    }, skip_duplicates=True)
    
    # Create a separate recording for each video file
    recording_keys = []
    for idx, video_file in enumerate(video_files):
        recording_key = {
            **base_key,
            "recording_id": idx + 1,  # Start from 1
        }
        recording_keys.append(recording_key)
        
        # Insert recording
        pipeline.model.VideoRecording.insert1(
            {**recording_key, "device": "Camera1"}, skip_duplicates=True
        )
        
        # Insert single video file for this recording
        # Store file path relative to root directory
        video_file_abs = Path(video_file).resolve()
        dlc_root_dir_abs = Path(dj.config["custom"].get("dlc_root_data_dir", str(video_dir.absolute()))).resolve()
        
        try:
            relative_path = video_file_abs.relative_to(dlc_root_dir_abs)
        except ValueError:
            # If video_file is not under dlc_root_dir, use just the filename
            # This handles the case where videos are in the root directory itself
            relative_path = Path(video_file.name)
        
        pipeline.model.VideoRecording.File.insert1(
            {**recording_key, "file_id": 0, "file_path": str(relative_path)},
            skip_duplicates=True,
        )
        status.sub(f"Created recording {recording_key['recording_id']} for {video_file.name}", icon="✅", indent=5)
    
    status.sub(f"Created {len(recording_keys)} separate recording(s)", icon="✅", indent=3)
    
    # 5. Extract video metadata
    status.step("Extracting video metadata")
    try:
        pipeline.model.RecordingInfo.populate()
        # Show info for all recordings
        for rec_key in recording_keys:
            rec_info = (pipeline.model.RecordingInfo & rec_key).fetch1()
            status.sub(
                f"Recording {rec_key['recording_id']}: {rec_info['px_width']}x{rec_info['px_height']}, "
                f"{rec_info['nframes']} frames, {rec_info['fps']:.1f} fps",
                icon="✅",
                indent=5
            )
    except ModuleNotFoundError as e:
        if "cv2" in str(e):
            status.sub(f"Error: {e}", icon="❌", indent=3)
            status.sub("OpenCV (cv2) is required for video metadata extraction.", indent=3)
            status.sub("Install it with: pip install opencv-python", indent=5)
            status.sub("Or: conda install -c conda-forge opencv", indent=5)
            return  # Exit gracefully
        raise
    
    # 6. Clean up any tasks that might be using wrong models (from other test scripts)
    status.step("Cleaning up any conflicting tasks")
    for rec_key in recording_keys:
        # Find all tasks for this recording, regardless of model_name
        all_tasks = (pipeline.model.PoseEstimationTask & rec_key).fetch("model_name")
        for task_model_name in set(all_tasks):
            if task_model_name != model_name:
                status.sub(f"Found task with different model '{task_model_name}' for recording {rec_key['recording_id']}", icon="⚠️", indent=3)
                status.sub(f"Deleting task (expected model: '{model_name}')...", indent=5)
                (pipeline.model.PoseEstimationTask & {**rec_key, "model_name": task_model_name}).delete()
                status.sub("Task deleted", icon="✅", indent=7)
    
    # 6. Create pose estimation tasks for each recording
    status.step("Creating pose estimation tasks")
    for rec_key in recording_keys:
        # Check if results already exist
        task_key = {**rec_key, "model_name": model_name}
        output_dir = pipeline.model.PoseEstimationTask.infer_output_dir(
            task_key, relative=False, mkdir=False
        )
        
        # Check if results exist - look for H5 files directly
        results_exist = False
        output_path = Path(output_dir)
        if output_path.exists():
            # Check for result files (H5, pickle, or JSON)
            h5_files = list(output_path.glob("*.h5"))
            pickle_files = list(output_path.glob("*.pickle"))
            json_files = list(output_path.glob("*.json"))
            
            if h5_files or pickle_files or json_files:
                results_exist = True
                status.sub(
                    f"Results found for recording {rec_key['recording_id']} in: {output_dir.name} "
                    f"({len(h5_files)} H5, {len(pickle_files)} pickle, {len(json_files)} JSON)",
                    icon="✅",
                    indent=5
                )
            else:
                status.sub(f"No result files found for recording {rec_key['recording_id']} in: {output_dir.name}", icon="⚠️", indent=5)
        else:
            status.sub(f"Output directory doesn't exist for recording {rec_key['recording_id']}: {output_dir.name}", icon="⚠️", indent=5)
        
        # Generate task - it will auto-detect and set task_mode appropriately
        # Always use "load" mode if results exist - never re-run inference
        # This prevents expensive re-computation when results already exist
        if results_exist:
            task_mode = "load"  # Use existing results - do NOT trigger inference
            status.sub(f"Results exist for recording {rec_key['recording_id']} - setting task_mode='load'", icon="✅", indent=5)
        else:
            task_mode = None  # Auto-detect (will be "trigger" if no results)
            status.sub(f"No results found for recording {rec_key['recording_id']} - will auto-detect task_mode", icon="ℹ️", indent=5)
        
        pipeline.model.PoseEstimationTask.generate(
            rec_key,
            model_name=model_name,
            task_mode=task_mode,
            analyze_videos_params={
                "video_inference": {
                    "scale": 0.4,  # Adjust if needed (0.3-0.5 recommended)
                    "batchsize": 8,  # Adjust based on GPU memory
                }
            },
        )
        status.sub(f"Task created for recording {rec_key['recording_id']}", icon="✅", indent=5)
    
    # 6.5. Update task modes if needed (in case results were created after task creation)
    status.step("Checking and updating task modes")
    for rec_key in recording_keys:
        task_key = {**rec_key, "model_name": model_name}
        output_dir = pipeline.model.PoseEstimationTask.infer_output_dir(
            task_key, relative=False, mkdir=False
        )
        
        # Check if results exist
        output_path = Path(output_dir)
        results_exist = False
        if output_path.exists():
            h5_files = list(output_path.glob("*.h5"))
            pickle_files = list(output_path.glob("*.pickle"))
            json_files = list(output_path.glob("*.json"))
            if h5_files or pickle_files or json_files:
                results_exist = True
        
        # Check current task mode
        try:
            current_task = (pipeline.model.PoseEstimationTask & task_key).fetch1()
            current_mode = current_task.get("task_mode", "trigger")
            
            # Update to "load" if results exist but task is in "trigger" mode
            if results_exist and current_mode == "trigger":
                pipeline.model.PoseEstimationTask.update1(
                    {**task_key, "task_mode": "load"}
                )
                status.sub(f"Updated recording {rec_key['recording_id']} task_mode to 'load' (results exist)", icon="✅", indent=5)
            elif not results_exist and current_mode == "load":
                pipeline.model.PoseEstimationTask.update1(
                    {**task_key, "task_mode": "trigger"}
                )
                status.sub(f"Updated recording {rec_key['recording_id']} task_mode to 'trigger' (no results)", icon="⚠️", indent=5)
            else:
                status.sub(f"Recording {rec_key['recording_id']} task_mode is '{current_mode}' (correct)", icon="ℹ️", indent=5)
        except Exception as e:
            status.sub(f"Could not check/update task for recording {rec_key['recording_id']}: {e}", icon="⚠️", indent=5)
    
    # 7. Run inference or load existing results
    status.step("Running inference or loading existing results")
    
    # Check if DeepLabCut is available FIRST (before checking task modes)
    # This import might print "Loading DLC..." so we do it early
    deeplabcut_available = False
    error_msg = None
    
    # First check if PyTorch is available (needed for SuperAnimal)
    try:
        import torch
        pytorch_available = True
        pytorch_version = torch.__version__
    except ImportError:
        pytorch_available = False
        pytorch_version = None
    
    try:
        import deeplabcut
        deeplabcut_available = True
        logger.info("DeepLabCut imported successfully")
        # Verify it's actually usable by checking for a key function
        if hasattr(deeplabcut, "video_inference_superanimal"):
            logger.info("DeepLabCut has video_inference_superanimal function")
        else:
            logger.warning("DeepLabCut imported but video_inference_superanimal not found")
    except (ImportError, TypeError, Exception) as e:
        error_msg = str(e)
        deeplabcut_available = False
        logger.warning(f"DeepLabCut import failed: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
    
    # Double-check task modes and results before proceeding
    # This ensures we never trigger inference if results exist
    all_in_load_mode = True
    any_results_exist = False
    for rec_key in recording_keys:
        task_key = {**rec_key, "model_name": model_name}
        try:
            task_mode = (pipeline.model.PoseEstimationTask & task_key).fetch1("task_mode")
            
            # Check if results actually exist for this task
            output_dir = pipeline.model.PoseEstimationTask.infer_output_dir(
                task_key, relative=False, mkdir=False
            )
            output_path = Path(output_dir)
            results_exist = False
            if output_path.exists():
                h5_files = list(output_path.glob("*.h5"))
                pickle_files = list(output_path.glob("*.pickle"))
                json_files = list(output_path.glob("*.json"))
                if h5_files or pickle_files or json_files:
                    results_exist = True
                    any_results_exist = True
            
            # If results exist but task is in trigger mode, update it
            if results_exist and task_mode == "trigger":
                pipeline.model.PoseEstimationTask.update1(
                    {**task_key, "task_mode": "load"}
                )
                status.sub(f"Updated recording {rec_key['recording_id']} to 'load' mode (results exist)", icon="✅", indent=3)
                task_mode = "load"
            
            if task_mode != "load":
                all_in_load_mode = False
                status.sub(f"Recording {rec_key['recording_id']} is in '{task_mode}' mode", icon="ℹ️", indent=3)
        except Exception as e:
            # Task doesn't exist yet, so not in load mode
            all_in_load_mode = False
            status.sub(f"Task for recording {rec_key['recording_id']} doesn't exist: {e}", icon="⚠️", indent=3)
            break
    
    # NEVER run inference if results exist - always use load mode
    if any_results_exist:
        status.sub("Results exist - will use 'load' mode (skipping inference)", icon="ℹ️", indent=3)
        # Ensure all tasks with results are in load mode
        for rec_key in recording_keys:
            task_key = {**rec_key, "model_name": model_name}
            try:
                output_dir = pipeline.model.PoseEstimationTask.infer_output_dir(
                    task_key, relative=False, mkdir=False
                )
                output_path = Path(output_dir)
                if output_path.exists():
                    h5_files = list(output_path.glob("*.h5"))
                    pickle_files = list(output_path.glob("*.pickle"))
                    json_files = list(output_path.glob("*.json"))
                    if h5_files or pickle_files or json_files:
                        # Force load mode if results exist
                        current_task = (pipeline.model.PoseEstimationTask & task_key).fetch1()
                        if current_task.get("task_mode") != "load":
                            pipeline.model.PoseEstimationTask.update1(
                                {**task_key, "task_mode": "load"}
                            )
            except Exception:
                pass
    
    # Verify all tasks use the correct pretrained model before inference
    status.step("Verifying model configuration")
    for rec_key in recording_keys:
        task_key = {**rec_key, "model_name": model_name}
        if pipeline.model.PoseEstimationTask & task_key:
            task = (pipeline.model.PoseEstimationTask & task_key).fetch1()
            # Verify the model is actually a pretrained model
            model_record = (model.Model & {"model_name": model_name}).fetch1()
            config_template = model_record.get("config_template", {})
            if config_template.get("_pretrained_model_name") is None:
                status.sub(f"ERROR: Task for recording {rec_key['recording_id']} references a TRAINED model, not pretrained!", icon="❌", indent=3)
                status.sub("Deleting incorrect task...", indent=5)
                (pipeline.model.PoseEstimationTask & task_key).delete()
                status.sub("Task deleted. Please re-run the script to create correct tasks.", icon="✅", indent=5)
                return  # Exit gracefully
    status.sub("All tasks verified to use pretrained model", icon="✅", indent=3)
    
    if all_in_load_mode and deeplabcut_available:
        status.step("Loading existing results")
        status.sub("All tasks are in 'load' mode - will use existing results", icon="ℹ️", indent=3)
        status.sub("Skipping inference step (results already exist)", icon="⚠️", indent=3)
        try:
            pipeline.model.PoseEstimation.populate()
            # Check if there are any IndividualMapping entries (for multi-animal data)
            individual_mappings = (
                pipeline.model.PoseEstimation.IndividualMapping 
                & rec_key 
                & {"model_name": model_name}
            )
            num_mappings = len(individual_mappings)
            
            if num_mappings > 0:
                status.sub(f"Results loaded successfully! ({num_mappings} individual mappings created)", icon="✅", indent=3)
            else:
                # Check if this is multi-animal data that should have mappings
                individuals = (
                    pipeline.model.PoseEstimation.Individual 
                    & rec_key 
                    & {"model_name": model_name}
                )
                if len(individuals) > 0:
                    status.sub("Results loaded with warnings: Individual mappings could not be created", icon="⚠️", indent=3)
                    status.sub(f"Found {len(individuals)} individual(s) but 0 mappings", icon="ℹ️", indent=4)
                else:
                    status.sub("Results loaded successfully! (single-animal data)", icon="✅", indent=3)
        except Exception as e:
            status.sub(f"Error loading results: {e}", icon="❌", indent=3)
            raise
        return  # Exit early if we're just loading
    
    # Debug: print what we detected
    status.sub(f"DLC availability check: deeplabcut_available={deeplabcut_available}, error_msg={error_msg}", indent=3)
    
    # If DLC is not available, show error and exit
    if not deeplabcut_available:
        status.sub("DeepLabCut is not available - cannot run inference", icon="⚠️", indent=3)
        
        # Check if deeplabcut package exists but has missing dependencies
        if error_msg:
            spec = importlib.util.find_spec("deeplabcut")
            if spec is not None:
                # Package exists but import failed - likely missing dependency
                if "tensorflow" in error_msg.lower():
                    if pytorch_available:
                        status.sub("DeepLabCut is installed but TensorFlow is missing.", icon="⚠️", indent=3)
                        status.sub(f"PyTorch is available (version {pytorch_version})", icon="✅", indent=5)
                    status.sub("DeepLabCut's __init__.py tries to import TensorFlow by default,", indent=3)
                    status.sub("but SuperAnimal models only need PyTorch.", indent=3)
                    status.sub("Workaround options:", indent=3)
                    status.sub("1. Install TensorFlow (even if unused): pip install tensorflow", indent=5)
                    status.sub("2. Or try setting environment variable: export DLC_BACKEND='pytorch'", indent=5)
                    status.sub("3. Or use DeepLabCut's PyTorch-only installation:", indent=5)
                    status.sub("   pip install --upgrade 'deeplabcut[superanimal]'", indent=7)
                else:
                    status.sub("DeepLabCut is installed but TensorFlow is missing.", icon="⚠️", indent=3)
                    status.sub("For SuperAnimal pretrained models, you need PyTorch.", indent=3)
                    status.sub("To fix, install PyTorch: pip install torch", indent=5)
            if "torch" in error_msg.lower() or "pytorch" in error_msg.lower():
                status.sub("DeepLabCut is installed but PyTorch is missing.", icon="⚠️", indent=3)
                status.sub("For SuperAnimal pretrained models, PyTorch is required.", indent=3)
                status.sub("To fix, install PyTorch: pip install torch", indent=5)
            elif "tensorpack" in error_msg.lower():
                status.sub("DeepLabCut is installed but tensorpack is missing.", icon="⚠️", indent=3)
                if pytorch_available:
                    status.sub(f"PyTorch is available (version {pytorch_version})", icon="✅", indent=5)
                status.sub("This is a dependency issue. To fix, reinstall DeepLabCut:", indent=3)
                status.sub("pip uninstall deeplabcut", indent=5)
                status.sub("pip install 'deeplabcut[superanimal]==3.0.0rc13'", indent=5)
            elif "unsupported operand type(s) for |" in error_msg or "|:" in error_msg:
                import sys
                python_version = sys.version_info
                status.sub("Python version incompatibility detected.", icon="⚠️", indent=3)
                status.sub(f"Current Python version: {python_version.major}.{python_version.minor}.{python_version.micro}", indent=5)
                status.sub("DeepLabCut 3.0.0rc13 requires Python 3.10 or higher", indent=3)
                status.sub("(it uses modern type hints like 'int | None' which require Python 3.10+)", indent=5)
                status.sub("To fix:", indent=3)
                status.sub("1. Create a new conda environment with Python 3.10+:", indent=5)
                status.sub("   conda create -n element-dlc python=3.10", indent=7)
                status.sub("   conda activate element-dlc", indent=7)
                status.sub("   conda env update -f environment.yml", indent=7)
                status.sub("2. Or upgrade your current environment:", indent=5)
                status.sub("   conda install python=3.10", indent=7)
                status.sub("   pip install --upgrade 'deeplabcut[superanimal]==3.0.0rc13'", indent=7)
            else:
                status.sub(f"DeepLabCut is installed but has missing dependencies: {error_msg}", icon="⚠️", indent=3)
                if pytorch_available:
                    status.sub(f"PyTorch is available (version {pytorch_version})", icon="✅", indent=5)
                status.sub("For SuperAnimal pretrained models, you typically need PyTorch.", indent=3)
                status.sub("To fix, reinstall DeepLabCut:", indent=3)
                status.sub("pip uninstall deeplabcut", indent=5)
                status.sub("pip install 'deeplabcut[superanimal]'", indent=5)
        else:
            # Package doesn't exist
            status.sub("DeepLabCut is not installed. Skipping inference step.", icon="⚠️", indent=3)
            status.sub("To run inference with SuperAnimal models, install:", indent=3)
            status.sub("pip install 'deeplabcut[superanimal]'", indent=5)
            status.sub("This will install DeepLabCut with PyTorch support for pretrained models.", indent=5)
        
        status.sub("The workflow has been set up successfully up to this point:", indent=3)
        status.sub("Pretrained model registered", icon="✅", indent=5)
        status.sub("Model inserted", icon="✅", indent=5)
        status.sub("Test data created", icon="✅", indent=5)
        status.sub("Pose estimation tasks created", icon="✅", indent=5)
        status.sub("After fixing the issue, you can run:", indent=3)
        status.sub("pipeline.model.PoseEstimation.populate()", indent=5)
        return
    
    # Final check: NEVER run inference if results exist
    # Check one more time before proceeding
    has_any_results = False
    for rec_key in recording_keys:
        task_key = {**rec_key, "model_name": model_name}
        try:
            output_dir = pipeline.model.PoseEstimationTask.infer_output_dir(
                task_key, relative=False, mkdir=False
            )
            output_path = Path(output_dir)
            if output_path.exists():
                h5_files = list(output_path.glob("*.h5"))
                pickle_files = list(output_path.glob("*.pickle"))
                json_files = list(output_path.glob("*.json"))
                if h5_files or pickle_files or json_files:
                    has_any_results = True
                    # Force load mode - CRITICAL: prevent inference
                    pipeline.model.PoseEstimationTask.update1(
                        {**task_key, "task_mode": "load"}
                    )
                    status.sub(f"⚠️ Found results for recording {rec_key['recording_id']} - forcing 'load' mode (will NOT trigger inference)", icon="⚠️", indent=3)
        except Exception as e:
            logger.debug(f"Error checking results for {rec_key}: {e}")
    
    if has_any_results:
        status.sub("Results exist - using 'load' mode instead of triggering inference", icon="ℹ️", indent=3)
        try:
            pipeline.model.PoseEstimation.populate()
            status.sub("Results loaded successfully!", icon="✅", indent=3)
        except Exception as e:
            status.sub(f"Error loading results: {e}", icon="❌", indent=3)
            raise
        return  # Exit - do NOT run inference
    
    # Only run inference if NO results exist anywhere
    status.sub("DeepLabCut is available - proceeding with inference", icon="✅", indent=3)
    status.sub("This may take a while (requires GPU for reasonable speed)", icon="⚠️", indent=3)
    status.sub(f"Processing {len(recording_keys)} video(s)...", icon="⚠️", indent=3)
    try:
        pipeline.model.PoseEstimation.populate()
        status.sub("Inference completed!", icon="✅", indent=3)
    except Exception as e:
        status.sub(f"Error during inference: {e}", icon="❌", indent=3)
        status.sub("Troubleshooting:", indent=3)
        status.sub("- Ensure DeepLabCut is installed: pip install 'deeplabcut[superanimal]'", indent=5)
        status.sub("- Check GPU availability (or use CPU - will be slow)", indent=5)
        status.sub("- Try reducing batchsize or scale in analyze_videos_params", indent=5)
        raise
    
    # 8. Show results for each recording
    status.header("Results")
    
    for rec_key in recording_keys:
        status.sub(f"Recording {rec_key['recording_id']}:", icon="📹", indent=2)
        try:
            pose_estimation = (
                pipeline.model.PoseEstimation 
                & rec_key 
                & {"model_name": model_name}
            ).fetch1()
            
            status.sub(f"Completed at: {pose_estimation['pose_estimation_time']}", icon="✅", indent=4)
            
            body_parts = (
                pipeline.model.PoseEstimation.BodyPartPosition 
                & rec_key 
                & {"model_name": model_name}
            ).fetch("body_part")
            
            status.sub(f"Detected body parts ({len(set(body_parts))}): {sorted(set(body_parts))}", icon="📊", indent=4)
            
            # Show stats for first body part as example
            if len(set(body_parts)) > 0:
                bp = sorted(set(body_parts))[0]
                bp_data = (
                    pipeline.model.PoseEstimation.BodyPartPosition 
                    & rec_key 
                    & {"model_name": model_name, "body_part": bp}
                ).fetch1()
                
                x_pos = bp_data["x_pos"]
                y_pos = bp_data["y_pos"]
                likelihood = bp_data["likelihood"]
                
                status.sub(f"Example ({bp}): {len(x_pos)} frames, avg likelihood: {likelihood.mean():.3f}", icon="📈", indent=4)
        except Exception as e:
            status.sub(f"No results yet or error: {e}", icon="⚠️", indent=4)
    
    status.header("Test completed successfully!")
    status.sub("Next steps:", indent=2)
    status.sub("- Check output directory for DLC results", indent=4)
    status.sub("- Visualize results using DLC's plotting functions", indent=4)
    status.sub("- Query PoseEstimation.BodyPartPosition for analysis", indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test pretrained DeepLabCut inference with video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_video_inference.py
  python test_video_inference.py superanimal_quadruped
  python test_video_inference.py superanimal_topviewmouse
        """
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="superanimal_quadruped",
        choices=["superanimal_quadruped", "superanimal_topviewmouse"],
        help="Pretrained model to use (default: superanimal_quadruped)"
    )
    
    args = parser.parse_args()
    
    # Store model name as attribute so main() can access it
    main.pretrained_model_name = args.model
    
    main()


