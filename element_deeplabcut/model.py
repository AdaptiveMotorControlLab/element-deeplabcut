"""
Code adapted from the Mathis Lab
MIT License Copyright (c) 2022 Mackenzie Mathis
DataJoint Schema for DeepLabCut 2.x, Supports 2D and 3D DLC via triangulation.
"""

import datajoint as dj
import os
import csv
from ruamel.yaml import YAML
import inspect
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from element_interface.utils import find_full_path, find_root_directory
try:
    from element_interface.utils import memoized_result
except ImportError:
    # Fallback if memoized_result not available (e.g., older element-interface versions)
    def memoized_result(*args, **kwargs):
        def decorator(func):
            return func  # Return function unchanged if memoization not available
        return decorator
from .readers import dlc_reader

schema = dj.schema()
logger = dj.logger

_linking_module = None

# Apply dlclibrary bug patch early at module import
def _apply_dlclibrary_patch_early():
    """Apply dlclibrary ModelZoo bug patch at module import time."""
    try:
        import dlclibrary.dlcmodelzoo.modelzoo_download as modelzoo_download
        if hasattr(modelzoo_download, '_handle_downloaded_file'):
            original_handle = modelzoo_download._handle_downloaded_file
            
            def patched_handle_downloaded_file(file_name, target_dir, rename_mapping):
                """Patched version that handles rename_mapping being a string."""
                # Fix: If rename_mapping is a string, convert to dict or use empty dict
                if isinstance(rename_mapping, str):
                    rename_mapping = {}
                elif rename_mapping is None:
                    rename_mapping = {}
                
                # Call original function with fixed rename_mapping
                return original_handle(file_name, target_dir, rename_mapping)
            
            # Apply the patch
            modelzoo_download._handle_downloaded_file = patched_handle_downloaded_file
            logger.debug("Applied dlclibrary ModelZoo bug patch at module import")
    except (ImportError, AttributeError):
        # dlclibrary not available yet, will be patched later if needed
        pass

# Try to apply patch early (will be applied again in _do_pretrained_inference if needed)
try:
    _apply_dlclibrary_patch_early()
except Exception:
    # Silently fail - patch will be applied later when needed
    pass


def activate(
    model_schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: bool = None,
):
    """Activate this schema.

    Args:
        model_schema_name (str): schema name on the database server
        create_schema (bool): when True (default), create schema in the database if it
                            does not yet exist.
        create_tables (bool): when True (default), create schema tables in the database
                             if they do not yet exist.
        linking_module (str): a module (or name) containing the required dependencies.

    Dependencies:
    Upstream tables:
        Session: A parent table to VideoRecording, identifying a recording session.
        Equipment: A parent table to VideoRecording, identifying a recording device.
    Functions:
        get_dlc_root_data_dir(): Returns absolute path for root data director(y/ies)
                                 with all behavioral recordings, as (list of) string(s).
        get_dlc_processed_data_dir(): Optional. Returns absolute path for processed
                                      data. Defaults to session video subfolder.
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(
        linking_module
    ), "The argument 'dependency' must be a module's name or a module"
    assert hasattr(
        linking_module, "get_dlc_root_data_dir"
    ), "The linking module must specify a lookup function for a root data directory"

    global _linking_module
    _linking_module = linking_module

    # activate
    schema.activate(
        model_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )


# -------------- Functions required by element-deeplabcut ---------------


def get_dlc_root_data_dir() -> list:
    """Pulls relevant func from parent namespace to specify root data dir(s).

    It is recommended that all paths in DataJoint Elements stored as relative
    paths, with respect to some user-configured "root" director(y/ies). The
    root(s) may vary between data modalities and user machines. Returns a full path
    string or list of strings for possible root data directories.
    """
    root_directories = _linking_module.get_dlc_root_data_dir()
    
    # Handle None case
    if root_directories is None:
        return []
    
    if isinstance(root_directories, (str, Path)):
        root_directories = [root_directories]

    if (
        hasattr(_linking_module, "get_dlc_processed_data_dir")
        and _linking_module.get_dlc_processed_data_dir() is not None
        and _linking_module.get_dlc_processed_data_dir() not in root_directories
    ):
        root_directories.append(_linking_module.get_dlc_processed_data_dir())

    return root_directories


def get_dlc_processed_data_dir() -> Optional[str]:
    """Pulls relevant func from parent namespace. Defaults to DLC's project /videos/.

    Method in parent namespace should provide a string to a directory where DLC output
    files will be stored. If unspecified, output files will be stored in the
    session directory 'videos' folder, per DeepLabCut default.
    """
    if hasattr(_linking_module, "get_dlc_processed_data_dir"):
        return _linking_module.get_dlc_processed_data_dir()
    else:
        return None


# ----------------------------- Table declarations ----------------------


@schema
class VideoRecording(dj.Manual):
    """Set of video recordings for DLC inferences.

    Attributes:
        Session (foreign key): Session primary key.
        recording_id (int): Unique recording ID.
        Device (foreign key): Device table primary key, used for default output
            directory path information.
    """

    definition = """
    -> Session
    recording_id: int
    ---
    -> Device
    """

    class File(dj.Part):
        """File IDs and paths associated with a given recording_id

        Attributes:
            VideoRecording (foreign key): Video recording primary key.
            file_path ( varchar(255) ): file path of video, relative to root data dir.
        """

        definition = """
        -> master
        file_id: int
        ---
        file_path: varchar(255)  # filepath of video, relative to root data directory
        """


@schema
class RecordingInfo(dj.Imported):
    """Automated table with video file metadata.

    Attributes:
        VideoRecording (foreign key): Video recording key.
        px_height (smallint): Height in pixels.
        px_width (smallint): Width in pixels.
        nframes (int): Number of frames.
        fps (int): Optional. Frames per second, Hz.
        recording_datetime (datetime): Optional. Datetime for the start of recording.
        recording_duration (float): video duration (s) from nframes / fps."""

    definition = """
    -> VideoRecording
    ---
    px_height                 : smallint  # height in pixels
    px_width                  : smallint  # width in pixels
    nframes                   : int  # number of frames 
    fps = NULL                : int       # (Hz) frames per second
    recording_datetime = NULL : datetime  # Datetime for the start of the recording
    recording_duration        : float     # video duration (s) from nframes / fps
    """

    @property
    def key_source(self):
        """Defines order of keys for make function when called via `populate()`"""
        return VideoRecording & VideoRecording.File

    def make(self, key):
        """Populates table with video metadata using CV2."""
        import cv2

        file_paths = (VideoRecording.File & key).fetch("file_path")

        nframes = 0
        px_height, px_width, fps = None, None, None

        for file_path in file_paths:
            file_path = (find_full_path(get_dlc_root_data_dir(), file_path)).as_posix()

            cap = cv2.VideoCapture(file_path)
            info = (
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FPS)),
            )
            if px_height is not None:
                # Allow different dimensions, but warn if they differ
                if (px_height, px_width, fps) != info:
                    logger.warning(
                        f"Video files in recording have different properties. "
                        f"First video: {px_width}x{px_height} @ {fps} fps, "
                        f"Current video ({Path(file_path).name}): {info[1]}x{info[0]} @ {info[2]} fps. "
                        f"Using first video's properties for metadata."
                    )
                # Use first video's properties, but still count frames from all videos
            else:
                # First video - set as reference
                px_height, px_width, fps = info
            nframes += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        self.insert1(
            {
                **key,
                "px_height": px_height,
                "px_width": px_width,
                "nframes": nframes,
                "fps": fps,
                "recording_duration": nframes / fps,
            }
        )


@schema
class BodyPart(dj.Lookup):
    """Body parts tracked by DeepLabCut models

    Attributes:
        body_part ( varchar(32) ): Body part short name.
        body_part_description ( varchar(1000),optional ): Full description

    """

    definition = """
    body_part                : varchar(32)
    ---
    body_part_description='' : varchar(1000)
    """

    @classmethod
    def extract_new_body_parts(cls, dlc_config: dict, verbose: bool = True):
        """Returns list of body parts present in dlc config, but not BodyPart table.

        Args:
            dlc_config ( varchar(255) ): Path to a config.y*ml.
            verbose (bool): Default True. Print both existing and new items to console.
        """
        if not isinstance(dlc_config, dict):
            dlc_config_fp = find_full_path(get_dlc_root_data_dir(), Path(dlc_config))
            assert dlc_config_fp.exists() and dlc_config_fp.suffix in (
                ".yml",
                ".yaml",
            ), f"dlc_config is neither dict nor filepath\n Check: {dlc_config_fp}"
            if dlc_config_fp.suffix in (".yml", ".yaml"):
                yaml = YAML(typ="safe", pure=True)
                with open(dlc_config_fp, "rb") as f:
                    dlc_config = yaml.load(f)
        # -- Check and insert new BodyPart --
        assert "bodyparts" in dlc_config, f"Found no bodyparts section in {dlc_config}"
        tracked_body_parts = cls.fetch("body_part")
        new_body_parts = np.setdiff1d(dlc_config["bodyparts"], tracked_body_parts)
        if verbose:  # Added to silence duplicate prompt during `insert_new_model`
            logger.info(f"Existing body parts: {tracked_body_parts}")
            logger.info(f"New body parts: {new_body_parts}")
        return new_body_parts

    @classmethod
    def insert_from_config(
        cls, dlc_config: dict, descriptions: list = None, prompt=True
    ):
        """Insert all body parts from a config file.

        Args:
            dlc_config ( varchar(255) ): Path to a config.y*ml.
            descriptions (list): Optional. List of strings describing new body parts.
            prompt (bool): Optional, default True. Prompt for confirmation before insert.
        """

        # handle dlc_config being a yaml file
        new_body_parts = cls.extract_new_body_parts(dlc_config, verbose=False)
        if new_body_parts is not None:  # Required bc np.array is ambiguous as bool
            if descriptions:
                assert len(descriptions) == len(new_body_parts), (
                    "Descriptions list does not match "
                    + " the number of new_body_parts"
                )
                logger.info(f"New descriptions: {descriptions}")
            if descriptions is None:
                descriptions = ["" for x in range(len(new_body_parts))]

            if (
                prompt
                and dj.utils.user_choice(
                    f"Insert {len(new_body_parts)} new body " + "part(s)?"
                )
                != "yes"
            ):
                logger.info("Canceled insert.")
                return
            cls.insert(
                [
                    {"body_part": b, "body_part_description": d}
                    for b, d in zip(new_body_parts, descriptions)
                ]
            )


@schema
class PretrainedModel(dj.Lookup):
    """Pretrained DeepLabCut models available for use.

    Attributes:
        pretrained_model_name ( varchar(64) ): Name of the pretrained model (e.g., "superanimal_quadruped").
        version ( varchar(32) ): Optional. Version of the pretrained model.
        species ( varchar(64) ): Optional. Species this model was trained on.
        source ( varchar(128) ): Source of the pretrained model (e.g., "DLC Model Zoo",
            "SuperAnimal", "ResNet", etc.).
        backbone_model_name ( varchar(64) ): Optional. Backbone model name (e.g., "hrnet_w32").
        detector_name ( varchar(128) ): Optional. Detector name (e.g., "fasterrcnn_resnet50_fpn_v2").
        default_params (longblob): Optional. Default inference parameters (dict-like, e.g.,
            {"video_adapt": False, "scale": 0.4, "batchsize": 8}).
        weights_path ( varchar(255) ): Optional. Path to model weights file/URI (if applicable).
        description ( varchar(1000) ): Optional. Description of the pretrained model.
    """

    definition = """
    pretrained_model_name : varchar(64)  # Name of the pretrained model (e.g., "superanimal_quadruped")
    ---
    version=''            : varchar(32)  # Version of the pretrained model
    species=''            : varchar(64)   # Species this model was trained on
    source=''             : varchar(128)  # Source (e.g., "DLC Model Zoo", "SuperAnimal", "ResNet")
    backbone_model_name='' : varchar(64)  # Optional. Backbone model name (e.g., "hrnet_w32")
    detector_name=''       : varchar(128) # Optional. Detector name (e.g., "fasterrcnn_resnet50_fpn_v2")
    default_params=null   : longblob      # Optional. Default inference parameters (dict-like)
    weights_path=''       : varchar(255)  # Optional. Path to model weights file (if applicable)
    description=''        : varchar(1000) # Description of the pretrained model
    """

    @classmethod
    def is_pretrained(cls, pretrained_model_name: str) -> bool:
        """Check if a pretrained model name exists in the lookup table.
        
        Args:
            pretrained_model_name: Name of the pretrained model to check.
            
        Returns:
            bool: True if the model exists in the lookup table, False otherwise.
        """
        return bool(cls & {"pretrained_model_name": pretrained_model_name})

    @classmethod
    def add(
        cls,
        pretrained_model_name: str,
        source: str = "",
        version: str = "",
        species: str = "",
        backbone_model_name: str = "",
        detector_name: str = "",
        weights_path: str = "",
        default_params: dict = None,
        description: str = "",
    ):
        """Register a pretrained model in the lookup table.
        
        This is a convenience method that inserts a pretrained model if it doesn't
        already exist. For Lookup tables, models should be explicitly registered
        with proper configuration.
        
        Args:
            pretrained_model_name: Name of the pretrained model (e.g., "superanimal_quadruped").
            source: Source of the pretrained model (e.g., "SuperAnimal", "DLC Model Zoo").
                Recommended for proper model identification.
            version: Optional. Version of the pretrained model.
            species: Optional. Species this model was trained on.
            backbone_model_name: Optional. Backbone model name (e.g., "hrnet_w32").
            detector_name: Optional. Detector name (e.g., "fasterrcnn_resnet50_fpn_v2").
            weights_path: Optional. Path to model weights file/URI (if applicable).
            default_params: Optional. Default inference parameters (dict-like, e.g.,
                {"video_adapt": False, "scale": 0.4, "batchsize": 8}).
            description: Optional. Description of the pretrained model.
            
        Returns:
            bool: True if model exists or was successfully inserted.
            
        Raises:
            ValueError: If model doesn't exist and essential information is missing.
        """
        if cls.is_pretrained(pretrained_model_name):
            return True
        
        # Validate that essential information is provided
        # At minimum, should have source or default_params for meaningful registration
        if not source and not default_params and not weights_path:
            raise ValueError(
                f"Cannot register pretrained model '{pretrained_model_name}' without "
                "essential information. Please provide at least one of: "
                "source, default_params, or weights_path."
            )
        
        cls.insert1(
            {
                "pretrained_model_name": pretrained_model_name,
                "version": version,
                "species": species,
                "source": source,
                "backbone_model_name": backbone_model_name,
                "detector_name": detector_name,
                "weights_path": weights_path,
                "default_params": default_params,
                "description": description,
            },
            skip_duplicates=True,
        )
        return True

    @classmethod
    def populate_common_models(cls, models: list = None):
        """Populate the lookup table with common pretrained models.
        
        This method registers well-known pretrained models (e.g., SuperAnimal models)
        with their default configurations. Models are only inserted if they don't
        already exist in the table.
        
        Args:
            models: Optional. List of model names to populate. If None, populates all
                common models. Valid options: "superanimal_quadruped", "superanimal_topviewmouse".
                
        Returns:
            dict: Summary of registration results with keys 'inserted', 'skipped', 'failed'.
        """
        # Define common models with their configurations
        # Only include models that are valid SuperAnimal identifiers in DeepLabCut
        common_models = {
            "superanimal_quadruped": {
                "pretrained_model_name": "superanimal_quadruped",
                "source": "SuperAnimal",
                "version": "1.0",
                "species": "quadruped",
                "backbone_model_name": "hrnet_w32",
                "detector_name": "fasterrcnn_resnet50_fpn_v2",
                "default_params": {
                    "video_adapt": False,
                    "scale": 0.4,
                    "batchsize": 8,
                },
                "description": "SuperAnimal model for quadruped animals (mice, rats, etc.)",
            },
            "superanimal_topviewmouse": {
                "pretrained_model_name": "superanimal_topviewmouse",
                "source": "SuperAnimal",
                "version": "1.0",
                "species": "mouse",
                "backbone_model_name": "hrnet_w32",
                "detector_name": "fasterrcnn_resnet50_fpn_v2",
                "default_params": {
                    "video_adapt": False,
                    "scale": 0.4,
                    "batchsize": 8,
                },
                "description": "SuperAnimal model for top-view mouse pose estimation",
            },
        }
        
        # Determine which models to populate
        if models is None:
            models_to_populate = list(common_models.keys())
        else:
            models_to_populate = models if isinstance(models, list) else [models]
        
        # Validate model names
        invalid_models = [m for m in models_to_populate if m not in common_models]
        if invalid_models:
            raise ValueError(
                f"Unknown model names: {invalid_models}. "
                f"Valid options: {list(common_models.keys())}"
            )
        
        # Register models
        results = {"inserted": [], "skipped": [], "failed": []}
        
        for model_name in models_to_populate:
            model_config = common_models[model_name]
            
            try:
                if cls.is_pretrained(model_name):
                    results["skipped"].append(model_name)
                    logger.info(f"Model '{model_name}' already exists, skipping.")
                else:
                    cls.insert1(model_config, skip_duplicates=True)
                    results["inserted"].append(model_name)
                    logger.info(f"Registered pretrained model: '{model_name}'")
            except Exception as e:
                results["failed"].append((model_name, str(e)))
                logger.error(f"Failed to register '{model_name}': {e}")
        
        # Log summary
        logger.info(
            f"Populate summary: {len(results['inserted'])} inserted, "
            f"{len(results['skipped'])} skipped, {len(results['failed'])} failed"
        )
        
        return results


@schema
class Model(dj.Manual):
    """DeepLabCut Models applied to generate pose estimations.

    Attributes:
        model_name ( varchar(64) ): User-friendly model name.
        task ( varchar(32) ): Task in the config yaml.
        date ( varchar(16) ): Date in the config yaml.
        iteration (int): Iteration/version of this model.
        snapshotindex (int): Which snapshot for prediction (if -1, latest).
        shuffle (int): Which shuffle of the training dataset.
        trainingsetindex (int): Which training set fraction to generate model.
        engine (str): Engine used for model. Either 'tensorflow' or 'pytorch'.
        scorer ( varchar(64) ): Scorer/network name - DLC's GetScorerName().
        config_template (longblob): Dictionary of the config for analyze_videos().
        project_path ( varchar(255) ): DLC's project_path in config relative to root.
        model_prefix ( varchar(32) ): Optional. Prefix for model files.
        model_description ( varchar(300) ): Optional. User-entered description.
        TrainingParamSet (foreign key): Optional. Training parameters primary key.

    Note:
        Models are uniquely identified by the union of task, date, iteration, shuffle,
        snapshotindex, and trainingsetindex.
    """

    definition = """
    model_name           : varchar(64)  # User-friendly model name
    ---
    task                 : varchar(32)  # Task in the config yaml
    date                 : varchar(16)  # Date in the config yaml
    iteration            : int          # Iteration/version of this model
    snapshotindex        : int          # which snapshot for prediction (if -1, latest)
    shuffle              : int          # Shuffle (1) or not (0)
    trainingsetindex     : int          # Index of training fraction list in config.yaml
    engine='tensorflow'  : varchar(16)  # Engine used for model. Either 'tensorflow' or 'pytorch'
    unique index (task, date, iteration, shuffle, snapshotindex, trainingsetindex, engine)
    scorer               : varchar(64)  # Scorer/network name - DLC's GetScorerName()
    config_template      : longblob     # Dictionary of the config for analyze_videos()
    project_path         : varchar(255) # DLC's project_path in config relative to root
    model_prefix=''      : varchar(32)
    model_description='' : varchar(300)
    -> [nullable] train.TrainingParamSet
    """
    # project_path is the only item required downstream in the pose schema

    class BodyPart(dj.Part):
        """Body parts associated with a given model

        Attributes:
            body_part ( varchar(32) ): Short name. Also called joint.
            body_part_description ( varchar(1000) ): Optional. Longer description."""

        definition = """
        -> master
        -> BodyPart
        """

    @classmethod
    def insert_new_model(
        cls,
        model_name: str,
        dlc_config,
        *,
        shuffle: int,
        trainingsetindex,
        model_description="",
        model_prefix="",
        paramset_idx: int = None,
        prompt=True,
        params=None,
    ):
        """Insert new model into the dlc.Model table.

        Args:
            model_name (str): User-friendly name for this model.
            dlc_config ( varchar(255) ): Path to a config.y*ml.
            shuffle (int): Which shuffle of the training dataset.
            trainingsetindex (int): Index of training fraction list in config.yaml.
            model_description (str): Optional. Description of this model.
            model_prefix (str): Optional. Filename prefix used across DLC project
            paramset_idx (int): Optional. Index from the TrainingParamSet table
            prompt (bool): Optional, default True. Prompt the user with all info before inserting.
            params (dict): Optional. If dlc_config is path, dict of override items
        """
        # handle dlc_config being a yaml file
        dlc_config_fp = find_full_path(get_dlc_root_data_dir(), Path(dlc_config))
        assert dlc_config_fp.exists(), (
            "dlc_config is not a filepath" + f"\n Check: {dlc_config_fp}"
        )
        if dlc_config_fp.suffix in (".yml", ".yaml"):
            yaml = YAML(typ="safe", pure=True)
            with open(dlc_config_fp, "rb") as f:
                dlc_config = yaml.load(f)
        if isinstance(params, dict):
            dlc_config.update(params)

        # ---- Get and resolve project path ----
        project_path = dlc_config_fp.parent
        dlc_config["project_path"] = project_path.as_posix()  # update if different
        root_dir = find_root_directory(get_dlc_root_data_dir(), project_path)

        # ---- Verify config ----
        needed_attributes = [
            "Task",
            "date",
            "iteration",
            "snapshotindex",
            "TrainingFraction",
        ]
        for attribute in needed_attributes:
            assert attribute in dlc_config, f"Couldn't find {attribute} in config"

        engine = dlc_config.get("engine")
        if engine is None:
            logger.warning(
                "DLC engine not specified in config file. Defaulting to TensorFlow."
            )
            engine = "tensorflow"

        if engine == "tensorflow":
            from deeplabcut.utils.auxiliaryfunctions import GetScorerName  # isort:skip

            # ---- Get scorer name ----
            # "or 'f'" below covers case where config returns None. str_to_bool handles else
            scorer_legacy = str_to_bool(dlc_config.get("scorer_legacy", "f"))
            dlc_scorer = GetScorerName(
                cfg=dlc_config,
                shuffle=shuffle,
                trainFraction=dlc_config["TrainingFraction"][int(trainingsetindex)],
                modelprefix=model_prefix,
            )[scorer_legacy]
        elif engine == "pytorch":
            from deeplabcut.pose_estimation_pytorch.apis.utils import get_scorer_name

            dlc_scorer = get_scorer_name(
                cfg=dlc_config,
                shuffle=shuffle,
                train_fraction=dlc_config["TrainingFraction"][int(trainingsetindex)],
                modelprefix=model_prefix,
            )
        else:
            raise ValueError(f"Unknown engine type {engine}")

        if dlc_config["snapshotindex"] == -1:
            dlc_scorer = "".join(dlc_scorer.split("_")[:-1])

        # ---- Insert ----
        model_dict = {
            "model_name": model_name,
            "model_description": model_description,
            "scorer": dlc_scorer,
            "task": dlc_config["Task"],
            "date": dlc_config["date"],
            "iteration": dlc_config["iteration"],
            "snapshotindex": dlc_config["snapshotindex"],
            "shuffle": shuffle,
            "trainingsetindex": int(trainingsetindex),
            "engine": engine,
            "project_path": project_path.relative_to(root_dir).as_posix(),
            "paramset_idx": paramset_idx,
            "config_template": dlc_config,
        }

        # -- prompt for confirmation --
        if prompt:
            logger.info("--- DLC Model specification to be inserted ---")
            for k, v in model_dict.items():
                if k != "config_template":
                    logger.info("\t{}: {}".format(k, v))
                else:
                    logger.info("\t-- Template/Contents of config.yaml --")
                    for k, v in model_dict["config_template"].items():
                        logger.info("\t\t{}: {}".format(k, v))

        if (
            prompt
            and dj.utils.user_choice("Proceed with new DLC model insert?") != "yes"
        ):
            logger.info("Canceled insert.")
            return

        def _do_insert():
            cls.insert1(model_dict)
            # Returns array, so check size for unambiguous truth value
            if BodyPart.extract_new_body_parts(dlc_config, verbose=False).size > 0:
                BodyPart.insert_from_config(dlc_config, prompt=prompt)
            cls.BodyPart.insert((model_name, bp) for bp in dlc_config["bodyparts"])

        # ____ Insert into table ----
        if cls.connection.in_transaction:
            _do_insert()
        else:
            with cls.connection.transaction:
                _do_insert()

    @classmethod
    def insert_pretrained_model(
        cls,
        model_name: str,
        pretrained_model_name: str,
        *,
        model_description="",
        model_prefix="",
        prompt=True,
        config_overrides: dict = None,
    ):
        """Insert a pretrained model into the dlc.Model table.

        This method can only be used if the pretrained_model_name exists in the
        PretrainedModel lookup table. It handles config paths and training-related
        columns differently (set to NULL / "pretrained" as appropriate).

        Args:
            model_name (str): User-friendly name for this model instance.
            pretrained_model_name (str): Name from PretrainedModel lookup table.
            model_description (str): Optional. Description of this model.
            model_prefix (str): Optional. Filename prefix used across DLC project.
            prompt (bool): Optional, default True. Prompt the user with all info before inserting.
            config_overrides (dict): Optional. Dict of config items to override defaults.
        """
        # Check if pretrained model exists in lookup - return if not found
        if not PretrainedModel.is_pretrained(pretrained_model_name):
            logger.warning(
                f"Pretrained model '{pretrained_model_name}' not found in "
                "PretrainedModel lookup table. Cannot insert model. "
                "Please add it to PretrainedModel first."
            )
            return
        
        pretrained_info = (PretrainedModel & {"pretrained_model_name": pretrained_model_name}).fetch1()

        # Load default config from pretrained model
        default_params = pretrained_info.get("default_params") or {}
        if config_overrides:
            default_params.update(config_overrides)

        # Build config template - use defaults from pretrained model
        dlc_config = default_params.copy()
        
        # Set required fields for pretrained models
        # For pretrained models, we use placeholder values for training-related fields
        # Include model_name in task to ensure uniqueness across different model instances
        # This prevents duplicate key errors when inserting multiple instances of the same pretrained model
        # Task field is varchar(32), so we need to keep it short
        # Use a hash or shortened version of model_name to ensure uniqueness while staying within limit
        import hashlib
        model_name_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]  # First 8 chars of hash
        task_value = f"pt_{pretrained_model_name[:10]}_{model_name_hash}"  # Keep under 32 chars
        # Ensure it's exactly 32 chars or less
        task_value = task_value[:32]
        dlc_config.setdefault("Task", task_value)
        dlc_config.setdefault("date", "pretrained")
        dlc_config.setdefault("iteration", 0)
        dlc_config.setdefault("snapshotindex", -1)
        dlc_config.setdefault("TrainingFraction", [1.0])  # Placeholder
        
        engine = dlc_config.get("engine", "pytorch")
        if engine is None:
            logger.warning(
                "DLC engine not specified. Defaulting to PyTorch."
            )
            engine = "pytorch"

        # For pretrained models, scorer is based on the pretrained model name
        scorer = f"{pretrained_model_name}_pretrained"

        # Mark as pretrained in config_template for detection
        # Convention: _pretrained_model_name in config_template identifies pretrained models
        # This allows detection without modifying the Model table schema
        dlc_config["_is_pretrained"] = True
        dlc_config["_pretrained_model_name"] = pretrained_model_name
        
        # Build model dict - set training-related fields appropriately
        # For pretrained models: no project_path, minimal config_template
        model_dict = {
            "model_name": model_name,
            "model_description": model_description,
            "scorer": scorer,
            "task": dlc_config["Task"],
            "date": dlc_config["date"],
            "iteration": dlc_config["iteration"],
            "snapshotindex": dlc_config["snapshotindex"],
            "shuffle": 0,  # Not applicable for pretrained
            "trainingsetindex": 0,  # Not applicable for pretrained
            "engine": engine,
            "project_path": "",  # Empty for pretrained models
            "model_prefix": model_prefix,
            "paramset_idx": None,  # No training param set for pretrained
            "config_template": dlc_config,
        }

        # -- prompt for confirmation --
        if prompt:
            logger.info("--- Pretrained DLC Model specification to be inserted ---")
            for k, v in model_dict.items():
                if k != "config_template":
                    logger.info("\t{}: {}".format(k, v))
                else:
                    logger.info("\t-- Template/Contents of config.yaml --")
                    for ck, cv in model_dict["config_template"].items():
                        logger.info("\t\t{}: {}".format(ck, cv))

        if (
            prompt
            and dj.utils.user_choice("Proceed with pretrained DLC model insert?") != "yes"
        ):
            logger.info("Canceled insert.")
            return

        def _do_insert():
            cls.insert1(model_dict)
            # Extract body parts from config if available
            if "bodyparts" in dlc_config:
                if BodyPart.extract_new_body_parts(dlc_config, verbose=False).size > 0:
                    BodyPart.insert_from_config(dlc_config, prompt=prompt)
                cls.BodyPart.insert((model_name, bp) for bp in dlc_config["bodyparts"])

        # ____ Insert into table ----
        if cls.connection.in_transaction:
            _do_insert()
        else:
            with cls.connection.transaction:
                _do_insert()


@schema
class ModelEvaluation(dj.Computed):
    """Performance characteristics model calculated by `deeplabcut.evaluate_network`

    Attributes:
        Model (foreign key): Model name.
        train_iterations (int): Training iterations.
        train_error (float): Optional. Train error (px).
        test_error (float): Optional. Test error (px).
        p_cutoff (float): Optional. p-cutoff used.
        train_error_p (float): Optional. Train error with p-cutoff.
        test_error_p (float): Optional. Test error with p-cutoff."""

    definition = """
    -> Model
    ---
    train_iterations   : int   # Training iterations
    train_error=null   : float # Train error (px)
    test_error=null    : float # Test error (px)
    p_cutoff=null      : float # p-cutoff used
    train_error_p=null : float # Train error with p-cutoff
    test_error_p=null  : float # Test error with p-cutoff
    """

    def make(self, key):
        from deeplabcut import evaluate_network  # isort:skip
        from deeplabcut.utils.auxiliaryfunctions import (
            get_evaluation_folder,
        )  # isort:skip

        """.populate() method will launch evaluation for each unique entry in Model."""
        dlc_config, project_path, model_prefix, shuffle, trainingsetindex = (
            Model & key
        ).fetch1(
            "config_template",
            "project_path",
            "model_prefix",
            "shuffle",
            "trainingsetindex",
        )

        project_path = find_full_path(get_dlc_root_data_dir(), project_path)
        yml_path, _ = dlc_reader.read_yaml(project_path)

        evaluate_network(
            yml_path,
            Shuffles=[shuffle],  # this needs to be a list
            trainingsetindex=trainingsetindex,
            comparisonbodyparts="all",
        )

        eval_folder = get_evaluation_folder(
            trainFraction=dlc_config["TrainingFraction"][trainingsetindex],
            shuffle=shuffle,
            cfg=dlc_config,
            modelprefix=model_prefix,
        )
        eval_path = project_path / eval_folder
        assert eval_path.exists(), f"Couldn't find evaluation folder:\n{eval_path}"

        eval_csvs = list(eval_path.glob("*csv"))
        max_modified_time = 0
        for eval_csv in eval_csvs:
            modified_time = os.path.getmtime(eval_csv)
            if modified_time > max_modified_time:
                eval_csv_latest = eval_csv
        with open(eval_csv_latest, newline="") as f:
            results = list(csv.DictReader(f, delimiter=","))[0]
        # in testing, test_error_p returned empty string
        self.insert1(
            dict(
                key,
                train_iterations=results["Training iterations:"],
                train_error=results[" Train error(px)"],
                test_error=results[" Test error(px)"],
                p_cutoff=results["p-cutoff used"],
                train_error_p=results["Train error with p-cutoff"],
                test_error_p=results["Test error with p-cutoff"],
            )
        )


@schema
class PoseEstimationTask(dj.Manual):
    """Staging table for pairing of video recording and model before inference.

    Attributes:
        VideoRecording (foreign key): Video recording key.
        Model (foreign key): Model name.
        task_mode (load or trigger): Optional. Default load. Or trigger computation.
        pose_estimation_output_dir ( varchar(255) ): Optional. Output dir relative to
                                                     get_dlc_root_data_dir.
        pose_estimation_params (longblob): Optional. Params for DLC's analyze_videos
                                           params, if not default."""

    definition = """
    -> VideoRecording                           # Session -> Recording + File part table
    -> Model                                    # Must specify a DLC project_path
    ---
    task_mode='load' : enum('load', 'trigger')  # load results or trigger computation
    pose_estimation_output_dir='': varchar(255) # output dir relative to the root dir
    pose_estimation_params=null  : longblob     # analyze_videos params, if not default
    """

    @classmethod
    def infer_output_dir(cls, key: dict, relative: bool = False, mkdir: bool = False):
        """Return the expected pose_estimation_output_dir.

        Spaces in model name are replaced with hyphens.
        Based on convention: / video_dir / Device_{}_Recording_{}_Model_{}

        Args:
            key: DataJoint key specifying a pairing of VideoRecording and Model.
            relative (bool): Report directory relative to get_dlc_processed_data_dir().
            mkdir (bool): Default False. Make directory if it doesn't exist.
        """
        root_dirs = get_dlc_root_data_dir()
        if not root_dirs:
            raise ValueError(
                "DLC_ROOT_DATA_DIR is not configured. "
                "Please set DLC_ROOT_DATA_DIR environment variable or configure it in dj_local_conf.json"
            )
        
        # Get the stored file path from the database
        stored_file_path = (VideoRecording.File & key).fetch("file_path", limit=1)[0]
        
        try:
            video_filepath = find_full_path(root_dirs, stored_file_path)
        except FileNotFoundError as e:
            # Provide more helpful error message with diagnostic information
            error_msg = (
                f"Could not find video file: {stored_file_path}\n"
                f"Searched in root directories: {root_dirs}\n"
            )
            # Check if any files exist in the root directories
            for root in root_dirs:
                root_path = Path(root)
                if root_path.exists():
                    video_files = list(root_path.glob("*.mp4")) + list(root_path.glob("*.avi")) + list(root_path.glob("*.mov"))
                    if video_files:
                        error_msg += f"\nFound {len(video_files)} video file(s) in {root}:\n"
                        for vf in video_files[:5]:  # Show first 5
                            error_msg += f"  - {vf.name}\n"
                        if len(video_files) > 5:
                            error_msg += f"  ... and {len(video_files) - 5} more\n"
                    else:
                        error_msg += f"\nNo video files found in {root}\n"
                else:
                    error_msg += f"\nRoot directory does not exist: {root}\n"
            
            # Check if the stored path is absolute and exists
            stored_path = Path(stored_file_path)
            if stored_path.is_absolute() and stored_path.exists():
                # File exists at absolute path but not under any root directory
                # Use it directly as a fallback
                logger.warning(
                    f"Video file {stored_file_path} exists at absolute path but is not under any configured root directory. "
                    f"Using absolute path directly."
                )
                video_filepath = stored_path
            else:
                # Check if the stored path is absolute
                if stored_path.is_absolute():
                    error_msg += (
                        f"\nNote: Stored path is absolute: {stored_file_path}\n"
                        "If the file exists at this absolute path, it may not be under any configured root directory.\n"
                    )
                    if stored_path.exists():
                        error_msg += f"The file exists at this absolute path, but it's not under any root directory.\n"
                
                raise FileNotFoundError(error_msg) from e
        
        # Ensure video_filepath is an absolute Path
        video_filepath = Path(video_filepath).resolve()
        
        # Find the root directory that contains this video file
        root_dir = None
        video_parent = video_filepath.parent
        
        # First, check if the video file itself is directly in a root directory
        for root in root_dirs:
            root_path = Path(root).resolve()
            if video_filepath.parent == root_path:
                root_dir = root_path
                break
        
        # If not found, check if video file is in a subdirectory of a root directory
        if root_dir is None:
            for root in root_dirs:
                root_path = Path(root).resolve()
                try:
                    # Check if video_filepath is under this root
                    video_filepath.relative_to(root_path)
                    root_dir = root_path
                    break
                except ValueError:
                    # video_filepath is not under this root, continue
                    continue
        
        # If still not found, try find_root_directory on the parent directory
        if root_dir is None:
            try:
                root_dir = Path(find_root_directory(root_dirs, video_filepath.parent)).resolve()
            except FileNotFoundError:
                # Last resort: if video is not under any root, use the first root directory
                # This handles edge cases where the video path might be absolute but outside roots
                logger.warning(
                    f"Video file {video_filepath} is not under any configured root directory. "
                    f"Using first root directory as fallback: {root_dirs[0]}"
                )
                root_dir = Path(root_dirs[0]).resolve()
        recording_key = VideoRecording & key
        device = "-".join(
            str(v)
            for v in (_linking_module.Device & recording_key).fetch1("KEY").values()
        )
        if get_dlc_processed_data_dir():
            processed_dir = Path(get_dlc_processed_data_dir())
        else:  # if processed not provided, default to where video is
            processed_dir = root_dir

        # Calculate relative path from root_dir to video's parent directory
        try:
            video_relative_path = video_filepath.parent.relative_to(root_dir)
        except ValueError:
            # Video is not under root_dir (edge case - should be rare)
            # Use video's parent directory name as the relative path
            logger.warning(
                f"Video {video_filepath} is not under root directory {root_dir}. "
                f"Using parent directory name as relative path."
            )
            video_relative_path = Path(video_filepath.parent.name)

        output_dir = (
            processed_dir
            / video_relative_path
            / (
                f'device_{device}_recording_{key["recording_id"]}_model_'
                + key["model_name"].replace(" ", "-")
            )
        )
        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir.relative_to(processed_dir) if relative else output_dir

    @classmethod
    def generate(
        cls,
        video_recording_key: dict,
        model_name: str,
        *,
        task_mode: str = None,
        analyze_videos_params: dict = None,
    ):
        """Insert PoseEstimationTask in inferred output dir.

        Based on the convention / video_dir / device_{}_recording_{}_model_{}

        Args:
            video_recording_key (dict): DataJoint key specifying a VideoRecording.

            model_name (str): Name of DLC model (from Model table) to be used for inference.
            task_mode (str): Default 'trigger' computation. Or 'load' existing results.
            analyze_videos_params (dict): Optional. Parameters passed to DLC's analyze_videos:
                videotype, gputouse, save_as_csv, batchsize, cropping, TFGPUinference,
                dynamic, robust_nframes, allow_growth, use_shelve
        """
        output_dir = cls.infer_output_dir(
            {**video_recording_key, "model_name": model_name},
            relative=False,
            mkdir=True,
        )
        
        # Get processed_dir for relative path calculation
        processed_dir = get_dlc_processed_data_dir()
        if processed_dir is None or processed_dir == "":
            # If no processed_dir, use root_dir (same logic as infer_output_dir)
            root_dirs = get_dlc_root_data_dir()
            if not root_dirs:
                raise ValueError(
                    "DLC_ROOT_DATA_DIR is not configured. "
                    "Please set DLC_ROOT_DATA_DIR environment variable or configure it in dj_local_conf.json"
                )
            video_filepath = find_full_path(
                root_dirs,
                (VideoRecording.File & {**video_recording_key}).fetch("file_path", limit=1)[0],
            )
            video_filepath = Path(video_filepath).resolve()
            video_parent = video_filepath.parent
            processed_dir = None
            for root in root_dirs:
                root_path = Path(root).resolve()
                if video_parent == root_path:
                    processed_dir = root_path
                    break
            if processed_dir is None:
                root_dir_result = find_root_directory(root_dirs, video_filepath.parent)
                if root_dir_result is None:
                    raise ValueError(
                        f"Could not determine root directory for video file: {video_filepath}"
                    )
                processed_dir = Path(root_dir_result).resolve()
        else:
            processed_dir = Path(processed_dir)
        
        # Ensure processed_dir is not None before using it
        if processed_dir is None:
            raise ValueError(
                "Could not determine processed data directory. "
                "Please configure DLC_PROCESSED_DATA_DIR or ensure DLC_ROOT_DATA_DIR is set correctly."
        )

        if task_mode is None:
            # Check if results exist by looking for result files directly (more reliable)
            output_path = Path(output_dir)
            results_exist = False
            if output_path.exists():
                # Check for result files (H5, pickle, or JSON)
                h5_files = list(output_path.glob("*.h5"))
                pickle_files = list(output_path.glob("*.pickle"))
                json_files = list(output_path.glob("*.json"))
                if h5_files or pickle_files or json_files:
                    results_exist = True
                    logger.info(
                        f"Found existing results in {output_dir}: "
                        f"{len(h5_files)} H5, {len(pickle_files)} pickle, {len(json_files)} JSON files"
                    )
            
            # Also try the reader as a fallback
            if not results_exist:
                try:
                    _ = dlc_reader.PoseEstimation(output_dir)
                    results_exist = True
                    logger.info(f"Found existing results via dlc_reader in {output_dir}")
                except (FileNotFoundError, Exception) as e:
                    logger.debug(f"No results found via dlc_reader in {output_dir}: {e}")
            
            task_mode = "load" if results_exist else "trigger"
            logger.info(f"Auto-detected task_mode='{task_mode}' for {video_recording_key} (output_dir: {output_dir})")

        cls.insert1(
            {
                **video_recording_key,
                "model_name": model_name,
                "task_mode": task_mode,
                "pose_estimation_params": analyze_videos_params,
                "pose_estimation_output_dir": output_dir.relative_to(
                    processed_dir
                ).as_posix(),
            },
            skip_duplicates=True,
        )

    insert_estimation_task = generate


@schema
class PoseEstimation(dj.Computed):
    """Results of pose estimation.

    Attributes:
        PoseEstimationTask (foreign key): Pose Estimation Task key.
        post_estimation_time (datetime): time of generation of this set of DLC results.
    """

    definition = """
    -> PoseEstimationTask
    ---
    pose_estimation_time: datetime  # time of generation of this set of DLC results
    """

    class Individual(dj.Part):
        """Individuals/animals tracked in this pose estimation.
        
        For single-animal data, this table will be empty.
        For multi-animal data, each individual is tracked separately.
        
        Attributes:
            PoseEstimation (foreign key): Pose Estimation key.
            individual_id (varchar): Individual/animal identifier (e.g., 'animal0', 'animal1').
        """
        
        definition = """
        -> master
        ---
        individual_id : varchar(32)  # Individual/animal identifier (e.g., 'animal0', 'animal1')
    """

    class BodyPartPosition(dj.Part):
        """Position of individual body parts by frame index

        Attributes:
            PoseEstimation (foreign key): Pose Estimation key.
            Model.BodyPart (foreign key): Body Part key.
            frame_index (longblob): Frame index in model.
            x_pos (longblob): X position.
            y_pos (longblob): Y position.
            z_pos (longblob): Optional. Z position.
            likelihood (longblob): Model confidence."""

        definition = """ # uses DeepLabCut h5 output for body part position
        -> master
        -> Model.BodyPart
        ---
        frame_index : longblob     # frame index in model
        x_pos       : longblob
        y_pos       : longblob
        z_pos=null  : longblob
        likelihood  : longblob
        """
    
    class IndividualMapping(dj.Part):
        """Maps body part positions to individuals for multi-animal tracking.
        
        For single-animal data, this table will be empty.
        For multi-animal data, links BodyPartPosition entries to individuals.
        Note: In multi-animal data, each individual has separate position data,
        so we encode the individual in a unique identifier.
        
        Attributes:
            PoseEstimation (foreign key): Pose Estimation key.
            body_part (varchar): Body part name (from BodyPartPosition, via Model.BodyPart).
            individual_id (varchar): Individual identifier (must match Individual.individual_id).
        """
        
        definition = """
        -> master
        body_part: varchar(32)  # Body part name (must match BodyPartPosition.body_part)
        individual_id: varchar(32)  # Individual identifier (must match Individual.individual_id)
        """

    @staticmethod
    def _patch_dlclibrary_modelzoo_bug():
        """Monkey patch to fix dlclibrary ModelZoo download bug.
        
        The bug: In dlclibrary.dlcmodelzoo.modelzoo_download._handle_downloaded_file,
        rename_mapping is sometimes a string instead of a dict, causing AttributeError.
        
        This patch ensures rename_mapping is always treated as a dict.
        """
        try:
            import dlclibrary.dlcmodelzoo.modelzoo_download as modelzoo_download
            original_handle = modelzoo_download._handle_downloaded_file
            
            def patched_handle_downloaded_file(file_name, target_dir, rename_mapping):
                """Patched version that handles rename_mapping being a string."""
                # Fix: If rename_mapping is a string, convert to dict or use empty dict
                if isinstance(rename_mapping, str):
                    logger.warning(
                        f"dlclibrary bug: rename_mapping is a string '{rename_mapping}' instead of dict. "
                        "Using empty dict as fallback."
                    )
                    rename_mapping = {}
                elif rename_mapping is None:
                    rename_mapping = {}
                
                # Call original function with fixed rename_mapping
                return original_handle(file_name, target_dir, rename_mapping)
            
            # Apply the patch
            modelzoo_download._handle_downloaded_file = patched_handle_downloaded_file
            logger.debug("Applied dlclibrary ModelZoo bug patch")
            return True
        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not patch dlclibrary (may not be needed): {e}")
            return False

    @classmethod
    def _do_pretrained_inference(
        cls,
        pretrained_model_name: str,
        video_filepaths: list,
        output_dir: Path,
        inference_params: dict = None,
    ):
        """Run pretrained (SuperAnimal / Model Zoo) inference on videos.

        This uses the PretrainedModel lookup as the single source of truth for:
        - which pretrained model to call
        - default inference parameters
        - optional backbone/detector names, etc.

        It supports DLC's `video_inference_superanimal` API when available,
        and falls back to `video_inference` if exposed by the installed DLC version.

        Args:
            pretrained_model_name: Name of the pretrained model (e.g., "superanimal_quadruped").
            video_filepaths: List of full paths to video files.
            output_dir: Directory to save output files.
            inference_params: Optional. Parameters for inference function (overrides defaults).
        """
        import inspect
        import deeplabcut
        
        # Apply monkey patch to fix dlclibrary bug before inference
        cls._patch_dlclibrary_modelzoo_bug()

        # --- Fetch pretrained model metadata from lookup ---
        try:
            pm = (PretrainedModel & {"pretrained_model_name": pretrained_model_name}).fetch1()
        except dj.DataJointError:
            raise ValueError(
                f"Pretrained model '{pretrained_model_name}' is not registered in PretrainedModel. "
                "Please insert it before running pretrained inference."
            )

        default_params = pm.get("default_params") or {}
        merged_params = {**default_params, **(inference_params or {})}

        backbone_model_name = pm.get("backbone_model_name") or None
        detector_name = pm.get("detector_name") or None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        destfolder_str = str(output_dir)

        # --- Prefer SuperAnimal-style API if available ---
        if hasattr(deeplabcut, "video_inference_superanimal"):
            inference_func = deeplabcut.video_inference_superanimal
            sig = inspect.signature(inference_func)
            param_names = list(sig.parameters.keys())

            kwargs = {
                k: v
                for k, v in merged_params.items()
                if k in sig.parameters
            }

            # Set dest_folder (note: DLC 3.x uses dest_folder, not destfolder)
            if "dest_folder" in sig.parameters:
                kwargs.setdefault("dest_folder", destfolder_str)
            elif "destfolder" in sig.parameters:
                kwargs.setdefault("destfolder", destfolder_str)

            # DLC 3.x signature: video_inference_superanimal(videos, superanimal_name, model_name, ...)
            # model_name is required, so we need to provide it
            if not backbone_model_name:
                # If no backbone_model_name in metadata, use a default or raise error
                logger.warning(
                    f"No backbone_model_name found for pretrained model '{pretrained_model_name}'. "
                    "Using default 'superanimal' model name."
                )
                backbone_model_name = "superanimal"

            # Set detector_name if available
            if detector_name and "detector_name" in sig.parameters:
                kwargs.setdefault("detector_name", detector_name)

            logger.info(
                f"Running video_inference_superanimal with "
                f"superanimal_name={pretrained_model_name}, "
                f"model_name={backbone_model_name}, "
                f"{len(video_filepaths)} videos, "
                f"dest_folder={destfolder_str}, "
                f"kwargs={kwargs}"
            )
            
            # Verify output directory exists and is unique
            if not Path(destfolder_str).exists():
                logger.warning(f"Output directory does not exist, creating: {destfolder_str}")
                Path(destfolder_str).mkdir(parents=True, exist_ok=True)
            logger.info(f"Output will be saved to: {destfolder_str}")

            # Call with correct signature: videos, superanimal_name, model_name, **kwargs
            try:
                result = inference_func(
                    video_filepaths,
                    pretrained_model_name,  # superanimal_name (positional, required)
                    backbone_model_name,    # model_name (positional, required)
                    **kwargs,
                )
            except ValueError as e:
                error_msg = str(e)
                if "need at least one array to stack" in error_msg or "at least one array" in error_msg.lower():
                    # No animals detected in the video
                    logger.warning(
                        f"No animals detected in video(s): {video_filepaths}. "
                        "This can happen if: "
                        "1) The detector threshold is too high (try lowering bbox_threshold), "
                        "2) The animals are too small or not visible, "
                        "3) The video quality is poor, or "
                        "4) The model is not suitable for this video type."
                    )
                    logger.info(
                        "No animals detected. The pipeline will skip this recording gracefully. "
                        "No pose estimation data will be inserted for this video."
                    )
                    
                    # Return None to indicate no detections
                    # Downstream code will check for result files and skip if none exist
                    return None
                else:
                    # Different ValueError, re-raise it
                    raise

            # Verify files were saved to the correct location
            output_path = Path(destfolder_str)
            if output_path.exists():
                saved_files = list(output_path.glob("*.h5")) + list(output_path.glob("*.pickle"))
                logger.info(f"Saved {len(saved_files)} result file(s) to {destfolder_str}")
            else:
                logger.warning(f"Output directory {destfolder_str} does not exist after inference!")
            
            return result

        # --- Fallback: our own generic video_inference API, if present ---
        if hasattr(deeplabcut, "video_inference"):
            inference_func = deeplabcut.video_inference
            sig = inspect.signature(inference_func)

            kwargs = {
                k: v
                for k, v in merged_params.items()
                if k in sig.parameters
            }

            if "model_name" in sig.parameters and backbone_model_name:
                kwargs.setdefault("model_name", backbone_model_name)
            if "detector_name" in sig.parameters and detector_name:
                kwargs.setdefault("detector_name", detector_name)
            if "destfolder" in sig.parameters:
                kwargs.setdefault("destfolder", destfolder_str)

            logger.info(
                f"Running video_inference (fallback) with pretrained_model_name={pretrained_model_name}, "
                f"{len(video_filepaths)} videos, kwargs={kwargs}"
            )

            return inference_func(
                video_filepaths,
                **kwargs,
            )

        raise NotImplementedError(
            "No compatible pretrained inference function found in the installed DeepLabCut. "
            "Expected `video_inference_superanimal` or a compatible `video_inference` wrapper."
        )

    @staticmethod
    def _sanitize_pytorch_config_yaml(project_path: Path, dlc_config: dict, dlc_model_: dict):
        """Sanitize pytorch_config.yaml files by removing ruamel.yaml-specific tags.
        
        DeepLabCut's training process creates pytorch_config.yaml files with ruamel.yaml
        round-trip mode tags that can't be read by the safe YAML loader. This function
        finds and sanitizes these files by reading with ruamel.yaml and rewriting with
        a safe YAML writer.
        
        Args:
            project_path: Full path to the directory containing the trained model.
            dlc_config: DeepLabCut config dictionary.
            dlc_model_: Model record dictionary.
        """
        def _convert_to_plain_python(obj):
            """Recursively convert ruamel.yaml objects to plain Python types."""
            from ruamel.yaml.comments import CommentedMap, CommentedSeq
            if isinstance(obj, CommentedMap):
                return {k: _convert_to_plain_python(v) for k, v in obj.items()}
            elif isinstance(obj, CommentedSeq):
                return [_convert_to_plain_python(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _convert_to_plain_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert_to_plain_python(item) for item in obj]
            else:
                return obj
        
        try:
            from deeplabcut.utils.auxiliaryfunctions import get_model_folder
        except ImportError:
            try:
                from deeplabcut.utils.auxiliaryfunctions import GetModelFolder as get_model_folder
            except ImportError:
                logger.warning("Could not import get_model_folder, skipping pytorch_config.yaml sanitization")
                return
        
        # Find the model training folder
        try:
            model_folder = get_model_folder(
                trainFraction=dlc_config.get("TrainingFraction", [0.95])[dlc_model_.get("trainingsetindex", 0)],
                shuffle=dlc_model_.get("shuffle", 1),
                cfg=dlc_config,
                modelprefix=dlc_model_.get("model_prefix", ""),
            )
            model_train_folder = project_path / model_folder / "train"
        except Exception as e:
            logger.warning(f"Could not determine model folder: {e}. Searching for pytorch_config.yaml files...")
            model_train_folder = None
        
        # Search for pytorch_config.yaml files in common locations
        search_paths = []
        if model_train_folder and model_train_folder.exists():
            search_paths.append(model_train_folder / "pytorch_config.yaml")
        
        # Also search in dlc-models-pytorch directories
        dlc_models_pytorch = project_path / "dlc-models-pytorch"
        if dlc_models_pytorch.exists():
            for iteration_dir in dlc_models_pytorch.glob("iteration-*"):
                for model_dir in iteration_dir.glob("*"):
                    train_dir = model_dir / "train"
                    if train_dir.exists():
                        search_paths.append(train_dir / "pytorch_config.yaml")
        
        # Sanitize each found pytorch_config.yaml file
        for config_path in search_paths:
            if not config_path.exists():
                continue
            
            try:
                # First, check if file can be read by DLC's read_config_as_dict
                # If it can, and doesn't have ruamel tags, skip sanitization
                try:
                    from deeplabcut.core import config as config_utils
                    test_read = config_utils.read_config_as_dict(str(config_path))
                    if test_read is not None and "method" in test_read:
                        # File is already readable by DLC, check if it has ruamel tags
                        with open(config_path, "r") as f:
                            content = f.read()
                        if "!!python/object/new:ruamel.yaml" not in content:
                            logger.debug(f"pytorch_config.yaml is already valid, skipping sanitization: {config_path}")
                            continue
                except (ImportError, Exception):
                    # Can't verify with DLC, proceed with sanitization
                    pass
                
                # Read with ruamel.yaml round-trip mode (can handle the tags)
                yaml_rt = YAML(typ="rt")  # round-trip mode
                with open(config_path, "r") as f:
                    config_data = yaml_rt.load(f)
                
                if config_data is None:
                    logger.warning(f"pytorch_config.yaml is empty or invalid: {config_path}")
                    continue
                
                # Convert ruamel.yaml objects to plain Python types
                config_dict = _convert_to_plain_python(config_data)
                
                # Validate required keys are present and add defaults if missing
                if not isinstance(config_dict, dict):
                    logger.error(f"pytorch_config.yaml is not a dict after conversion: {type(config_dict)}")
                    continue
                
                # Ensure required keys are present
                if "method" not in config_dict:
                    config_dict["method"] = "bu"  # bottom-up (default)
                    logger.info(f"Added default method='bu' to {config_path}")
                
                # Ensure metadata section exists (required by DLC)
                if "metadata" not in config_dict:
                    config_dict["metadata"] = {}
                    logger.info(f"Added default metadata section to {config_path}")
                
                # Ensure metadata has required fields
                if "bodyparts" not in config_dict.get("metadata", {}):
                    # Try to get from dlc_config if available
                    bodyparts = dlc_config.get("bodyparts", ["bodypart1", "bodypart2", "bodypart3"])
                    config_dict.setdefault("metadata", {})["bodyparts"] = bodyparts
                    logger.info(f"Added bodyparts to metadata in {config_path}")
                
                # Write back with safe YAML writer
                # Create backup before overwriting
                backup_path = config_path.with_suffix('.yaml.backup')
                try:
                    import shutil
                    shutil.copy2(config_path, backup_path)
                except Exception:
                    pass  # Backup is optional
                
                # Use DLC's own write_config if available, otherwise use ruamel.yaml safe mode
                yaml_safe = None
                try:
                    from deeplabcut.utils.auxiliaryfunctions import write_config
                    # DLC's write_config handles the format correctly
                    write_config(str(config_path), config_dict)
                    logger.debug(f"Used DLC's write_config to sanitize: {config_path}")
                except (ImportError, Exception) as write_err:
                    # Fallback to ruamel.yaml safe mode
                    logger.debug(f"DLC's write_config not available, using ruamel.yaml safe mode: {write_err}")
                    yaml_safe = YAML(typ="safe", pure=True)
                    yaml_safe.default_flow_style = False
                    
                    with open(config_path, "w") as f:
                        yaml_safe.dump(config_dict, f)
                
                # Verify the sanitized file can be read back by both our YAML loader and DLC's
                try:
                    # Test with our YAML loader (if we used ruamel.yaml, otherwise just test with DLC)
                    if yaml_safe is not None:
                        with open(config_path, "r") as f:
                            test_load = yaml_safe.load(f)
                        if test_load is None or "method" not in test_load:
                            logger.error(f"Sanitized file is invalid (missing method), restoring backup: {config_path}")
                            if backup_path.exists():
                                shutil.copy2(backup_path, config_path)
                            continue
                    
                    # Test with DLC's read_config_as_dict (the one that will actually be used)
                    try:
                        from deeplabcut.core import config as config_utils
                        dlc_test = config_utils.read_config_as_dict(str(config_path))
                        if dlc_test is None:
                            logger.error(
                                f"DLC's read_config_as_dict returned None for sanitized file: {config_path}. "
                                "Restoring backup."
                            )
                            if backup_path.exists():
                                shutil.copy2(backup_path, config_path)
                            continue
                        if "method" not in dlc_test:
                            logger.error(
                                f"DLC's read_config_as_dict missing 'method' key: {config_path}. "
                                "Restoring backup."
                            )
                            if backup_path.exists():
                                shutil.copy2(backup_path, config_path)
                            continue
                        logger.debug(f"Verified sanitized file can be read by DLC: {config_path}")
                    except ImportError:
                        # DLC's config_utils not available, skip DLC verification
                        pass
                    except Exception as e:
                        logger.warning(f"DLC verification failed (but file structure looks OK): {e}")
                        # Don't restore backup if our YAML loader can read it
                        # The DLC error might be for other reasons
                except Exception as e:
                    logger.error(f"Sanitized file verification failed: {e}. Restoring backup.")
                    if backup_path.exists():
                        shutil.copy2(backup_path, config_path)
                    continue
                
                logger.info(f"Sanitized pytorch_config.yaml: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to sanitize {config_path}: {e}")
                # Continue with other files even if one fails

    @classmethod
    def do_trained(
        cls,
        project_path: Path,
        video_filepaths: list,
        output_dir: Path,
        dlc_config: dict,
        dlc_model_: dict,
        analyze_video_params: dict = None,
    ):
        """Run trained model inference on videos.
        
        Args:
            project_path: Full path to the directory containing the trained model.
            video_filepaths: List of full paths to video files.
            output_dir: Directory to save output files.
            dlc_config: DeepLabCut config dictionary.
            dlc_model_: Model record dictionary.
            analyze_video_params: Optional. Parameters for analyze_videos function.
        """
        import inspect
        
        # Validate project_path is not empty for trained models
        if not project_path or str(project_path).strip() == "":
            raise ValueError(
                "project_path cannot be empty for trained models. "
                "Trained models require a valid project directory path."
            )
        
        if analyze_video_params is None:
            analyze_video_params = {}
        
        engine = dlc_model_.get("engine")
        if engine is None:
            logger.warning(
                "DLC engine not specified in config file. Defaulting to TensorFlow."
            )
            engine = "tensorflow"
        if engine == "pytorch":
            from deeplabcut.pose_estimation_pytorch import analyze_videos
            # Sanitize pytorch_config.yaml files before inference
            cls._sanitize_pytorch_config_yaml(project_path, dlc_config, dlc_model_)
            
            # Verify pytorch_config.yaml files are readable after sanitization
            try:
                from deeplabcut.utils.auxiliaryfunctions import get_model_folder
                from deeplabcut.core import config as config_utils
            except ImportError:
                try:
                    from deeplabcut.utils.auxiliaryfunctions import GetModelFolder as get_model_folder
                except ImportError:
                    get_model_folder = None
            
            if get_model_folder:
                try:
                    model_folder = get_model_folder(
                        trainFraction=dlc_config.get("TrainingFraction", [0.95])[dlc_model_.get("trainingsetindex", 0)],
                        shuffle=dlc_model_.get("shuffle", 1),
                        cfg=dlc_config,
                        modelprefix=dlc_model_.get("model_prefix", ""),
                    )
                    model_train_folder = project_path / model_folder / "train"
                    pytorch_config_path = model_train_folder / "pytorch_config.yaml"
                    
                    if pytorch_config_path.exists():
                        # Test if DLC can read it
                        try:
                            test_cfg = config_utils.read_config_as_dict(str(pytorch_config_path))
                            if test_cfg is None:
                                logger.error(
                                    f"pytorch_config.yaml exists but read_config_as_dict returned None: {pytorch_config_path}. "
                                    "File may be corrupted. Check the file manually."
                                )
                            elif "method" not in test_cfg:
                                logger.error(
                                    f"pytorch_config.yaml missing 'method' key: {pytorch_config_path}. "
                                    "This will cause inference to fail."
                                )
                        except Exception as e:
                            logger.error(
                                f"Failed to read pytorch_config.yaml with DLC's read_config_as_dict: {e}. "
                                f"File: {pytorch_config_path}"
                            )
                except Exception as e:
                    logger.debug(f"Could not verify pytorch_config.yaml: {e}")
        elif engine == "tensorflow":
            from deeplabcut.pose_estimation_tensorflow import analyze_videos
        else:
            raise ValueError(f"Unknown engine type {engine}")

        # ---- Update pytorch_config.yaml batch_size if provided (for PyTorch) ----
        # This must happen BEFORE we write the main config file
        if engine == "pytorch" and analyze_video_params:
            batch_size_override = analyze_video_params.get("batch_size") or analyze_video_params.get("batchsize")
            
            if batch_size_override is not None:
                try:
                    from deeplabcut.utils.auxiliaryfunctions import get_model_folder
                    from deeplabcut.core import config as config_utils
                    from deeplabcut.utils.auxiliaryfunctions import write_config
                except ImportError:
                    try:
                        from deeplabcut.utils.auxiliaryfunctions import GetModelFolder as get_model_folder
                    except ImportError:
                        get_model_folder = None
                        config_utils = None
                        write_config = None
                
                if get_model_folder and config_utils and write_config:
                    try:
                        model_folder = get_model_folder(
                            trainFraction=dlc_config.get("TrainingFraction", [0.95])[dlc_model_.get("trainingsetindex", 0)],
                            shuffle=dlc_model_.get("shuffle", 1),
                            cfg=dlc_config,
                            modelprefix=dlc_model_.get("model_prefix", ""),
                        )
                        model_train_folder = project_path / model_folder / "train"
                        pytorch_config_path = model_train_folder / "pytorch_config.yaml"
                        
                        if pytorch_config_path.exists():
                            # Read current config
                            pytorch_cfg = config_utils.read_config_as_dict(str(pytorch_config_path))
                            if pytorch_cfg and pytorch_cfg.get("batch_size") != batch_size_override:
                                # Update batch_size in config
                                pytorch_cfg["batch_size"] = batch_size_override
                                # Write back using DLC's write_config
                                write_config(str(pytorch_config_path), pytorch_cfg)
                                logger.info(f"Updated batch_size to {batch_size_override} in {pytorch_config_path}")
                    except Exception as e:
                        logger.debug(f"Could not update pytorch_config.yaml batch_size: {e}")

        # ---- Build and save DLC configuration (yaml) file ----
        dlc_project_path = Path(project_path)
        dlc_config["project_path"] = dlc_project_path.as_posix()

        # ---- Special handling for "cropping" ----
        # `analyze_videos` behavior:
        #   i) if is None, use the "cropping" from the config file
        #   ii) if defined, use the specified "cropping" values but not updating the config file
        # new behavior: if defined as "False", overwrite "cropping" to False in config file
        cropping = analyze_video_params.get("cropping", None)
        if cropping is not None:
            if cropping:
                dlc_config["cropping"] = True
                (
                    dlc_config["x1"],
                    dlc_config["x2"],
                    dlc_config["y1"],
                    dlc_config["y2"],
                ) = cropping
            else:  # cropping is False
                dlc_config["cropping"] = False

        # ---- Write config files ----
        config_filename = f"dj_dlc_config_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}.yaml"
        # To output dir: Important for loading/parsing output in datajoint
        _ = dlc_reader.save_yaml(output_dir, dlc_config)
        # To project dir: Required by DLC to run the analyze_videos
        if dlc_project_path != output_dir:
            config_filepath = dlc_reader.save_yaml(
                dlc_project_path,
                dlc_config,
                filename=config_filename,
            )
        else:
            config_filepath = output_dir / config_filename

        # ---- Final verification of pytorch_config.yaml (for PyTorch) ----
        # This is a final check before calling analyze_videos to ensure the file is valid
        if engine == "pytorch":
            try:
                from deeplabcut.utils.auxiliaryfunctions import get_model_folder
                from deeplabcut.core import config as config_utils
            except ImportError:
                try:
                    from deeplabcut.utils.auxiliaryfunctions import GetModelFolder as get_model_folder
                except ImportError:
                    get_model_folder = None
                    config_utils = None
            
            if get_model_folder and config_utils:
                try:
                    model_folder = get_model_folder(
                        trainFraction=dlc_config.get("TrainingFraction", [0.95])[dlc_model_.get("trainingsetindex", 0)],
                        shuffle=dlc_model_.get("shuffle", 1),
                        cfg=dlc_config,
                        modelprefix=dlc_model_.get("model_prefix", ""),
                    )
                    model_train_folder = project_path / model_folder / "train"
                    pytorch_config_path = model_train_folder / "pytorch_config.yaml"
                    
                    if not pytorch_config_path.exists():
                        # Search for pytorch_config.yaml in alternative locations
                        search_paths = []
                        
                        # Check if model_train_folder exists
                        if model_train_folder.exists():
                            search_paths.append(model_train_folder)
                        
                        # Search in dlc-models-pytorch directories
                        dlc_models_pytorch = project_path / "dlc-models-pytorch"
                        if dlc_models_pytorch.exists():
                            for iteration_dir in dlc_models_pytorch.glob("iteration-*"):
                                for model_dir in iteration_dir.glob("*"):
                                    train_dir = model_dir / "train"
                                    if train_dir.exists():
                                        search_paths.append(train_dir)
                        
                        # Also search in dlc-models directories (TensorFlow-style path)
                        dlc_models = project_path / "dlc-models"
                        if dlc_models.exists():
                            for iteration_dir in dlc_models.glob("iteration-*"):
                                for model_dir in iteration_dir.glob("*"):
                                    train_dir = model_dir / "train"
                                    if train_dir.exists():
                                        search_paths.append(train_dir)
                        
                        # Search for the file in all candidate directories
                        found_path = None
                        for search_dir in search_paths:
                            candidate = search_dir / "pytorch_config.yaml"
                            if candidate.exists():
                                found_path = candidate
                                logger.info(f"Found pytorch_config.yaml at alternative location: {found_path}")
                                pytorch_config_path = found_path
                                break
                        
                        if found_path is None:
                            # Check if this might be a TensorFlow model instead
                            tensorflow_indicators = []
                            if (model_train_folder / "snapshot").exists():
                                tensorflow_indicators.append("Found 'snapshot' directory (TensorFlow indicator)")
                            if (model_train_folder / "train").exists():
                                # Check for TensorFlow checkpoint files
                                train_dir = model_train_folder / "train"
                                if any(train_dir.glob("*.ckpt*")) or any(train_dir.glob("*.index")):
                                    tensorflow_indicators.append("Found TensorFlow checkpoint files")
                            
                            # Provide helpful error message with search locations
                            error_msg = (
                                f"pytorch_config.yaml not found at expected location: {pytorch_config_path}\n"
                                "This file is required for PyTorch inference. It should be created during training.\n"
                            )
                            if tensorflow_indicators:
                                error_msg += (
                                    "⚠️  WARNING: This appears to be a TensorFlow model, not PyTorch!\n"
                                    "Indicators found:\n"
                                )
                                for indicator in tensorflow_indicators:
                                    error_msg += f"  - {indicator}\n"
                                error_msg += (
                                    "If this model was trained with TensorFlow, set engine='tensorflow' instead of 'pytorch'.\n"
                                )
                            if search_paths:
                                error_msg += f"\nSearched in {len(search_paths)} alternative locations:\n"
                                for sp in search_paths[:5]:  # Show first 5
                                    error_msg += f"  - {sp}\n"
                                if len(search_paths) > 5:
                                    error_msg += f"  ... and {len(search_paths) - 5} more\n"
                            else:
                                error_msg += (
                                    f"\nModel training directory not found: {model_train_folder}\n"
                                    "This suggests the model may not have been trained yet, or training failed.\n"
                                )
                            error_msg += (
                                "\nPossible solutions:\n"
                                "1. Ensure the model was trained with PyTorch (engine='pytorch')\n"
                                "2. Check that training completed successfully\n"
                                "3. Verify the model path and training parameters are correct\n"
                            )
                            raise FileNotFoundError(error_msg)
                    
                    # Verify it can be read by DLC (this is the critical check)
                    test_cfg = config_utils.read_config_as_dict(str(pytorch_config_path))
                    if test_cfg is None:
                        # Try to read the file directly to see what's wrong
                        try:
                            with open(pytorch_config_path, "r") as f:
                                file_content = f.read()
                            logger.error(f"pytorch_config.yaml content (first 500 chars):\n{file_content[:500]}")
                        except Exception as read_err:
                            logger.error(f"Could not even read file: {read_err}")
                        
                        # Try one more sanitization attempt
                        logger.warning("pytorch_config.yaml cannot be read by DLC, attempting emergency sanitization...")
                        try:
                            cls._sanitize_pytorch_config_yaml(project_path, dlc_config, dlc_model_)
                            test_cfg = config_utils.read_config_as_dict(str(pytorch_config_path))
                            if test_cfg is None:
                                raise ValueError(
                                    f"pytorch_config.yaml still cannot be read after sanitization: {pytorch_config_path}. "
                                    "File may be fundamentally corrupted. Check the file manually."
                                )
                        except Exception as sanitize_err:
                            logger.error(f"Emergency sanitization failed: {sanitize_err}")
                            raise ValueError(
                                f"pytorch_config.yaml exists but cannot be read by DLC: {pytorch_config_path}. "
                                "File may be corrupted or invalid. Check the file manually. "
                                "This usually happens when the file has ruamel.yaml tags that DLC can't parse. "
                                "Sanitization attempts have failed."
                            )
                    
                    if "method" not in test_cfg:
                        raise ValueError(
                            f"pytorch_config.yaml missing required 'method' key: {pytorch_config_path}. "
                            f"Available keys: {list(test_cfg.keys())}. "
                            "File may be corrupted."
                        )
                    logger.debug(f"Verified pytorch_config.yaml is valid: {pytorch_config_path}")
                except Exception as e:
                    logger.error(f"pytorch_config.yaml validation failed: {e}")
                    raise  # Always raise - don't continue with invalid config

        # ---- Take valid parameters for analyze_videos ----
        # Get function signature to check what parameters it accepts
        sig = inspect.signature(analyze_videos)
        param_names = list(sig.parameters.keys())
        
        kwargs = {
            k: v
            for k, v in analyze_video_params.items()
            if k in param_names
        }
        
        # For PyTorch, ensure batch_size is passed if available (overrides config file default)
        # Try both 'batch_size' and 'batchsize' parameter names
        if "batch_size" in param_names and "batch_size" in analyze_video_params:
            kwargs["batch_size"] = analyze_video_params["batch_size"]
        elif "batchsize" in param_names and "batchsize" in analyze_video_params:
            kwargs["batchsize"] = analyze_video_params["batchsize"]
        elif "batch_size" in param_names and "batchsize" in analyze_video_params:
            # If function accepts batch_size but we have batchsize, convert it
            kwargs["batch_size"] = analyze_video_params["batchsize"]
        elif "batchsize" in param_names and "batch_size" in analyze_video_params:
            # If function accepts batchsize but we have batch_size, convert it
            kwargs["batchsize"] = analyze_video_params["batch_size"]

        # ---- Trigger DLC prediction job ----
        try:
            analyze_videos(
                config=config_filepath,
                videos=video_filepaths,
                shuffle=dlc_model_["shuffle"],
                trainingsetindex=dlc_model_["trainingsetindex"],
                destfolder=output_dir,
                modelprefix=dlc_model_.get("model_prefix", ""),
                **kwargs,
            )
        except ValueError as e:
            error_msg = str(e)
            # Handle case where no predictions were found (empty predictions)
            if "Shape of passed values is" in error_msg and "indices imply" in error_msg:
                # This happens when DLC tries to create a DataFrame but has no predictions
                logger.warning(
                    f"No predictions found for video(s): {video_filepaths}. "
                    "This can happen if: "
                    "1) No animals were detected in the video, "
                    "2) The model confidence threshold is too high, "
                    "3) The video quality is poor, or "
                    "4) The model is not suitable for this video type."
                )
                logger.info(
                    "No pose estimation data will be available for this video. "
                    "The pipeline will skip this recording gracefully."
                )
                # Return early - no result files will be created
                return
            else:
                # Different ValueError, re-raise it
                raise
        except TypeError as e:
            if "'NoneType' object is not subscriptable" in str(e):
                # This is the specific error we're trying to fix
                logger.error(
                    "DLC's read_config_as_dict returned None for pytorch_config.yaml. "
                    "This usually means the file is corrupted or missing required keys."
                )
                if engine == "pytorch":
                    logger.error(
                        "For PyTorch models, ensure pytorch_config.yaml exists and contains "
                        "at minimum: 'method' key (e.g., 'bu' or 'td')."
                    )
                    # Try to find and list all pytorch_config.yaml files for debugging
                    try:
                        from deeplabcut.utils.auxiliaryfunctions import get_model_folder
                        model_folder = get_model_folder(
                            trainFraction=dlc_config.get("TrainingFraction", [0.95])[dlc_model_.get("trainingsetindex", 0)],
                            shuffle=dlc_model_.get("shuffle", 1),
                            cfg=dlc_config,
                            modelprefix=dlc_model_.get("model_prefix", ""),
                        )
                        model_train_folder = project_path / model_folder / "train"
                        pytorch_config_path = model_train_folder / "pytorch_config.yaml"
                        logger.error(f"Expected pytorch_config.yaml at: {pytorch_config_path}")
                        logger.error(f"File exists: {pytorch_config_path.exists()}")
                        if pytorch_config_path.exists():
                            try:
                                with open(pytorch_config_path, "r") as f:
                                    content = f.read()
                                logger.error(f"File size: {len(content)} bytes")
                                logger.error(f"First 200 chars: {content[:200]}")
                            except Exception as read_err:
                                logger.error(f"Could not read file: {read_err}")
                    except Exception as debug_err:
                        logger.error(f"Could not determine expected path: {debug_err}")
            raise

    def make(self, key):
        """.populate() method will launch pose estimation inference for each PoseEstimationTask"""
        # ID model and directories
        dlc_model_ = (Model & key).fetch1()
        task_mode, output_dir = (PoseEstimationTask & key).fetch1(
            "task_mode", "pose_estimation_output_dir"
        )
        if not output_dir:
            output_dir = PoseEstimationTask.infer_output_dir(
                key, relative=True, mkdir=True
            )
            # update pose_estimation_output_dir
            PoseEstimationTask.update1(
                {**key, "pose_estimation_output_dir": output_dir.as_posix()}
            )

        try:
            output_dir = find_full_path(get_dlc_root_data_dir(), output_dir)
        except FileNotFoundError as e:
            if task_mode == "trigger":
                processed_dir = Path(get_dlc_processed_data_dir())
                output_dir = processed_dir / output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise e

        # Trigger PoseEstimation only if in "trigger" mode
        # If results already exist, task_mode should be "load" to avoid re-running inference
        if task_mode == "trigger":
            # Log the key being used to debug video grouping
            logger.info(f"PoseEstimation.make() called with key: {key}")
            logger.info(f"Output directory: {output_dir}")
            
            # Get videos for THIS specific recording only
            video_relpaths = list((VideoRecording.File & key).fetch("file_path"))
            logger.info(f"Found {len(video_relpaths)} video file(s) for key {key}: {video_relpaths}")
            
            video_filepaths = [
                find_full_path(get_dlc_root_data_dir(), fp).as_posix()
                for fp in video_relpaths
            ]
            logger.info(f"Resolved video filepaths: {video_filepaths}")
            pose_estimation_params = (PoseEstimationTask & key).fetch1(
                "pose_estimation_params"
            ) or {}

            # Check if this is a pretrained model by looking in config_template
            config_template = dlc_model_.get("config_template", {})
            pretrained_model_name = config_template.get("_pretrained_model_name")
            is_pretrained = pretrained_model_name is not None

            # Handle pretrained models differently
            if is_pretrained:
                # Ensure the pretrained model exists in lookup - require explicit registration
                if not PretrainedModel.is_pretrained(pretrained_model_name):
                    raise ValueError(
                        f"Pretrained model '{pretrained_model_name}' must be registered "
                        "in PretrainedModel lookup table before use. "
                        "Please add it using PretrainedModel.insert1() or PretrainedModel.add()."
                    )
                
                # Build inference_params from pose_estimation_params
                # (default_params will be merged inside _do_pretrained_inference)
                pose_inference_params = (
                    pose_estimation_params.get("video_inference") or pose_estimation_params
                )

                @memoized_result(
                    uniqueness_dict={
                        **pose_inference_params,
                        "pretrained_model_name": pretrained_model_name,
                        "video_filepaths": video_relpaths,
                    },
                    output_directory=output_dir,
                )
                def _do_pretrained_inference():
                    result = PoseEstimation._do_pretrained_inference(
                        pretrained_model_name=pretrained_model_name,
                        video_filepaths=video_filepaths,
                        output_dir=output_dir,
                        inference_params=pose_inference_params,
                    )
                    # If result is None, it means no animals were detected
                    # Check if result files exist, and if not, skip this recording
                    if result is None:
                        # Check if empty result files were created
                        output_path = Path(output_dir)
                        result_files = list(output_path.glob("*.h5")) + list(output_path.glob("*.pickle"))
                        if not result_files:
                            logger.warning(
                                f"No animals detected and no result files created for {key}. "
                                "Skipping this recording - no pose data will be inserted."
                            )
                            # Return early to skip inserting pose estimation data
                            return None
                
                inference_result = _do_pretrained_inference()
                # If inference returned None, check if result files were created
                # (empty result files may have been created to indicate no detections)
                if inference_result is None:
                    output_path = Path(output_dir)
                    result_files = list(output_path.glob("*.h5")) + list(output_path.glob("*.pickle"))
                    if not result_files:
                        logger.info(
                            f"No animals detected in video(s) for key {key} and no result files created. "
                            "Skipping pose estimation data insertion."
                        )
                        return  # Skip the rest of make() - no data to insert
                    else:
                        logger.info(
                            f"No animals detected but empty result files exist. "
                            "Will attempt to read them (may contain NaN values)."
                        )
            else:
                # Original trained model path
                # Triggering dlc for pose estimation required:
                # - project_path: full path to the directory containing the trained model
                # - video_filepaths: full paths to the video files for inference
                # - analyze_video_params: optional parameters to analyze video
                project_path = find_full_path(
                    get_dlc_root_data_dir(), dlc_model_["project_path"]
                )

                # expect a nested dictionary with "analyze_videos" params
                # if not, assume "pose_estimation_params" as a flat dictionary that include relevant "analyze_videos" params
                analyze_video_params = (
                    pose_estimation_params.get("analyze_videos") or pose_estimation_params
                )

                @memoized_result(
                    uniqueness_dict={
                        **analyze_video_params,
                        "project_path": dlc_model_["project_path"],
                        "shuffle": dlc_model_["shuffle"],
                        "trainingsetindex": dlc_model_["trainingsetindex"],
                        "video_filepaths": video_relpaths,
                    },
                    output_directory=output_dir,
                )
                def _do_trained_inference():
                    dlc_config = dlc_model_["config_template"].copy()
                    PoseEstimation.do_trained(
                        project_path=project_path,
                        video_filepaths=video_filepaths,
                        output_dir=output_dir,
                        dlc_config=dlc_config,
                        dlc_model_=dlc_model_,
                        analyze_video_params=analyze_video_params,
                    )

                _do_trained_inference()

        # Check if result files exist before trying to read them (use rglob to match dlc_reader behavior)
        output_path = Path(output_dir)
        result_files = list(output_path.rglob("*.h5")) + list(output_path.rglob("*.pickle"))
        if not result_files:
            logger.warning(
                f"No result files found in {output_dir} for key {key}. "
                "This may indicate that no animals were detected or inference failed. "
                "Skipping pose estimation data insertion."
            )
            return  # Skip the rest of make() - no data to insert
        
        # Try to initialize DLC result reader, handle FileNotFoundError gracefully
        try:
            dlc_result = dlc_reader.PoseEstimation(output_dir)
        except FileNotFoundError as e:
            error_msg = str(e)
            if "No DLC output file (.h5) found" in error_msg or ".h5" in error_msg or "No meta file" in error_msg:
                logger.warning(
                    f"No DLC result files found in {output_dir} for key {key}. "
                    "This likely means no animals were detected during inference. "
                    "Skipping pose estimation data insertion."
                )
                return  # Skip the rest of make() - no data to insert
            else:
                # Different FileNotFoundError, re-raise it
                raise
        creation_time = datetime.fromtimestamp(dlc_result.creation_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Handle different data structures (DLC 2.x vs 3.x, single vs multi-animal)
        body_parts = []
        
        # Check if this is multi-animal format (keys like 'animal0', 'animal1', etc.)
        # Single-animal format: keys are body part names (e.g., 'nose', 'tail')
        # Multi-animal format: keys contain 'animal' or 'individual' (e.g., 'animal0_nose', 'animal0_superanimal_...')
        data_keys = list(dlc_result.data.keys()) if dlc_result.data else []
        is_multi_animal = any(k.startswith('animal') or k.startswith('individual') for k in data_keys)
        
        if is_multi_animal:
            # Multi-animal format: each key is an animal, and each animal has body parts
            logger.info(
                f"Multi-animal format detected (keys: {data_keys[:5]}...). "
                "Extracting body parts from all animals."
            )
            
            # Helper function to extract base individual name (e.g., "animal0" from "animal0_superanimal_...")
            def extract_individual_id(full_key: str) -> str:
                """Extract base individual ID from full key name.
                
                Examples:
                    "animal0_superanimal_..." -> "animal0"
                    "animal1_model_name" -> "animal1"
                    "individual0" -> "individual0"
                """
                # Try to match pattern: animal<number> or individual<number>
                import re
                match = re.match(r'^(animal\d+|individual\d+)', full_key)
                if match:
                    return match.group(1)
                # Fallback: return first part before underscore
                return full_key.split('_')[0] if '_' in full_key else full_key
            
            # The structure from reformat_rawdata() should be: "individual_bodypart" -> {x, y, likelihood}
            # But if body parts extraction failed, keys might be just individual names
            # Check if keys already contain body parts (format: "individual_bodypart")
            # or if they're just individual names that need to be processed differently
            
            # First, check if keys already have x/y data directly (format: "individual_bodypart")
            keys_with_xy = [k for k in data_keys if isinstance(dlc_result.data.get(k), dict) 
                           and "x" in dlc_result.data[k] and "y" in dlc_result.data[k]]
            
            if keys_with_xy:
                # Keys are already in "individual_bodypart" format with x/y data
                logger.info(f"Found {len(keys_with_xy)} keys with direct x/y data. Processing as 'individual_bodypart' format.")
                for full_key in keys_with_xy:
                    key_data = dlc_result.data[full_key]
                    if isinstance(key_data, dict) and "x" in key_data and "y" in key_data:
                        # Extract individual and body part from key
                        # Format: "animal0_superanimal_..._bodypart" or "animal0_bodypart"
                        individual_id = extract_individual_id(full_key)
                        # Try to extract body part name (usually the last meaningful part)
                        # Remove the individual prefix and scorer/model suffix
                        parts = full_key.split('_')
                        # Find body part name (usually a short word at the end, not a model/scorer term)
                        # Common body part names and model terms to exclude
                        model_terms = {'superanimal', 'hrnet', 'fasterrcnn', 'resnet', 'fpn', 'v2', 'w32', 'w48', 'w64', 
                                      'resnet50', 'resnet101', 'mobilenet', 'efficientnet', 'densenet', 'inception'}
                        # Common body part names to prioritize (if found, use them)
                        common_body_parts = {'nose', 'head', 'eye', 'ear', 'neck', 'shoulder', 'elbow', 'wrist', 'hand', 
                                           'hip', 'knee', 'ankle', 'foot', 'toe', 'tail', 'back', 'belly', 'chest'}
                        body_part_name = None
                        
                        # First, check if any part matches common body part names
                        for part in reversed(parts):
                            if part.lower() in common_body_parts:
                                body_part_name = part.lower()
                                break
                        
                        # If no common body part found, look for any non-model term
                        if not body_part_name:
                            for part in reversed(parts):
                                if (part != individual_id and len(part) > 2 and 
                                    part.lower() not in model_terms and 
                                    not part.isdigit() and
                                    not part.lower().startswith('animal') and
                                    not part.lower().startswith('individual')):
                                    body_part_name = part.lower()
                                    break
                        
                        if body_part_name:
                            encoded_body_part = f"{individual_id}_{body_part_name}"
                            body_parts.append({
                **key,
                                "body_part": encoded_body_part,
                                "frame_index": np.arange(dlc_result.nframes),
                                "x_pos": key_data["x"],
                                "y_pos": key_data["y"],
                                "z_pos": key_data.get("z"),
                                "likelihood": key_data.get("likelihood", np.ones(dlc_result.nframes)),
                                "_individual_id": individual_id,
                                "_clean_body_part": body_part_name,
                            })
                        else:
                            logger.warning(f"Could not extract body part name from key '{full_key}'. Using key as body part name.")
                            # Fallback: use a sanitized version of the key
                            body_part_name = full_key.replace(f"{individual_id}_", "").replace("_", " ").title().replace(" ", "")
                            encoded_body_part = f"{individual_id}_{body_part_name}"
                            body_parts.append({
                                **key,
                                "body_part": encoded_body_part,
                                "frame_index": np.arange(dlc_result.nframes),
                                "x_pos": key_data["x"],
                                "y_pos": key_data["y"],
                                "z_pos": key_data.get("z"),
                                "likelihood": key_data.get("likelihood", np.ones(dlc_result.nframes)),
                                "_individual_id": individual_id,
                                "_clean_body_part": body_part_name,
                            })
            else:
                # Keys are just individual names, need to look for nested structure
                logger.info(f"Keys appear to be individual names only. Checking for nested body part structure...")
                for animal_key_full in data_keys:
                    animal_data = dlc_result.data[animal_key_full]
                    individual_id = extract_individual_id(animal_key_full)
                    
                    if isinstance(animal_data, dict):
                        # Check if this dict contains body parts directly
                        animal_dict_keys = list(animal_data.keys())
                        logger.debug(f"Individual '{individual_id}' (key: '{animal_key_full}') has {len(animal_dict_keys)} sub-key(s): {animal_dict_keys[:10]}")
                        
                        for sub_key, sub_data in animal_data.items():
                            if isinstance(sub_data, dict) and "x" in sub_data and "y" in sub_data:
                                # sub_key is the body part name
                                body_part_name = sub_key
                                encoded_body_part = f"{individual_id}_{body_part_name}"
                                body_parts.append({
                                    **key,
                                    "body_part": encoded_body_part,
                                    "frame_index": np.arange(dlc_result.nframes),
                                    "x_pos": sub_data["x"],
                                    "y_pos": sub_data["y"],
                                    "z_pos": sub_data.get("z"),
                                    "likelihood": sub_data.get("likelihood", np.ones(dlc_result.nframes)),
                                    "_individual_id": individual_id,
                                    "_clean_body_part": body_part_name,
                                })
                            elif isinstance(sub_data, dict):
                                # Nested further - sub_data might contain body parts
                                logger.debug(f"Sub-key '{sub_key}' is a dict but doesn't have x/y. Keys: {list(sub_data.keys())[:5]}")
                    else:
                        logger.debug(f"Key '{animal_key_full}' data is not a dict (type: {type(animal_data)})")
        else:
            # Single-animal format: keys are body parts
            for k, v in dlc_result.data.items():
                # Check if v is a dict with expected keys
                if isinstance(v, dict):
                    # DLC 2.x format: dict with 'x', 'y', 'likelihood' keys
                    if "x" in v and "y" in v:
                        body_parts.append({
                            **key,
                            "body_part": k,  # Single-animal format
                            # No individual_id - will be NULL (single-animal)
                "frame_index": np.arange(dlc_result.nframes),
                "x_pos": v["x"],
                "y_pos": v["y"],
                "z_pos": v.get("z"),
                            "likelihood": v.get("likelihood", np.ones(dlc_result.nframes)),  # Default to 1.0 if missing
                        })
                    else:
                        logger.warning(
                            f"Body part '{k}' data structure unexpected. Keys: {list(v.keys())}. "
                            "Skipping this body part."
                        )
                else:
                    logger.warning(
                        f"Body part '{k}' data is not a dict (type: {type(v)}). Skipping."
                    )
        
        if len(body_parts) == 0:
            # Instead of raising an error, log a warning and skip this recording
            logger.error(
                f"No valid body part data found in results for key {key}. "
                f"Data structure: {data_keys}. "
                f"First item structure: {type(list(dlc_result.data.values())[0]) if dlc_result.data else 'N/A'}. "
                "Skipping this recording and continuing with next."
            )
            # Return early without inserting - this will skip this key
            return

        # Extract unique body part names, clean names, and individuals from the results
        unique_body_parts_encoded = set()  # Encoded names like "animal0_nose"
        unique_body_parts_clean = set()  # Clean names like "nose"
        unique_individuals = set()
        individual_mappings = []  # Store mappings for IndividualMapping table
        
        for bp in body_parts:
            encoded_name = bp["body_part"]
            unique_body_parts_encoded.add(encoded_name)
            
            # Extract individual and clean body part name
            individual_id = bp.pop("_individual_id", None)
            clean_body_part = bp.pop("_clean_body_part", None)
            
            if individual_id:
                unique_individuals.add(individual_id)
                if clean_body_part:
                    unique_body_parts_clean.add(clean_body_part)
                    # Store mapping for IndividualMapping table
                    individual_mappings.append({
                        **key,
                        "body_part": encoded_name,  # The encoded name in BodyPartPosition
                        "individual_id": individual_id
                    })
            else:
                # Single-animal: encoded name is the clean name
                unique_body_parts_clean.add(encoded_name)
        
        # Register body part names in global BodyPart table
        # For multi-animal: register both clean names (e.g., "nose") and encoded names (e.g., "animal0_nose")
        # For single-animal: register clean names only (encoded = clean)
        model_name = key["model_name"]
        
        # Register clean body part names
        for clean_body_part in unique_body_parts_clean:
            if not (BodyPart & {"body_part": clean_body_part}):
                BodyPart.insert1(
                    {"body_part": clean_body_part, "body_part_description": ""},
                    skip_duplicates=True
                )
                logger.info(f"Registered new body part: {clean_body_part}")
        
        # Register encoded body part names (for multi-animal support)
        # These are different from clean names and need to be registered separately
        for encoded_body_part in unique_body_parts_encoded:
            # Only register if it's different from clean names (multi-animal case)
            if encoded_body_part not in unique_body_parts_clean:
                if not (BodyPart & {"body_part": encoded_body_part}):
                    BodyPart.insert1(
                        {"body_part": encoded_body_part, "body_part_description": ""},
                        skip_duplicates=True
                    )
                    logger.debug(f"Registered encoded body part: {encoded_body_part}")
        
        # Link body parts to model in Model.BodyPart
        # Use encoded names for multi-animal, clean names for single-animal
        for encoded_body_part in unique_body_parts_encoded:
            if not (Model.BodyPart & {"model_name": model_name, "body_part": encoded_body_part}):
                Model.BodyPart.insert1(
                    {"model_name": model_name, "body_part": encoded_body_part},
                    skip_duplicates=True
                )
                logger.debug(f"Linked body part {encoded_body_part} to model {model_name}")

        # Insert master row FIRST (required before inserting into Part tables)
        self.insert1({**key, "pose_estimation_time": creation_time})
        
        # Now insert into Part tables (they require the master row to exist)
        self.BodyPartPosition.insert(body_parts)
        
        # Register individuals (for multi-animal data) - must be after master row is inserted
        if unique_individuals:
            individuals_to_insert = [
                {**key, "individual_id": ind_id}
                for ind_id in unique_individuals
            ]
            self.Individual.insert(individuals_to_insert, skip_duplicates=True)
            logger.info(f"Registered {len(unique_individuals)} individual(s): {sorted(unique_individuals)}")
        
        # Insert individual mappings if this is multi-animal data
        if individual_mappings:
            for mapping in individual_mappings:
                # IndividualMapping needs: master key (PoseEstimation) + body_part + Individual key
                # PoseEstimation key: subject, session_datetime, recording_id, model_name
                # body_part: from BodyPartPosition (via Model.BodyPart)
                # Individual key: subject, session_datetime, recording_id, model_name, individual_id
                mapping_key = {
                    **key,  # subject, session_datetime, recording_id, model_name (from master)
                    "body_part": mapping["body_part"],  # from BodyPartPosition
                    "individual_id": mapping["individual_id"]  # from Individual
                }
                
                # Verify both BodyPartPosition and Individual exist
                bp_key = {**key, "body_part": mapping["body_part"]}
                ind_key = {**key, "individual_id": mapping["individual_id"]}
                
                bp_exists = bool(self.BodyPartPosition & bp_key)
                ind_exists = bool(self.Individual & ind_key)
                
                if bp_exists and ind_exists:
                    try:
                        self.IndividualMapping.insert1(
                            mapping_key,
                            skip_duplicates=True
                        )
                        logger.debug(f"Created mapping: {mapping_key}")
                    except Exception as e:
                        logger.warning(f"Could not insert mapping for {mapping_key}: {e}")
                else:
                    logger.debug(
                        f"Could not create mapping: bp_key={bp_key} (exists: {bp_exists}), "
                        f"ind_key={ind_key} (exists: {ind_exists})"
                    )

    @classmethod
    def get_trajectory(cls, key: dict, body_parts: list = "all") -> pd.DataFrame:
        """Returns a pandas dataframe of coordinates of the specified body_part(s)

        Args:
            key (dict): A DataJoint query specifying one PoseEstimation entry.
            body_parts (list, optional): Body parts as a list. If "all", all joints

        Returns:
            df: multi index pandas dataframe with DLC scorer names, body_parts
                and x/y coordinates of each joint name for a camera_id, similar to
                 output of DLC dataframe. If 2D, z is set of zeros
        """
        model_name = key["model_name"]

        if body_parts == "all":
            body_parts = (cls.BodyPartPosition & key).fetch("body_part")
        elif not isinstance(body_parts, list):
            body_parts = list(body_parts)

        df = None
        for body_part in body_parts:
            x_pos, y_pos, z_pos, likelihood = (
                cls.BodyPartPosition & {"body_part": body_part}
            ).fetch1("x_pos", "y_pos", "z_pos", "likelihood")
            if not z_pos:
                z_pos = np.zeros_like(x_pos)

            a = np.vstack((x_pos, y_pos, z_pos, likelihood))
            a = a.T
            pdindex = pd.MultiIndex.from_product(
                [[model_name], [body_part], ["x", "y", "z", "likelihood"]],
                names=["scorer", "bodyparts", "coords"],
            )
            frame = pd.DataFrame(a, columns=pdindex, index=range(0, a.shape[0]))
            df = pd.concat([df, frame], axis=1)
        return df


@schema
class LabeledVideo(dj.Computed):
    definition = """
    -> PoseEstimation
    """

    class File(dj.Part):
        definition = """
        -> master
        -> VideoRecording.File
        ---
        labeled_video_path: varchar(255)  # relative path to labeled video
        """

    @property
    def key_source(self):
        return PoseEstimation & RecordingInfo

    def make(self, key):
        import deeplabcut

        pose_estimation_params = (PoseEstimationTask & key).fetch1(
            "pose_estimation_params"
        ) or {}

        # expect a nested dictionary with "create_labeled_video" and "extract_outlier_frames" params
        # if not, assume "pose_estimation_params" as a flat dictionary
        create_labeled_video_params = (
            pose_estimation_params.get("create_labeled_video") or pose_estimation_params
        )

        outputframerate = create_labeled_video_params.pop(
            "outputframerate", 5
        )  # final labeled video FPS defaults to 5 Hz

        dlc_model_ = (Model & key).fetch1()
        fps, nframes = (RecordingInfo & key).fetch1("fps", "nframes")
        output_dir = (PoseEstimationTask & key).fetch1("pose_estimation_output_dir")
        output_dir = find_full_path(get_dlc_root_data_dir(), output_dir)

        project_path = find_full_path(
            get_dlc_root_data_dir(), dlc_model_["project_path"]
        )

        try:
            dlc_config = next(output_dir.glob("dj_dlc_config*.yaml"))
            dlc_config = project_path / dlc_config.name
            assert dlc_config.exists()
        except (StopIteration, AssertionError):
            dlc_config = next(project_path.glob("dj_dlc_config*.yaml"))
            logger.warning(
                f"No dj_dlc_config*.yaml file found in {output_dir} - this is unexpected.\nUsing {dlc_config}"
            )

        entries = []
        for vkey in (VideoRecording.File & key).fetch("KEY"):
            video_file = (VideoRecording.File & vkey).fetch1("file_path")
            video_file = find_full_path(get_dlc_root_data_dir(), video_file)

            # -- create labeled video --
            create_labeled_video_kwargs = {
                k: v
                for k, v in create_labeled_video_params.items()
                if k in inspect.signature(deeplabcut.create_labeled_video).parameters
            }
            create_labeled_video_kwargs.update(
                dict(
                    config=dlc_config.as_posix(),
                    videos=[video_file.as_posix()],
                    shuffle=dlc_model_["shuffle"],
                    trainingsetindex=dlc_model_["trainingsetindex"],
                    modelprefix=dlc_model_["model_prefix"],
                    destfolder=output_dir.as_posix(),
                    Frames2plot=np.arange(0, nframes, int(fps / outputframerate)),
                    outputframerate=outputframerate,
                )
            )
            deeplabcut.create_labeled_video(**create_labeled_video_kwargs)

            labeled_video_path = next(
                output_dir.glob(f"{video_file.stem}*_labeled.mp4")
            )
            entries.append(
                {
                    **key,
                    **vkey,
                    "labeled_video_path": labeled_video_path.relative_to(
                        get_dlc_processed_data_dir()
                    ).as_posix(),
                }
            )

        self.insert1(key)
        self.File.insert(entries)


def str_to_bool(value) -> bool:
    """Return whether the provided string represents true. Otherwise false.

    Args:
        value (any): Any input

    Returns:
        bool (bool): True if value in ("y", "yes", "t", "true", "on", "1")
    """
    # Due to distutils equivalent depreciation in 3.10
    # Adopted from github.com/PostHog/posthog/blob/master/posthog/utils.py
    if not value:
        return False
    return str(value).lower() in ("y", "yes", "t", "true", "on", "1")
