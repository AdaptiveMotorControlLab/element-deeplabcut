"""
Code adapted from the Mathis Lab
MIT License Copyright (c) 2022 Mackenzie Mathis
DataJoint Schema for DeepLabCut 2.x, Supports 2D and 3D DLC via triangulation.
"""

import datajoint as dj
import inspect
import importlib
import re
import logging
from pathlib import Path
import yaml

from element_interface.utils import find_full_path, dict_to_uuid
from .readers import dlc_reader

logger = logging.getLogger(__name__)

schema = dj.schema()
_linking_module = None


def activate(
    train_schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate this schema.

    Args:
        train_schema_name (str): schema name on the database server
        create_schema (bool): when True (default), create schema in the database if it
                            does not yet exist.
        create_tables (bool): when True (default), create schema tables in the database
                             if they do not yet exist.
        linking_module (str): a module (or name) containing the required dependencies.

    Dependencies:
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
        train_schema_name,
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
    if isinstance(root_directories, (str, Path)):
        root_directories = [root_directories]

    if (
        hasattr(_linking_module, "get_dlc_processed_data_dir")
        and get_dlc_processed_data_dir() not in root_directories
    ):
        root_directories.append(_linking_module.get_dlc_processed_data_dir())

    return root_directories


def get_dlc_processed_data_dir() -> str:
    """Pulls relevant func from parent namespace. Defaults to DLC's project /videos/.

    Method in parent namespace should provide a string to a directory where DLC output
    files will be stored. If unspecified, output files will be stored in the
    session directory 'videos' folder, per DeepLabCut default.
    """
    if hasattr(_linking_module, "get_dlc_processed_data_dir"):
        return _linking_module.get_dlc_processed_data_dir()
    else:
        return get_dlc_root_data_dir()[0]


# ----------------------------- Table declarations ----------------------


@schema
class VideoSet(dj.Manual):
    """Collection of videos included in a given training set.

    Attributes:
        video_set_id (int): Unique ID for collection of videos."""

    definition = """ # Set of vids in training set
    video_set_id: int
    """

    class File(dj.Part):
        """File IDs and paths in a given VideoSet

        Attributes:
            VideoSet (foreign key): VideoSet key.
            file_path ( varchar(255) ): Path to file on disk relative to root."""

        definition = """ # Paths of training files (e.g., labeled pngs, CSV or video)
        -> master
        file_id: int
        ---
        file_path: varchar(255)
        """


@schema
class TrainingParamSet(dj.Lookup):
    """Parameters used to train a model

    Attributes:
        paramset_idx (smallint): Index uniqely identifying paramset.
        paramset_desc ( varchar(128) ): Description of paramset.
        param_set_hash (uuid): Hash identifying this paramset.
        params (longblob): Dictionary of all applicable parameters.
        Note: param_set_hash must be unique."""

    definition = """
    # Parameters to specify a DLC model training instance
    # For DLC ≤ v2.0, include scorer_legacy = True in params
    paramset_idx                  : smallint
    ---
    paramset_desc: varchar(128)
    param_set_hash                : uuid      # hash identifying this parameterset
    unique index (param_set_hash)
    params                        : longblob  # dictionary of all applicable parameters
    """

    required_parameters = ("shuffle", "trainingsetindex")
    skipped_parameters = ("project_path", "video_sets")

    @classmethod
    def insert_new_params(
        cls, paramset_desc: str, params: dict, paramset_idx: int = None
    ):
        """
        Insert a new set of training parameters into dlc.TrainingParamSet.

        Args:
            paramset_desc (str): Description of parameter set to be inserted
            params (dict): Dictionary including all settings to specify model training.
                        Must include shuffle & trainingsetindex b/c not in config.yaml.
                        project_path and video_sets will be overwritten by config.yaml.
                        Note that trainingsetindex is 0-indexed
            paramset_idx (int): optional, integer to represent parameters.
        """

        for required_param in cls.required_parameters:
            assert required_param in params, (
                "Missing required parameter: " + required_param
            )
        for skipped_param in cls.skipped_parameters:
            if skipped_param in params:
                params.pop(skipped_param)

        if paramset_idx is None:
            paramset_idx = (
                dj.U().aggr(cls, n="max(paramset_idx)").fetch1("n") or 0
            ) + 1

        param_dict = {
            "paramset_idx": paramset_idx,
            "paramset_desc": paramset_desc,
            "params": params,
            "param_set_hash": dict_to_uuid(params),
        }
        param_query = cls & {"param_set_hash": param_dict["param_set_hash"]}
        # If the specified param-set already exists
        if param_query:
            existing_paramset_idx = param_query.fetch1("paramset_idx")
            if existing_paramset_idx == int(paramset_idx):  # If existing_idx same:
                return  # job done
        else:
            cls.insert1(param_dict)  # if duplicate, will raise duplicate error


@schema
class TrainingTask(dj.Manual):
    """Staging table for pairing videosets and training parameter sets

    Attributes:
        VideoSet (foreign key): VideoSet Key.
        TrainingParamSet (foreign key): TrainingParamSet key.
        training_id (int): Unique ID for training task.
        model_prefix ( varchar(32) ): Optional. Prefix for model files.
        project_path ( varchar(255) ): Optional. DLC's project_path in config relative
                                       to get_dlc_root_data_dir
    """

    definition = """      # Specification for a DLC model training instance
    -> VideoSet           # labeled video(s) for training
    -> TrainingParamSet
    training_id     : int
    ---
    model_prefix='' : varchar(32)
    project_path='' : varchar(255) # DLC's project_path in config relative to root
    """


@schema
class ModelTraining(dj.Computed):
    """Automated Model training information.

    Attributes:
        TrainingTask (foreign key): TrainingTask key.
        latest_snapshot (int unsigned): Latest exact snapshot index (i.e., never -1).
        config_template (longblob): Stored full config file."""

    definition = """
    -> TrainingTask
    ---
    latest_snapshot: int unsigned # latest exact snapshot index (i.e., never -1)
    config_template: longblob     # stored full config file
    """

    # To continue from previous training snapshot, devs suggest editing pose_cfg.yml
    # https://github.com/DeepLabCut/DeepLabCut/issues/70

    def make(self, key):
        import deeplabcut

        try:
            from deeplabcut.utils.auxiliaryfunctions import (
                get_model_folder,
                edit_config,
            )  # isort:skip
        except ImportError:
            from deeplabcut.utils.auxiliaryfunctions import (
                GetModelFolder as get_model_folder,
            )  # isort:skip

        """Launch training for each train.TrainingTask training_id via `.populate()`."""
        project_path, model_prefix = (TrainingTask & key).fetch1(
            "project_path", "model_prefix"
        )

        project_path = find_full_path(get_dlc_root_data_dir(), project_path)
        
        # Ensure project_path is a directory, not a file
        project_path = Path(project_path)
        if project_path.is_file():
            project_path = project_path.parent
        elif not project_path.is_dir():
            raise ValueError(f"project_path is neither a file nor a directory: {project_path}")

        # ---- Build and save DLC configuration (yaml) file ----
        _, dlc_config = dlc_reader.read_yaml(project_path)  # load existing
        training_params = (TrainingParamSet & key).fetch1("params")
        
        # Ensure shuffle and trainingsetindex from TrainingParamSet override config values
        # This is critical - the config file might have different values
        shuffle = training_params.get("shuffle", dlc_config.get("shuffle", 1))
        trainingsetindex = training_params.get("trainingsetindex", dlc_config.get("trainingsetindex", 0))
        
        # Explicitly set these values in config (they must match the training dataset)
        dlc_config["shuffle"] = int(shuffle)
        dlc_config["trainingsetindex"] = int(trainingsetindex)
        
        logger.info(f"Training parameters: shuffle={dlc_config['shuffle']}, trainingsetindex={dlc_config['trainingsetindex']}")
        
        # Update other params (but shuffle and trainingsetindex are already set above)
        other_params = {k: v for k, v in training_params.items() if k not in ["shuffle", "trainingsetindex"]}
        dlc_config.update(other_params)
        
        # Get engine from config
        # Note: DLC 3.x may have issues with engine setting for training
        # Try to detect or default appropriately
        engine = dlc_config.get("engine")
        
        # If no engine specified, don't set one - let DLC use default
        # DLC 3.x training may work better without explicit engine setting
        if engine is None:
            # Don't set engine - will use DLC's default behavior
            engine = None
        elif engine not in ["tensorflow", "pytorch"]:
            # Invalid engine, reset to None
            logger.warning(f"Invalid engine '{engine}', using DLC default")
            engine = None
            if "engine" in dlc_config:
                del dlc_config["engine"]
        # Ensure trainingsetindex is used correctly for train_fraction
        # Use the value we just set, not from config (which might be wrong)
        train_fraction = dlc_config["TrainingFraction"][int(trainingsetindex)]
        
        dlc_config.update(
            {
                "project_path": project_path.as_posix(),
                "modelprefix": model_prefix,
                "train_fraction": train_fraction,
                "training_filelist_datajoint": [  # don't overwrite origin video_sets
                    find_full_path(get_dlc_root_data_dir(), fp).as_posix()
                    for fp in (VideoSet.File & key).fetch("file_path")
                ],
            }
        )
        # Write dlc config file to base project folder with correct values
        dlc_cfg_filepath = dlc_reader.save_yaml(project_path, dlc_config)
        
        # Verify the saved config has correct values (re-read to confirm)
        _, saved_config = dlc_reader.read_yaml(project_path)
        saved_shuffle = saved_config.get("shuffle", shuffle)
        saved_trainingsetindex = saved_config.get("trainingsetindex", trainingsetindex)
        
        logger.info(f"Saved config verification: shuffle={saved_shuffle}, trainingsetindex={saved_trainingsetindex}")
        logger.info(f"Expected values: shuffle={shuffle}, trainingsetindex={trainingsetindex}")
        
        # If there's a mismatch, fix it
        if saved_trainingsetindex != trainingsetindex or saved_shuffle != shuffle:
            logger.warning(f"Config file mismatch detected! Fixing...")
            saved_config["shuffle"] = shuffle
            saved_config["trainingsetindex"] = trainingsetindex
            dlc_reader.save_yaml(project_path, saved_config, filename="config")
            # Use the corrected values
            dlc_config["shuffle"] = shuffle
            dlc_config["trainingsetindex"] = trainingsetindex
        else:
            # Use the verified values from saved config
            dlc_config["shuffle"] = saved_shuffle
            dlc_config["trainingsetindex"] = saved_trainingsetindex

        # ---- Update the project path in the DLC pose configuration (yaml) files ----
        model_folder = get_model_folder(
            trainFraction=dlc_config["train_fraction"],
            shuffle=dlc_config["shuffle"],
            cfg=dlc_config,
            modelprefix=dlc_config["modelprefix"],
        )
        model_train_folder = project_path / model_folder / "train"

        # Ensure model_train_folder exists (it's created when training starts)
        model_train_folder.mkdir(parents=True, exist_ok=True)
        
        # Check if pose_cfg.yaml exists, if not, it will be created by DLC during training
        pose_cfg_path = model_train_folder / "pose_cfg.yaml"
        if not pose_cfg_path.exists():
            # pose_cfg.yaml will be created by DLC's train_network function
            # Skip init_weights update for now - it will be handled by DLC
            logger.warning(
                f"pose_cfg.yaml not found at {pose_cfg_path}. "
                "It will be created by DLC during training. Skipping init_weights update."
            )
            init_weights_path = None
            pose_cfg = {}
        else:
            # update path of the init_weight
            with open(pose_cfg_path, "r") as f:
                pose_cfg = yaml.safe_load(f)
            init_weights_path = Path(pose_cfg["init_weights"])

            if (
                "pose_estimation_tensorflow/models/pretrained"
                in init_weights_path.as_posix()
            ):
                # this is the res_net models, construct new path here
                init_weights_path = (
                    Path(deeplabcut.__path__[0])
                    / "pose_estimation_tensorflow/models/pretrained"
                    / init_weights_path.name
                )
            else:
                # this is existing snapshot weights, update path here
                init_weights_path = model_train_folder / init_weights_path.name

            edit_config(
                model_train_folder / "pose_cfg.yaml",
                {
                    "project_path": project_path.as_posix(),
                    "init_weights": init_weights_path.as_posix(),
                    "dataset": Path(pose_cfg["dataset"]).as_posix(),
                    "metadataset": Path(pose_cfg["metadataset"]).as_posix(),
                },
            )

        # ---- Trigger DLC model training job ----
        # DLC 3.x: The compat layer checks engine from config file
        # Ensure engine is explicitly set in the config before training
        # The compat layer reads engine from the config file, not from the function call
        
        # DLC 3.x compat layer reads engine from metadata file, not config
        # Ensure engine is set correctly in config (needed for metadata updates)
        # Don't remove it - DLC needs it in the config to determine the correct engine
        if "engine" not in dlc_config:
            dlc_config["engine"] = engine or "pytorch"
            logger.info(f"Added engine='{dlc_config['engine']}' to config")
        
        # Ensure config has correct engine before saving
        if dlc_config.get("engine") != engine:
            dlc_config["engine"] = engine or "pytorch"
            logger.info(f"Updated engine in config to '{dlc_config['engine']}'")
        
        # Save config with correct engine
        dlc_reader.save_yaml(project_path, dlc_config, filename="config")
        logger.info(f"Saved config with engine='{dlc_config.get('engine')}'")
        
        # CRITICAL: DLC 3.x doesn't implement train_network for TensorFlow
        # We MUST use the PyTorch-specific function directly to bypass the compat layer
        # The compat layer will try to use TensorFlow if it detects engine=tensorflow in metadata/config
        
        # Force engine to pytorch if not already set
        if engine != "pytorch":
            logger.warning(f"Engine was '{engine}', forcing to 'pytorch' (DLC 3.x doesn't support TensorFlow training)")
            engine = "pytorch"
            dlc_config["engine"] = "pytorch"
            dlc_reader.save_yaml(project_path, dlc_config, filename="config")
            logger.info("Updated config to use engine=pytorch")
        
        # Use PyTorch-specific training function directly (bypasses compat layer)
        try:
            from deeplabcut.pose_estimation_pytorch import train_network as train_func_pytorch
            train_func = train_func_pytorch
            logger.info("Using PyTorch-specific train_network function (bypassing compat layer)")
        except (ImportError, AttributeError) as e:
            logger.error(f"PyTorch training function not available: {e}")
            logger.error("Falling back to generic train_network (may fail if engine is not pytorch)")
            train_func = deeplabcut.train_network
        
        train_network_input_args = list(inspect.signature(train_func).parameters)
        
        # Build kwargs from config, but explicitly override shuffle and trainingsetindex
        # to ensure they match what we set (DLC reads from config file, so we need to be explicit)
        train_network_kwargs = {
            k: int(v) if k in ("shuffle", "trainingsetindex", "maxiters") else v
            for k, v in dlc_config.items()
            if k in train_network_input_args
        }
        
        # CRITICAL: Explicitly set shuffle and trainingsetindex to match TrainingParamSet
        # These must match the training dataset that was created
        train_network_kwargs["shuffle"] = int(shuffle)
        train_network_kwargs["trainingsetindex"] = int(trainingsetindex)
        
        # Ensure other numeric params are integers
        for k in ["maxiters", "displayiters", "saveiters"]:
            if k in train_network_kwargs:
                train_network_kwargs[k] = int(train_network_kwargs[k])
        
        logger.info(f"Training with kwargs: shuffle={train_network_kwargs.get('shuffle')}, trainingsetindex={train_network_kwargs.get('trainingsetindex')}")
        
        # Final verification: re-read config one more time right before training
        # DLC reads from the file, so we need to ensure it's correct
        _, final_config = dlc_reader.read_yaml(project_path)
        final_trainingsetindex = final_config.get("trainingsetindex")
        final_shuffle = final_config.get("shuffle")
        
        # CRITICAL: Ensure engine is in final_config (needed for metadata update)
        if "engine" not in final_config:
            final_config["engine"] = engine or "pytorch"
        elif final_config.get("engine") != (engine or "pytorch"):
            final_config["engine"] = engine or "pytorch"
        
        logger.info(f"Final config check before training: shuffle={final_shuffle}, trainingsetindex={final_trainingsetindex}")
        logger.info(f"Expected values: shuffle={shuffle}, trainingsetindex={trainingsetindex}")
        
        # Also verify metadata file has correct structure before training
        try:
            import yaml
            training_datasets_dir = Path(project_path) / "training-datasets"
            metadata_files = list(training_datasets_dir.rglob("metadata.yaml"))
            if metadata_files:
                metadata_file = metadata_files[0]
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                shuffle_key = int(shuffle)
                trainingset_key = int(trainingsetindex)
                shuffle_data = metadata.get("shuffles", {}).get(shuffle_key, {})
                trainingset_data = shuffle_data.get(trainingset_key, {})
                
                if "train_fraction" not in trainingset_data:
                    logger.warning(f"Metadata missing train_fraction! Fixing now...")
                    train_fraction = float(final_config.get("TrainingFraction", [0.95])[trainingsetindex])
                    if shuffle_key not in metadata.get("shuffles", {}):
                        metadata.setdefault("shuffles", {})[shuffle_key] = {}
                    metadata["shuffles"][shuffle_key][trainingset_key] = {
                        "train_fraction": train_fraction,
                        "shuffle": shuffle_key
                    }
                    with open(metadata_file, 'w') as f:
                        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
                    logger.info(f"Fixed metadata file: {metadata_file}")
        except Exception as e:
            logger.warning(f"Could not verify/fix metadata file: {e}")
        
        if final_trainingsetindex != trainingsetindex or final_shuffle != shuffle:
            logger.error(
                f"CRITICAL: Config file mismatch! "
                f"Expected shuffle={shuffle}, trainingsetindex={trainingsetindex}, "
                f"but found shuffle={final_shuffle}, trainingsetindex={final_trainingsetindex}. "
                f"Fixing now..."
            )
            final_config["trainingsetindex"] = int(trainingsetindex)
            final_config["shuffle"] = int(shuffle)
            dlc_reader.save_yaml(project_path, final_config, filename="config")
            # Update the filepath to point to the newly saved config
            dlc_cfg_filepath = dlc_reader.save_yaml(project_path, final_config, filename="config")
            
            # Verify one more time
            _, verify_config = dlc_reader.read_yaml(project_path)
            logger.info(f"After fix: shuffle={verify_config.get('shuffle')}, trainingsetindex={verify_config.get('trainingsetindex')}")
            
            # Update dlc_config to match
            dlc_config["trainingsetindex"] = int(trainingsetindex)
            dlc_config["shuffle"] = int(shuffle)

        # Final check: verify metadata one more time right before training
        # DLC reads metadata when train_network is called, so we need to ensure it's correct
        try:
            import yaml
            training_datasets_dir = Path(project_path) / "training-datasets"
            metadata_files = list(training_datasets_dir.rglob("metadata.yaml"))
            if metadata_files:
                metadata_file = metadata_files[0]
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                shuffle_key = int(shuffle)
                trainingset_key = int(trainingsetindex)
                
                # DLC 3.x expects: shuffles[shuffle] = {train_fraction, index, engine}
                # NOT nested by trainingsetindex!
                if "shuffles" not in metadata:
                    metadata["shuffles"] = {}
                
                shuffle_data = metadata.get("shuffles", {}).get(shuffle_key, {})
                
                # Check if it's old structure (nested) or new structure (flat)
                if isinstance(shuffle_data, dict) and any(isinstance(v, dict) for v in shuffle_data.values() if isinstance(v, dict)):
                    # Old structure: shuffles[shuffle][trainingsetindex] = {...}
                    logger.warning("Found old metadata structure (nested by trainingsetindex)! Converting to DLC 3.x format...")
                    trainingset_data = shuffle_data.get(trainingset_key, {})
                    train_fraction = float(trainingset_data.get("train_fraction", final_config.get("TrainingFraction", [0.95])[trainingsetindex]))
                    # CRITICAL: Use the engine variable directly, not from config (which might be wrong)
                    correct_engine = engine or "pytorch"
                    # Convert to new structure
                    # IMPORTANT: index is the SHUFFLE number, not trainingsetindex!
                    metadata["shuffles"][shuffle_key] = {
                        "train_fraction": train_fraction,
                        "index": shuffle_key,  # This is the SHUFFLE number, not trainingsetindex!
                        "engine": correct_engine,  # Use correct engine
                    }
                    with open(metadata_file, 'w') as f:
                        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
                    logger.info(f"Converted metadata to DLC 3.x format: {metadata_file}")
                else:
                    # New structure: shuffles[shuffle] = {train_fraction, index, engine}
                    # IMPORTANT: index is the SHUFFLE number, not trainingsetindex!
                    if shuffle_key not in metadata["shuffles"]:
                        metadata["shuffles"][shuffle_key] = {}
                    
                    shuffle_data = metadata["shuffles"][shuffle_key]
                    
                    # CRITICAL: Use the engine variable directly, not from config
                    correct_engine = engine or "pytorch"
                    
                    # Ensure train_fraction exists
                    if "train_fraction" not in shuffle_data:
                        train_fraction = float(final_config.get("TrainingFraction", [0.95])[trainingsetindex])
                        shuffle_data["train_fraction"] = train_fraction
                        shuffle_data["index"] = shuffle_key  # This is the SHUFFLE number, not trainingsetindex!
                        shuffle_data["engine"] = correct_engine
                        with open(metadata_file, 'w') as f:
                            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
                        logger.info(f"CRITICAL FIX: Added train_fraction to metadata right before training: {metadata_file}")
                        logger.info(f"Metadata structure: {metadata}")
                    else:
                        # Verify index matches shuffle number (not trainingsetindex!)
                        needs_update = False
                        if shuffle_data.get("index") != shuffle_key:
                            shuffle_data["index"] = shuffle_key
                            needs_update = True
                            logger.info(f"Fixed index mismatch (expected {shuffle_key}, found {shuffle_data.get('index')})")
                        
                        # CRITICAL: Ensure engine matches (DLC reads engine from metadata!)
                        current_engine = shuffle_data.get("engine")
                        if current_engine != correct_engine:
                            logger.warning(f"CRITICAL: Engine mismatch in metadata! Expected {correct_engine}, found {current_engine}. Fixing NOW...")
                            shuffle_data["engine"] = correct_engine
                            needs_update = True
                            logger.info(f"Updated engine in metadata from {current_engine} to {correct_engine}")
                        
                        if needs_update:
                            with open(metadata_file, 'w') as f:
                                yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
                            logger.info(f"Updated metadata file: {metadata_file}")
                        
                        logger.info(f"Metadata verified: train_fraction={shuffle_data.get('train_fraction')}, index={shuffle_data.get('index')} (shuffle number), engine={shuffle_data.get('engine')}")
        except Exception as e:
            logger.error(f"CRITICAL: Could not verify/fix metadata before training: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        try:
            # Log the exact metadata structure right before training
            logger.info("=" * 80)
            logger.info("FINAL METADATA CHECK BEFORE TRAINING")
            logger.info("=" * 80)
            training_datasets_dir = Path(project_path) / "training-datasets"
            metadata_files = list(training_datasets_dir.rglob("metadata.yaml"))
            if metadata_files:
                metadata_file = metadata_files[0]
                logger.info(f"Metadata file: {metadata_file}")
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                logger.info(f"Full metadata structure: {metadata}")
                logger.info(f"Shuffles: {metadata.get('shuffles', {})}")
                shuffle_key = int(shuffle)
                
                # DLC 3.x uses flat structure: shuffles[shuffle] = {train_fraction, index, engine}
                if shuffle_key in metadata.get('shuffles', {}):
                    shuffle_data = metadata['shuffles'][shuffle_key]
                    logger.info(f"Shuffle {shuffle_key} exists: {shuffle_data}")
                    
                    needs_update = False
                    
                    # CRITICAL: Ensure train_fraction exists and is the right type
                    if "train_fraction" not in shuffle_data:
                        train_fraction = float(final_config.get("TrainingFraction", [0.95])[trainingsetindex])
                        shuffle_data["train_fraction"] = train_fraction
                        needs_update = True
                        logger.warning(f"CRITICAL: train_fraction was missing! Adding it now: {train_fraction}")
                    else:
                        # Ensure it's a float, not int
                        shuffle_data["train_fraction"] = float(shuffle_data["train_fraction"])
                        logger.info(f"train_fraction exists: {shuffle_data['train_fraction']} (type: {type(shuffle_data['train_fraction'])})")
                    
                    # CRITICAL: Ensure index matches shuffle number
                    if shuffle_data.get("index") != shuffle_key:
                        shuffle_data["index"] = shuffle_key
                        needs_update = True
                        logger.warning(f"CRITICAL: index mismatch! Expected {shuffle_key}, found {shuffle_data.get('index')}. Fixing...")
                    
                    # CRITICAL: Ensure engine matches config (DLC reads engine from metadata!)
                    # Use the engine from the variable, not from final_config (which might be wrong)
                    expected_engine = engine or final_config.get("engine", "pytorch")
                    current_engine = shuffle_data.get("engine")
                    if current_engine != expected_engine:
                        logger.warning(f"CRITICAL: Engine mismatch! Expected {expected_engine}, found {current_engine}. Fixing...")
                        shuffle_data["engine"] = expected_engine
                        needs_update = True
                        logger.info(f"Updated engine in metadata from {current_engine} to {expected_engine}")
                    
                    if needs_update:
                        # Write the fixed metadata back immediately
                        with open(metadata_file, 'w') as f:
                            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
                        logger.info(f"Wrote fixed metadata to: {metadata_file}")
                        
                        # Verify it was written correctly
                        with open(metadata_file, 'r') as f:
                            verify_meta = yaml.safe_load(f)
                            verify_data = verify_meta['shuffles'][shuffle_key]
                            logger.info(f"Verification: train_fraction={verify_data.get('train_fraction')}, index={verify_data.get('index')}, engine={verify_data.get('engine')}")
                    else:
                        logger.info(f"Metadata verified: train_fraction={shuffle_data.get('train_fraction')}, index={shuffle_data.get('index')}, engine={shuffle_data.get('engine')}")
            logger.info("=" * 80)
            
            train_func(dlc_cfg_filepath, **train_network_kwargs)
        except KeyError as e:
            if "train_fraction" in str(e):
                logger.error(f"CRITICAL: DLC still can't find train_fraction! Error: {e}")
                logger.error("This suggests DLC is reading metadata from a different source or format.")
                logger.error("Please check DLC documentation or create an issue on GitHub.")
                raise
            else:
                raise
        except KeyboardInterrupt:  # Instructions indicate to train until interrupt
            print("DLC training stopped via Keyboard Interrupt")

        # DLC goes by snapshot magnitude when judging 'latest' for evaluation
        # Here, we mean most recently generated
        snapshots = sorted(model_train_folder.glob("snapshot*.index"))
        if not snapshots:
            raise FileNotFoundError(
                f"No snapshot files found in {model_train_folder}. "
                "Training may have failed or not generated any snapshots."
            )
        
        max_modified_time = 0
        latest_snapshot_file = None
        latest_snapshot = None
        
        for snapshot in snapshots:
            modified_time = snapshot.stat().st_mtime
            if modified_time > max_modified_time:
                latest_snapshot_file = snapshot
                latest_snapshot = int(
                    re.search(r"(\d+)\.index", latest_snapshot_file.name).group(1)
                )
                max_modified_time = modified_time

        if latest_snapshot_file is None:
            raise ValueError("Failed to determine latest snapshot file")
        
        # update snapshotindex in the config
        snapshotindex = snapshots.index(latest_snapshot_file)

        dlc_config["snapshotindex"] = snapshotindex
        edit_config(
            dlc_cfg_filepath,
            {"snapshotindex": snapshotindex},
        )

        self.insert1(
            {**key, "latest_snapshot": latest_snapshot, "config_template": dlc_config}
        )
