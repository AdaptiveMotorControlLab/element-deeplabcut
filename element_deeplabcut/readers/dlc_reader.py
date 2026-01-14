import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from ruamel.yaml import YAML
from element_interface.utils import find_root_directory, dict_to_uuid
from datajoint.errors import DataJointError

logger = logging.getLogger("datajoint")


class PoseEstimation:
    """Class for handling DLC pose estimation files."""

    def __init__(
        self,
        dlc_dir: str = None,
        pkl_path: str = None,
        h5_path: str = None,
        yml_path: str = None,
        filename_prefix: str = "",
    ):
        if dlc_dir is None:
            assert pkl_path and h5_path and yml_path, (
                'If "dlc_dir" is not provided, then pkl_path, h5_path, and yml_path '
                + "must be provided"
            )
        else:
            self.dlc_dir = Path(dlc_dir)
            if not self.dlc_dir.exists():
                raise FileNotFoundError(f"Unable to find {dlc_dir}")

        # meta file: pkl - info about this DLC run (input video, configuration, etc.)
        # DLC 2.x uses *meta.pickle, DLC 3.x uses *_results.pickle or UUID-prefixed .pickle files
        if pkl_path is None:
            # Try DLC 2.x format first
            self.pkl_paths = sorted(
                self.dlc_dir.rglob(f"{filename_prefix}*meta.pickle")
            )
            # If not found, try DLC 3.x formats
            if not len(self.pkl_paths) > 0:
                # Try *_results.pickle pattern
                self.pkl_paths = sorted(
                    self.dlc_dir.rglob(f"{filename_prefix}*_results.pickle")
                )
            if not len(self.pkl_paths) > 0:
                # Try any .pickle file (DLC 3.x may use UUID prefixes)
                all_pickle = sorted(self.dlc_dir.rglob(f"{filename_prefix}*.pickle"))
                # Filter out non-meta files (prefer files with 'meta' or 'results' in name)
                meta_pickle = [p for p in all_pickle if 'meta' in p.name.lower() or 'results' in p.name.lower()]
                if meta_pickle:
                    self.pkl_paths = meta_pickle
                elif all_pickle:
                    # Fallback: use any pickle file if no meta/results found
                    self.pkl_paths = all_pickle
            
            # Check if we have H5 files - if so, pickle is optional (DLC 3.x may not create it)
            h5_files_exist = len(list(self.dlc_dir.glob(f"{filename_prefix}*.h5"))) > 0
            
            if not len(self.pkl_paths) > 0:
                if h5_files_exist:
                    # H5 files exist but no pickle - this is OK for DLC 3.x pretrained models
                    logger.warning(
                        f"No meta file (.pickle) found in: {self.dlc_dir}, "
                        "but H5 files are present. This is common for DLC 3.x pretrained models. "
                        "Will extract metadata from H5 files."
                    )
                    self.pkl_paths = []  # Empty list - we'll handle this in the pkl property
                else:
                    # No pickle AND no H5 files - this is an error
                    raise FileNotFoundError(
                        f"No meta file (.pickle) or H5 files found in: {self.dlc_dir}. "
                        f"Looked for: *meta.pickle, *_results.pickle, *.pickle, and *.h5"
                    )
        else:
            pkl_path = Path(pkl_path)
            if not pkl_path.exists():
                raise FileNotFoundError(f"{pkl_path} not found")
            self.pkl_paths = [pkl_path]

        # data file: h5 - body part outputs from the DLC post estimation step
        if h5_path is None:
            self.h5_paths = sorted(self.dlc_dir.rglob(f"{filename_prefix}*.h5"))
            if not len(self.h5_paths) > 0:
                raise FileNotFoundError(
                    f"No DLC output file (.h5) found in: {self.dlc_dir}"
                )
        else:
            h5_path = Path(h5_path)
            if not h5_path.exists():
                raise FileNotFoundError(f"{h5_path} not found")
            self.h5_paths = [h5_path]

        # validate number of files
        # DLC 3.x might have different number of pickle vs h5 files (pickle might be empty/placeholder)
        # So we allow mismatch but warn
        if len(self.h5_paths) != len(self.pkl_paths):
            logger.warning(
                f"Unequal number of .h5 files ({len(self.h5_paths)}) and .pickle files ({len(self.pkl_paths)}). "
                "This is common with DLC 3.x. Will extract metadata from H5 files if needed."
            )

        # DLC 2.x: pickle stem should match h5 stem + "_meta"
        # DLC 3.x: naming might be different, so we make this check more lenient
        if len(self.pkl_paths) > 0 and len(self.h5_paths) > 0:
            h5_stem = self.h5_paths[0].stem
            pkl_stem = self.pkl_paths[0].stem
            # Check if they match DLC 2.x pattern or if it's DLC 3.x (different naming)
            if not (pkl_stem == h5_stem + "_meta" or pkl_stem.endswith("_results") or "_" in pkl_stem):
                logger.warning(
                    f"Pickle file name ({pkl_stem}) doesn't match expected pattern for H5 file ({h5_stem}). "
                    "This might be DLC 3.x format with different naming convention."
                )

        # config file: yaml - configuration for invoking the DLC post estimation step
        # Note: DLC 3.x pretrained models may not create a YAML file in the output directory
        if yml_path is None:
            yml_paths = list(self.dlc_dir.glob(f"{filename_prefix}*.y*ml"))
            # If multiple, defer to the one we save.
            if len(yml_paths) > 1:
                yml_paths = [val for val in yml_paths if val.stem == "dj_dlc_config"]
            if len(yml_paths) == 0:
                # No YAML file found - this is common for DLC 3.x pretrained models
                # We can still read the results from H5/pickle files
                logger.warning(
                    f"No YAML file found in {self.dlc_dir}. "
                    "This is common for DLC 3.x pretrained model outputs. "
                    "Will proceed without config file."
                )
                self.yml_path = None
            elif len(yml_paths) == 1:
                self.yml_path = yml_paths[0]
            else:
                raise FileNotFoundError(
                    f"Unable to find one unique .yaml file in: {self.dlc_dir} - Found: {len(yml_paths)}"
                )
        else:
            self.yml_path = Path(yml_path)
            if not self.yml_path.exists():
                logger.warning(f"YAML path specified but not found: {self.yml_path}. Proceeding without it.")
                self.yml_path = None

        self._pkl = None
        self._rawdata = None
        self._yml = None
        self._data = None

        # Handle case where YAML is missing (common for DLC 3.x pretrained models)
        if not self.yml or len(self.yml) == 0:
            # For pretrained models, we may not have all the metadata
            # Use defaults or extract from pickle/H5 files
            logger.warning("YAML config is empty - using defaults for pretrained model metadata")
            
            # Try to extract what we can from pickle
            scorer = self.pkl.get("Scorer", "unknown")
            try:
                shuffle_match = re.search(r"shuffle(\d+)", scorer)
                shuffle = int(shuffle_match.groups()[0]) if shuffle_match else 0
            except (AttributeError, IndexError):
                shuffle = 0
            
            try:
                train_iter = int(scorer.split("_")[-1])
            except (ValueError, IndexError):
                train_iter = 0
            
            self.model = {
                "Scorer": scorer,
                "Task": self.pkl.get("Task", "pretrained"),
                "date": self.pkl.get("date", "unknown"),
                "iteration": self.pkl.get("iteration (active-learning)", 0),
                "shuffle": shuffle,
                "snapshotindex": -1,  # Default for pretrained models
                "trainingsetindex": 0,  # Default for pretrained models
                "training_iteration": train_iter,
            }
        else:
            # Original logic for trained models with YAML
            train_idx = np.where(
                (np.array(self.yml["TrainingFraction"]) * 100).astype(int)
                == int(self.pkl["training set fraction"] * 100)
            )[0][0]
            train_iter = int(self.pkl["Scorer"].split("_")[-1])

            self.model = {
                "Scorer": self.pkl["Scorer"],
                "Task": self.yml["Task"],
                "date": self.yml["date"],
                "iteration": self.pkl["iteration (active-learning)"],
                "shuffle": int(re.search(r"shuffle(\d+)", self.pkl["Scorer"]).groups()[0]),
                "snapshotindex": self.yml["snapshotindex"],
                "trainingsetindex": train_idx,
                "training_iteration": train_iter,
            }

        # Get fps and nframes from pickle if available, otherwise from H5
        pkl_data = self.pkl  # This will extract from H5 if pickle is missing
        self.fps = pkl_data.get("fps", 30)  # Default to 30 fps if not found
        self.nframes = pkl_data.get("nframes", len(self.rawdata) if hasattr(self, 'rawdata') else 0)
        self.creation_time = self.h5_paths[0].stat().st_mtime

    @property
    def pkl(self):
        """Pickle file contents"""
        if self._pkl is None:
            nframes = 0
            meta_hash = None
            valid_meta = None
            
            for fp in self.pkl_paths:
                try:
                    # Check if file is too small (likely empty or placeholder)
                    if fp.stat().st_size < 100:  # Less than 100 bytes
                        logger.warning(
                            f"Pickle file {fp} is very small ({fp.stat().st_size} bytes), "
                            "may be empty or placeholder. Trying to extract metadata from H5 files."
                        )
                        continue
                    
                    with open(fp, "rb") as f:
                        meta = pickle.load(f)
                    
                    # DLC 2.x format: meta["data"] contains the actual metadata
                    # DLC 3.x might have different structure
                    if isinstance(meta, dict) and "data" in meta:
                        meta_data = meta["data"]
                        nframes += meta_data.pop("nframes", 0)
                        
                        # remove variable fields
                        for k in ("start", "stop", "run_duration"):
                            meta_data.pop(k, None)
                        
                        # confirm identical setting in all .pickle files
                        if meta_hash is None:
                            meta_hash = dict_to_uuid(meta)
                            valid_meta = meta_data
                        else:
                            assert meta_hash == dict_to_uuid(
                                meta
                            ), f"Inconsistent DLC-model-config file used: {fp}"
                    else:
                        # DLC 3.x might have different structure
                        logger.warning(f"Unexpected pickle structure in {fp}, trying to extract from H5")
                        continue
                        
                except (EOFError, pickle.UnpicklingError, KeyError) as e:
                    logger.warning(
                        f"Could not read pickle file {fp}: {e}. "
                        "Trying to extract metadata from H5 files."
                    )
                    continue
            
            # If no valid pickle metadata found, try to extract from H5 files
            if valid_meta is None:
                logger.warning(
                    "No valid pickle metadata found. Extracting metadata from H5 files."
                )
                # Extract nframes from H5 files
                nframes = sum(len(pd.read_hdf(fp)) for fp in self.h5_paths)
                
                # Create minimal metadata structure
                # Try to get fps from yml config if available
                try:
                    fps = self.yml.get("fps", 30)
                except (AttributeError, NameError):
                    # yml not initialized yet, use default
                    fps = 30
                
                # Try to extract scorer/model info from H5 file structure or filename
                scorer = "DLC_3.x"
                if self.h5_paths:
                    # Try to extract from filename
                    h5_name = self.h5_paths[0].stem
                    # DLC 3.x filenames often contain model info
                    if "superanimal" in h5_name:
                        scorer = f"DLC_superanimal_{h5_name.split('superanimal')[1].split('_')[0]}"
                
                valid_meta = {
                    "nframes": nframes,
                    "fps": fps,
                    "Scorer": scorer,
                    "iteration (active-learning)": 0,
                    "training set fraction": 1.0,
                }
                logger.info(
                    f"Created minimal metadata: nframes={nframes}, fps={fps}, scorer={scorer}"
                )
            
            self._pkl = valid_meta
            self._pkl["nframes"] = nframes
        return self._pkl

    @property
    def yml(self):
        """json-structured config.yaml file contents"""
        if self._yml is None:
            if self.yml_path is None:
                # No YAML file available (common for DLC 3.x pretrained models)
                # Return an empty dict as fallback
                logger.warning("No YAML file available, returning empty config dict")
                self._yml = {}
            else:
                with open(self.yml_path, "rb") as f:
                    yaml = YAML(typ="safe", pure=True)
                    self._yml = yaml.load(f)
        return self._yml

    @property
    def rawdata(self):
        """Raw data from h5 file"""
        if self._rawdata is None:
            self._rawdata = pd.concat([pd.read_hdf(fp) for fp in self.h5_paths])
        return self._rawdata

    @property
    def data(self):
        """Data from the h5 file, restructured as a dict"""
        if self._data is None:
            self._data = self.reformat_rawdata()
        return self._data

    @property
    def df(self):
        """Data as dataframe"""
        top_level = self.rawdata.columns.levels[0][0]
        return self.rawdata.get(top_level)

    @property
    def body_parts(self):
        """Set of body parts present in data file"""
        return self.df.columns.levels[0]

    def reformat_rawdata(self):
        """Transform raw h5 data into dict"""
        # For DLC 3.x, nframes might not be in pickle, so use len(rawdata) as fallback
        expected_nframes = self.pkl.get("nframes", len(self.rawdata))
        if len(self.rawdata) != expected_nframes:
            logger.warning(
                f"Total frames from .h5 file ({len(self.rawdata)}) differs "
                f'from .pickle ({expected_nframes}). Using H5 file count.'
            )

        body_parts_position = {}
        
        # Check if this is DLC 2.x format (MultiIndex columns) or DLC 3.x format
        if isinstance(self.rawdata.columns, pd.MultiIndex):
            # Check the number of levels in the MultiIndex
            n_levels = self.rawdata.columns.nlevels
            
            if n_levels == 4:
                # DLC 3.x multi-animal format: (scorer, individuals, bodyparts, coords)
                # Structure: scorer -> individuals (animal0, animal1, ...) -> bodyparts -> coords (x, y, likelihood)
                logger.info("Detected DLC 3.x multi-animal format (4-level MultiIndex)")
                
                # Get the scorer (first level)
                scorer = self.rawdata.columns.levels[0][0]
                
                # Get all individuals (second level)
                individuals = self.rawdata.columns.levels[1]
                logger.debug(f"Found {len(individuals)} individual(s): {list(individuals)[:5]}")
                
                # For each individual, extract body parts
                for individual in individuals:
                    # Get data for this individual
                    try:
                        individual_data = self.rawdata.xs(individual, level=1, axis=1)
                    except KeyError:
                        logger.warning(f"Could not extract data for individual {individual} at level 1")
                        continue
                    
                    # After xs(level=1), we should have a 2-level MultiIndex: (bodypart, coords)
                    # Get body parts for this individual (third level, now first level after xs)
                    if isinstance(individual_data.columns, pd.MultiIndex):
                        # After xs(level=1), remaining levels should be (bodypart, coord)
                        # Body parts are at level 0 (was level 2 in original)
                        bodyparts = individual_data.columns.levels[0]
                        logger.debug(f"Individual {individual}: found {len(bodyparts)} body part(s) at level 0: {list(bodyparts)[:10]}")
                    else:
                        # Not a MultiIndex - try to infer body parts from column names
                        logger.warning(f"Individual {individual} data columns are not MultiIndex after xs. "
                                     f"Column type: {type(individual_data.columns)}, "
                                     f"Columns: {list(individual_data.columns[:10])}")
                        # Try to extract body parts from column names
                        # Columns might be like "bodypart_x", "bodypart_y", etc.
                        bodyparts = set()
                        for col in individual_data.columns:
                            # Try to extract body part name (first part before underscore)
                            if '_' in str(col):
                                bp_name = str(col).split('_')[0]
                                bodyparts.add(bp_name)
                        logger.debug(f"Extracted {len(bodyparts)} body part(s) from column names: {list(bodyparts)[:10]}")
                    
                    if len(bodyparts) == 0:
                        logger.error(f"No body parts found for individual {individual}. "
                                   f"Column structure: {type(individual_data.columns)}, "
                                   f"Number of columns: {len(individual_data.columns)}, "
                                   f"Sample columns: {list(individual_data.columns[:10])}")
                        # Don't create empty entry - skip this individual
                        continue
                    
                    for bodypart in bodyparts:
                        # Get coordinates for this body part
                        # After xs(level=1), bodypart is now at level=0 (was level=2)
                        try:
                            if isinstance(individual_data.columns, pd.MultiIndex):
                                bodypart_data = individual_data.xs(bodypart, level=0, axis=1)
                            else:
                                # Not MultiIndex - try to find columns matching this body part
                                bodypart_cols = [col for col in individual_data.columns if str(col).startswith(f"{bodypart}_")]
                                if bodypart_cols:
                                    bodypart_data = individual_data[bodypart_cols]
                                else:
                                    bodypart_data = None
                        except (KeyError, IndexError) as e:
                            logger.debug(f"Body part {bodypart} not found for individual {individual}: {e}")
                            bodypart_data = None
                        
                        if bodypart_data is not None and len(bodypart_data.columns) > 0:
                            # Create key: individual_bodypart (e.g., "animal0_nose")
                            key = f"{individual}_{bodypart}"
                            
                            # Extract x, y, likelihood from bodypart_data
                            # If MultiIndex, coords are at level 1 (was level 3)
                            # If not, columns are like "bodypart_x", "bodypart_y", etc.
                            if isinstance(bodypart_data.columns, pd.MultiIndex):
                                x_data = bodypart_data.xs("x", level=1, axis=1).values.flatten() if "x" in bodypart_data.columns.levels[1] else None
                                y_data = bodypart_data.xs("y", level=1, axis=1).values.flatten() if "y" in bodypart_data.columns.levels[1] else None
                                likelihood_data = bodypart_data.xs("likelihood", level=1, axis=1).values.flatten() if "likelihood" in bodypart_data.columns.levels[1] else np.ones(len(bodypart_data))
                                z_data = bodypart_data.xs("z", level=1, axis=1).values.flatten() if "z" in bodypart_data.columns.levels[1] else None
                            else:
                                # Flat columns
                                x_col = [c for c in bodypart_data.columns if 'x' in str(c).lower() and 'likelihood' not in str(c).lower()]
                                y_col = [c for c in bodypart_data.columns if 'y' in str(c).lower() and 'likelihood' not in str(c).lower()]
                                likelihood_col = [c for c in bodypart_data.columns if 'likelihood' in str(c).lower()]
                                z_col = [c for c in bodypart_data.columns if 'z' in str(c).lower()]
                                
                                x_data = bodypart_data[x_col[0]].values if x_col else None
                                y_data = bodypart_data[y_col[0]].values if y_col else None
                                likelihood_data = bodypart_data[likelihood_col[0]].values if likelihood_col else np.ones(len(bodypart_data))
                                z_data = bodypart_data[z_col[0]].values if z_col else None
                            
                            if x_data is not None and y_data is not None:
                                body_parts_position[key] = {
                                    "x": x_data,
                                    "y": y_data,
                                    "likelihood": likelihood_data,
                                }
                                if z_data is not None:
                                    body_parts_position[key]["z"] = z_data
                                logger.debug(f"Successfully extracted {bodypart} for {individual}: {len(x_data)} frames")
                            else:
                                logger.warning(f"Could not extract x/y coordinates for {individual}_{bodypart}")
                        else:
                            logger.warning(f"Could not extract data for body part {bodypart} of individual {individual}")
                
            elif n_levels == 3:
                # DLC 2.x format: MultiIndex columns like (scorer, bodypart, coords)
                logger.info("Detected DLC 2.x format (3-level MultiIndex)")
                for body_part in self.body_parts:
                    body_parts_position[body_part] = {
                        c: self.df.get(body_part).get(c).values
                        for c in self.df.get(body_part).columns
                    }
            else:
                logger.warning(f"Unexpected MultiIndex depth: {n_levels} levels")
                # Try to handle as 3-level (DLC 2.x)
                for body_part in self.body_parts:
                    body_parts_position[body_part] = {
                        c: self.df.get(body_part).get(c).values
                        for c in self.df.get(body_part).columns
                    }
        else:
            # DLC 3.x format: might have different structure
            # Try to infer structure from column names
            logger.info(f"DLC 3.x format detected. Columns: {list(self.rawdata.columns[:10])}")
            
            # Common patterns: columns might be like "bodypart_x", "bodypart_y", "bodypart_likelihood"
            # or MultiIndex but flattened
            if len(self.rawdata.columns) > 0:
                # Try to extract body parts from column names
                body_parts_set = set()
                coord_types = set()
                
                for col in self.rawdata.columns:
                    # Pattern: "scorer_bodypart_coord" or "bodypart_coord"
                    parts = str(col).split('_')
                    if len(parts) >= 2:
                        # Last part is usually the coordinate type (x, y, likelihood)
                        coord = parts[-1].lower()
                        if coord in ['x', 'y', 'z', 'likelihood']:
                            coord_types.add(coord)
                            # Everything before the last part is the body part name
                            body_part = '_'.join(parts[:-1])
                            body_parts_set.add(body_part)
                
                # If we found body parts, structure the data
                if body_parts_set and coord_types:
                    for body_part in body_parts_set:
                        body_parts_position[body_part] = {}
                        for coord in ['x', 'y', 'likelihood']:
                            # Try different column name patterns
                            possible_cols = [
                                f"{body_part}_{coord}",
                                f"DLC_{body_part}_{coord}",
                                f"DLC_superanimal_{body_part}_{coord}",
                            ]
                            found_col = None
                            for col_name in possible_cols:
                                if col_name in self.rawdata.columns:
                                    found_col = col_name
                                    break
                            
                            if found_col:
                                body_parts_position[body_part][coord] = self.rawdata[found_col].values
                            elif coord == 'z':  # z is optional
                                body_parts_position[body_part][coord] = None
                            else:
                                logger.warning(f"Could not find column for {body_part}_{coord}")
                else:
                    # Fallback: try to use MultiIndex structure even if not detected
                    logger.warning("Could not parse DLC 3.x column structure, trying MultiIndex fallback")
                    try:
                        # Try to access as if it's MultiIndex
                        top_level = self.rawdata.columns.levels[0][0] if hasattr(self.rawdata.columns, 'levels') else None
                        if top_level:
                            df = self.rawdata.get(top_level)
                            for body_part in df.columns.levels[0] if hasattr(df.columns, 'levels') else []:
                                body_parts_position[body_part] = {
                                    c: df.get(body_part).get(c).values
                                    for c in df.get(body_part).columns
                                }
                    except Exception as e:
                        logger.error(f"Failed to parse H5 file structure: {e}")
                        raise ValueError(
                            f"Could not parse H5 file structure. "
                            f"Columns: {list(self.rawdata.columns[:20])}. "
                            f"Please check the file format."
                        )

        return body_parts_position


def read_yaml(fullpath: str, filename: str = "*") -> tuple:
    """Return contents of yml in fullpath. If available, defer to DJ-saved version

    Args:
        fullpath (str): String or pathlib path. Directory with yaml files
        filename (str, optional): Filename, no extension. Permits wildcards.

    Returns:
        Tuple of (a) filepath as pathlib.PosixPath and (b) file contents as dict
    """
    from deeplabcut.utils.auxiliaryfunctions import read_config

    fullpath = Path(fullpath)
    
    # Ensure it's a directory, not a file
    if fullpath.is_file():
        fullpath = fullpath.parent
        logger.warning(f"read_yaml received a file path, using parent directory: {fullpath}")
    
    if not fullpath.exists():
        raise FileNotFoundError(f"Directory does not exist: {fullpath}")
    
    if not fullpath.is_dir():
        raise ValueError(f"Path is not a directory: {fullpath}")

    # Take the DJ-saved if there. If not, return list of available
    yml_paths = list(fullpath.glob("dj_dlc_config.yaml")) or sorted(
        list(fullpath.glob(f"{filename}.y*ml"))
    )

    if not yml_paths:
        raise FileNotFoundError(f"No YAML files found in: {fullpath}")

    # If multiple YAML files are present, choose the most appropriate one:
    # 1. Prefer explicit config.yaml/config.yml
    # 2. Then prefer dj_dlc_config*.yaml (most recent by modification time)
    # 3. Otherwise fall back to the first in the sorted list with a warning
    chosen_path = None

    # Prefer standard DLC config filenames
    for name in ("config.yaml", "config.yml"):
        for p in yml_paths:
            if p.name == name:
                chosen_path = p
                break
        if chosen_path:
            break

    # Prefer dj_dlc_config* if no explicit config.* was found
    if chosen_path is None:
        dj_configs = [p for p in yml_paths if p.name.startswith("dj_dlc_config")]
        if dj_configs:
            # Choose the most recently modified dj_dlc_config*
            chosen_path = max(dj_configs, key=lambda p: p.stat().st_mtime)

    # Fallback: first match, but emit a warning for debugging
    if chosen_path is None:
        chosen_path = yml_paths[0]
        if len(yml_paths) > 1:
            logger.warning(
                "Multiple YAML files found in %s, using %s. Candidates: %s",
                fullpath,
                chosen_path,
                [p.name for p in yml_paths],
            )

    return chosen_path, read_config(chosen_path)


def save_yaml(
    output_dir: str,
    config_dict: dict,
    filename: str = "dj_dlc_config",
    mkdir: bool = True,
) -> str:
    """Save config_dict to output_path as filename.yaml. By default, preserves original.

    Args:
        output_dir (str): where to save yaml file (directory path)
        config_dict (str): dict of config params or element-deeplabcut model.Model dict
        filename (str, optional): default 'dj_dlc_config' or preserve original 'config'
            Set to 'config' to overwrite original file.
            If extension is included, removed and replaced with "yaml".
        mkdir (bool): Optional, True. Make new directory if output_dir not exist

    Returns:
        path of saved file as string - due to DLC func preference for strings
    """
    from deeplabcut.utils.auxiliaryfunctions import write_config

    if "config_template" in config_dict:  # if passed full model.Model dict
        config_dict = config_dict["config_template"]
    
    output_dir = Path(output_dir)
    
    # Ensure it's a directory, not a file
    if output_dir.is_file():
        output_dir = output_dir.parent
        logger.warning(f"save_yaml received a file path, using parent directory: {output_dir}")
    
    if mkdir:
        output_dir.mkdir(exist_ok=True)
    if "." in filename:  # if user provided extension, remove
        filename = filename.split(".")[0]

    output_filepath = output_dir / f"{filename}.yaml"
    write_config(output_filepath, config_dict)
    return str(output_filepath)


def do_pose_estimation(
    key: dict,
    video_filepaths: list,
    dlc_model: dict,
    project_path: str,
    output_dir: str,
    videotype="",
    gputouse=None,
    save_as_csv=False,
    batchsize=None,
    cropping=None,
    TFGPUinference=True,
    dynamic=(False, 0.5, 10),
    robust_nframes=False,
    allow_growth=False,
    use_shelve=False,
):
    """Launch DLC's analyze_videos within element-deeplabcut.

    Also saves a copy of the current config in the output dir, with ensuring analyzed
    videos in the video_set. NOTE: Config-specificed cropping not supported when adding
    to config in this manner.

    Args:
        video_filepaths (list): list of videos to analyze
        dlc_model (dict): element-deeplabcut dlc.Model
        project_path (str): path to project config.yml
        output_dir (str): where to save output
            # BELOW FROM DLC'S DOCSTRING

        videotype (str, optional, default=""):
            Checks for the extension of the video in case the input to the video is a
            directory. Only videos with this extension are analyzed. If unspecified,
            videos with common extensions ('avi', 'mp4', 'mov', 'mpeg', 'mkv') are kept.
        gputouse (int or None, optional, default=None):
            Indicates the GPU to use (see number in ``nvidia-smi``). If none, ``None``.
            See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
        save_as_csv (bool, optional, default=False):
            Saves the predictions in a .csv file.
        batchsize (int or None, optional, default=None):
            Change batch size for inference; if given overwrites ``pose_cfg.yaml``
        cropping (list or None, optional, default=None):
            List of cropping coordinates as [x1, x2, y1, y2].
            Note that the same cropping parameters will then be used for all videos.
            If different video crops are desired, run ``analyze_videos`` on individual
            videos with the corresponding cropping coordinates.
        TFGPUinference (bool, optional, default=True):
            Perform inference on GPU with TensorFlow code. Introduced in "Pretraining
            boosts out-of-domain robustness for pose estimation" by Alexander Mathis,
            Mert Yüksekgönül, Byron Rogers, Matthias Bethge, Mackenzie W. Mathis.
            Source https://arxiv.org/abs/1909.11229
        dynamic (tuple(bool, float, int) triple (state, detectiontreshold, margin)):
            If the state is true, then dynamic cropping will be performed. That means
            that if an object is detected (i.e. any body part > detectiontreshold),
            then object boundaries are computed according to the smallest/largest x
            position and smallest/largest y position of all body parts. This  window is
            expanded by the margin and from then on only the posture within this crop
            is analyzed (until the object is lost, i.e. <detectiontreshold). The
            current position is utilized for updating the crop window for the next
            frame (this is why the margin is important and should be set large enough
            given the movement of the animal).
        robust_nframes (bool, optional, default=False):
            Evaluate a video's number of frames in a robust manner.
            This option is slower (as the whole video is read frame-by-frame),
            but does not rely on metadata, hence its robustness against file corruption.
        allow_growth (bool, optional, default=False.):
            For some smaller GPUs the memory issues happen. If ``True``, the memory
            allocator does not pre-allocate the entire specified GPU memory region,
            instead starting small and growing as needed.
            See issue: https://forum.image.sc/t/how-to-stop-running-out-of-vram/30551/2
        use_shelve (bool, optional, default=False):
            By default, data are dumped in a pickle file at the end of the video
            analysis. Otherwise, data are written to disk on the fly using a "shelf";
            i.e., a pickle-based, persistent, database-like object by default,
            resulting in constant memory footprint.

    """
    # this function should no longer be used, throw a deprecation warning
    logger.warning(
        "This function is deprecated and will be removed in a future release. "
        + "Its usage is now incorporated into model.PoseEstimation's `make` function"
    )

    from deeplabcut.pose_estimation_tensorflow import analyze_videos

    # ---- Build and save DLC configuration (yaml) file ----
    dlc_config = dlc_model["config_template"]
    dlc_project_path = Path(project_path)
    dlc_config["project_path"] = dlc_project_path.as_posix()

    # ---- Add current video to config ---
    # FIXME: I don't think the code block below is necessary
    for video_filepath in video_filepaths:
        if video_filepath not in dlc_config["video_sets"]:
            try:
                from .. import model

                px_width, px_height = (model.RecordingInfo & key).fetch1(
                    "px_width", "px_height"
                )
            except DataJointError:
                logger.warn(
                    f"Could not find RecordingInfo for {video_filepath.stem}"
                    + "\n\tUsing zeros for crop value in config."
                )
                px_height, px_width = 0, 0
            dlc_config["video_sets"].update(
                {str(video_filepath): {"crop": f"0, {px_width}, 0, {px_height}"}}
            )

    # ---- Write config files ----
    # To output dir: Important for loading/parsing output in datajoint
    _ = save_yaml(output_dir, dlc_config)
    # To project dir: Required by DLC to run the analyze_videos
    if dlc_project_path != output_dir:
        config_filepath = save_yaml(dlc_project_path, dlc_config)

    # ---- Trigger DLC prediction job ----
    analyze_videos(
        config=config_filepath,
        videos=video_filepaths,
        shuffle=dlc_model["shuffle"],
        trainingsetindex=dlc_model["trainingsetindex"],
        destfolder=output_dir,
        modelprefix=dlc_model["model_prefix"],
        videotype=videotype,
        gputouse=gputouse,
        save_as_csv=save_as_csv,
        batchsize=batchsize,
        cropping=cropping,
        TFGPUinference=TFGPUinference,
        dynamic=dynamic,
        robust_nframes=robust_nframes,
        allow_growth=allow_growth,
        use_shelve=use_shelve,
    )
