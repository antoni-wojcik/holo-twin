"""
Experiment I/O operations for saving data.
Loading files is handled by the user.
"""

import os, glob, re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import sys, inspect
from pathlib import Path
import shutil

from src.io import img_handler, santec_io
from src.config import DATA_ROOT, REPO_ROOT # path to the data folder, set in src/config.py

def data_path(relative_path: str) -> Path:
    """
    Convert a path relative to the data root folder to an absolute path.

    Parameters
    ----------
    relative_path : str
        Path relative to the data root folder.

    Returns
    -------
    path: Path
        Absolute path to the file or folder.
    """
    return DATA_ROOT / relative_path

class ExpIO:
    """
    Experiment I/O operations for saving data.
    Loading files is handled by the user.

    Parameters
    ----------
    base_name : str
        Base name of the experiment (e.g., "cgh_experiment")."
    exp_name : str, optional
        Specific name of the experiment (e.g., "run1"). If None, a generic name is used.
    description : str, optional
        Description of the experiment, saved in the info.txt file.
    log : bool, optional
        If True, logs all printed messages including stdout and stderr to a file in the experiment folder.
    """ 
    class _Tee:
        """
        A simple class to duplicate stdout and stderr to a file.
        """
        def __init__(self, filename):
            self.file = open(filename, "w")
            self.stdout = sys.stdout  # Keep original stdout

        def write(self, text):
            self.stdout.write(text)  # Print to console
            self.file.write(text)  # Save to file

        def flush(self):
            self.stdout.flush()
            self.file.flush()

    def __init__(self, base_name: str, exp_name: str = None, description: str = None, log: bool = True):
        self._base_name = base_name
        self._exp_name = exp_name
        self._description = description
        self._start_time = datetime.now()

        # Path to the experiment folder, will be created on first save
        self._path = None 

        # If logging is enabled, this will be a Tee object that duplicates stdout and stderr to a log file
        self._tee = None

        # Walk the call stack to find the first caller outside this file
        stack = inspect.stack()
        for frame_info in stack:
            caller_file = os.path.abspath(frame_info.filename)
            if caller_file != os.path.abspath(__file__):
                break
        self._caller_path = caller_file

        if log:
            self._start_logging()
            
        # copy all the source files in the caller's directory to a subdirectory in the experiment folder
        self._copy_source_tree()
    
    @property
    def path(self):
        """
        Returns the absolute path to the experiment folder if it has been created, otherwise None.
        """
        return self._path

    @property
    def relative_path(self):
        """
        Returns the experiment folder's path relative to DATA_ROOT, or None
        if it hasn't been created yet. This is the form config fields like
        acquisition_path/load_twin_path expect, so it's what a pipeline
        script should return for chaining into a downstream config.
        """
        if self._path is None:
            return None
        return os.path.relpath(self._path, DATA_ROOT)

    def get_new_path(self, name: str, extension: str, subdir_name: str = None):
        """
        Returns a unique path for saving a file, without actually creating the file.
        Used for saving files not supported by this class. Also creates the experiment 
        folder if it doesn't exist yet.

        Parameters
        ----------
        name : str
            Base name of the file (without extension).
        extension : str
            File extension (without dot).
        subdir_name : str, optional
            Name of a subdirectory under the experiment folder to save the file in.
            Used for grouping related files together. 
            If None, the file is saved directly in the experiment folder.
        """
        self._make_dir()
        path = self._get_unique_path(name, extension, subdir_name)
        return path
    
    def save_capture_pair(self, holo, capture, idx, subdir_name: str, save_images: bool = True):
        """
        Save an acquired (hologram, capture) pair as:
            `<subdir_name>/holos/<idx>.npy`
            `<subdir_name>/captures/<idx>.npy`
        Optionally saves the capture as a PNG image for quick viewing in:
            `<subdir_name>/captures/imgs/<idx>.png`
        """
        self.save_npy(holo, f"{idx}", subdir_name=f"{subdir_name}/holos")
        self.save_npy(capture, f"{idx}", subdir_name=f"{subdir_name}/captures")
        if save_images:
            self.save_image(capture, f"{idx}", subdir_name=f"{subdir_name}/captures/imgs")
            # self.save_hologram(holo, f"{idx}", subdir_name=f"{subdir_name}/holos/csv")

    def save_hologram(self, values, name="holo", subdir_name=None):
        path = self.get_new_path(name, 'csv', subdir_name)
        values_ushort = np.array(values, dtype=np.uint16)
        santec_io.save_hologram_csv(values_ushort, path)
        return path

    def save_figure(self, fig: plt.Figure, name="fig", subdir_name=None, type='pdf'):
        path = self.get_new_path(name, type, subdir_name)
        fig.savefig(path, bbox_inches='tight')
        return path

    def save_npy(self, data: np.ndarray, name, subdir_name=None):
        path = self.get_new_path(name, 'npy', subdir_name)
        np.save(path, data)
        return path

    def save_image(self, image, name, subdir_name=None, extension='png'):
        path = self.get_new_path(name, extension, subdir_name)
        img_handler.save_image(image, path)
        return path
    
    def save_image_color(self, image, cmap="gray", square=False, name=None, subdir_name=None, extension='png'):
        path = self.get_new_path(name or "image", extension, subdir_name)
        plt.imsave(path, image, cmap=cmap)
        return path

    def save_text(self, text, name="text", subdir_name=None):
        path = self.get_new_path(name, 'txt', subdir_name)
        with open(path, 'w') as f:
            f.write(text)
        return path

    def _start_logging(self):
        path = self.get_new_path("log", 'txt')
        self._tee = self._Tee(path)
        sys.stdout = self._tee
        sys.stderr = self._tee

    def _stop_logging(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def _get_unique_path_file(self, name, extension, base_path=None):
        """
        Generate a unique file path in self.path with the pattern:
          - first: name.extension
          - on 2nd: rename existing to name_0.extension, return name_1.extension
          - thereafter: name_n.extension for next available n
        """
        if base_path is None:
            base = os.path.join(self._path, name)
        else:
            base = os.path.join(base_path, name)
        plain = f"{base}.{extension}"
        # only consider suffixes that match numeric pattern: base_\n.ext
        pattern = re.compile(rf"^{re.escape(base)}_(\d+)\.{re.escape(extension)}$")
        all_suffixes = glob.glob(f"{base}_*.{extension}")
        numbered = [f for f in all_suffixes if pattern.match(f)]

        # if no plain file and no numbered files exist, use plain name
        if not os.path.exists(plain) and not numbered:
            return plain

        # if plain exists but _0 is missing, rename plain -> base_0.ext
        first = f"{base}_0.{extension}"
        if os.path.exists(plain) and not os.path.exists(first):
            os.rename(plain, first)
            # return next slot
            return f"{base}_1.{extension}"

        # scan existing numbered and pick next index
        nums = [int(pattern.match(f).group(1)) for f in numbered]
        next_idx = max(nums) + 1 if nums else 1
        return f"{base}_{next_idx}.{extension}"

    def _get_unique_path(self, name, extension, subdir_name=None):
        """
        Unified entry point for unique path generation:
          - if subdir_name is provided, use _get_unique_path_dir to place files in a subdirectory
        """
        if subdir_name is not None:
            subdir_path = os.path.join(self._path, subdir_name)
            os.makedirs(subdir_path, exist_ok=True)

            return self._get_unique_path_file(name, extension, base_path=subdir_path)
        else:
            return self._get_unique_path_file(name, extension)

    def _make_dir(self):
        if self._path is None:
            date_str = self._start_time.strftime('%Y-%m-%d')
            base_path = os.path.join(DATA_ROOT, f'{date_str}_{self._base_name}')
            os.makedirs(base_path, exist_ok=True)

            run_name = 'generic' if self._exp_name is None else self._exp_name

            # First run for this name gets the bare name, no suffix. Only once that
            # already exists do we start numbering, from _1 (never _0).
            candidate = run_name
            suffix = 1
            while os.path.exists(os.path.join(base_path, candidate)):
                candidate = f'{run_name}_{suffix}'
                suffix += 1

            self._path = os.path.join(base_path, candidate)
            os.makedirs(self._path)

            self._save_experiment_info()

    def _save_experiment_info(self):
        path = self.get_new_path('info', 'txt')
        with open(path, 'w') as f:
            f.write(f'Experiment: {self._base_name}, {self._exp_name}\n')
            # Write start time in year month day hour minute second format
            f.write(f'Start time: {self._start_time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            # Write which run directory this is (no longer a bare index -- see _make_dir)
            f.write(f'Run directory: {os.path.basename(self._path)}\n')
            # Write the description if it was provided
            if self._description:
                f.write(f'Description: {self._description}\n')
            else:
                f.write('No description provided\n')
            f.write('\n')

    def _copy_source_tree(self):
        """
        Copy all Python source files used for the project into

            <experiment>/scripts/

        preserving the directory structure.

        Copied locations:
            REPO_ROOT/
            REPO_ROOT/src/
            REPO_ROOT/pipelines/
        """

        scripts_root = Path(self._path) / "scripts"
        scripts_root.mkdir(exist_ok=True)

        project_root = Path(REPO_ROOT)

        # Check that the expected source directories exist, 
        # to avoid copying from an unexpected location
        assert (REPO_ROOT / "src").is_dir()
        assert (REPO_ROOT / "pipelines").is_dir()

        # Copy .py files from the project root
        for py_file in project_root.glob("*.py"):
            shutil.copy2(py_file, scripts_root / py_file.name)

        # Copy selected source directories recursively
        for dirname in ("src", "pipelines"):
            src_dir = project_root / dirname
            if not src_dir.exists():
                continue

            for py_file in src_dir.rglob("*.py"):
                relative = py_file.relative_to(project_root)
                destination = scripts_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(py_file, destination)

    def close(self):
        """
        Stop logging and close any open resources.
        """
        if self._tee:
            end_time = datetime.now()
            self.update_experiment_info(f'End time: {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
            self._stop_logging()
            self._tee.file.close()

    def update_experiment_info(self, info: str):
        """
        Append additional information to the experiment info file.
        """
        path = os.path.join(self._path, 'info.txt')
        with open(path, 'a') as f:
            f.write(f'{info}\n')

    # Context manager support for 'with' statement
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()