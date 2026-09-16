import pandas as pd
import numpy as np
import os, sys

SANTEC_GRAYSCALE_RANGE = 1024  # Santec SLM grayscale range (0-1023)

def save_hologram_csv(values: np.ndarray, path: str):
    """
    Save a 2D array of hologram values to a CSV file, using the data format 
    specified in the Santec SLM documentation. The values are converted to unsigned 16-bit 
    integers, and wrapped to the range [0, 1023].
    The first column and row of the CSV will be the indices (0, 1, 2, ...), 
    and the remaining cells contain the respective hologram values.

    Parameters
    ----------
    values : np.ndarray
        2D array of integr grayscale hologram values to be saved.
    path : str
        Path to the output CSV file.
    """
    values_fixed = values.astype(np.ushort) % SANTEC_GRAYSCALE_RANGE

    # Reconstruct the dataframe with flipped contents
    indices = [i for i in range(values_fixed.shape[0])]
    columns = [i for i in range(values_fixed.shape[1])]
    df = pd.DataFrame(values_fixed, columns=columns, index=indices)

    # Reinsert the first column
    df.insert(0, 'Y/X', indices)

    # Get the proper path
    df.to_csv(path, index=False)

def get_backplane_correction_grayscale(wavelength: float, path: str = None):
    """
    Get the backplane correction values from a csv file and convert them to grayscale values.
    
    Parameters
    ----------
    wavelength : float
        The wavelength in nanometers for which the backplane correction is to be calculated.
    path : str, optional
        Path to the CSV file containing backplane thickness values. If None, a default path is used.

    Returns
    -------
    np.ndarray
        An array of backplane correction values in grayscale, normalized to the range [0, 1023].
    """
    if path is None:
        path = os.path.join('data', 'backplane', 'backplane_correction.csv')

    in_df = pd.read_csv(path, header=None, index_col=None)

    # Select the contents of the file excluding the first column and row
    backplane_thickness = in_df.iloc[:, :].values

    num_wavelengths = backplane_thickness / wavelength * 1e3
    residual = num_wavelengths - np.floor(num_wavelengths)

    backplane_correction_values = residual * (SANTEC_GRAYSCALE_RANGE - 1) # 0 to 1023
    backplane_correction_values = backplane_correction_values.astype(np.uint16) % SANTEC_GRAYSCALE_RANGE

    return backplane_correction_values