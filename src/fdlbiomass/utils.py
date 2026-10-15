from __future__ import annotations

import numpy as np
import tifffile as tiff
from shapely.geometry import Point, Polygon


def read_biomass_data(full_path, crop_struct, data_type):
    """
    Reads biomass data from the file write_biomass_data.m (not processed)

    Parameters:
    full_path (str): The full path to the file.
    crop_struct (dict): A dictionary containing cropping parameters.
        Keys:
            'S': number of samples (dia only)
            'L': number of lines (dia only)
            'Sin': first sample to be read (starting from 1)
            'Sfin': last sample to be read
            'Lin': first line to be read (starting from 1)
            'Lfin': last line to be read
    data_type (int): The type of data.
        1 for Complex
        2 for Real

    Returns:
    np.ndarray: The read data.
    """
    # Single precision size
    N = 4

    try:
        with open(full_path, "rb") as fid:
            # Read total dimensions
            NrTot = int(np.fromfile(fid, dtype="<f4", count=1)[0])
            NcTot = int(np.fromfile(fid, dtype="<f4", count=1)[0])

            if not crop_struct:
                crop_struct = {
                    "S": NrTot,
                    "Sin": 1,
                    "Sfin": NrTot,
                    "L": NcTot,
                    "Lin": 1,
                    "Lfin": NcTot,
                }

            if "S" in crop_struct and "L" in crop_struct:
                if NrTot != crop_struct["S"] or NcTot != crop_struct["L"]:
                    print(
                        f"Warning: Dimension mismatch. Expected {crop_struct['S']}x{crop_struct['L']}, got {NrTot}x{NcTot}"
                    )

            Nr = int(crop_struct["Sfin"] - crop_struct["Sin"] + 1)
            Nc = int(crop_struct["Lfin"] - crop_struct["Lin"] + 1)

            if data_type == 1:  # Complex
                fid.seek(
                    2 * N
                    + ((crop_struct["Lin"] - 1) * NrTot + crop_struct["Sin"] - 1)
                    * N
                    * 2
                )

                if crop_struct["Sin"] == 1 and crop_struct["Sfin"] == NrTot:
                    I = np.fromfile(fid, dtype="<f4", count=2 * Nr * Nc).reshape(
                        2 * Nr, Nc, order="F"
                    )
                else:
                    I = np.zeros((2 * Nr, Nc), dtype="<f4")
                    for i in range(Nc):
                        I[:, i] = np.fromfile(fid, dtype="<f4", count=2 * Nr)
                        fid.seek(
                            (NrTot - crop_struct["Sfin"] + crop_struct["Sin"] - 1)
                            * N
                            * 2,
                            1,
                        )

                I = I[0::2] + 1j * I[1::2]

            elif data_type == 2:  # Real
                fid.seek(
                    2 * N
                    + ((crop_struct["Lin"] - 1) * NrTot + crop_struct["Sin"] - 1) * N
                )

                if crop_struct["Sin"] == 1 and crop_struct["Sfin"] == NrTot:
                    I = np.fromfile(fid, dtype="<f4", count=Nr * Nc).reshape(
                        Nr, Nc, order="F"
                    )
                else:
                    I = np.zeros((Nr, Nc), dtype="<f4")
                    for i in range(Nc):
                        I[:, i] = np.fromfile(fid, dtype="<f4", count=Nr)
                        fid.seek(
                            (NrTot - crop_struct["Sfin"] + crop_struct["Sin"] - 1) * N,
                            1,
                        )

            else:
                print("Error: Unrecognized data type.")
                return None

    except OSError:
        print(f"Error: Cannot open file {full_path}")
        return None

    return I


import numpy as np
from scipy.io import loadmat


def load_biomass_data(
    parent_path,
    band_opt,
    heading=None,
    acquisition_mode=None,
    polarimetric_channel=None,
):
    print("Starting to load the biomass data")
    # Loading data
    # Loading normalized SAR intensity data cube
    if band_opt == "P":
        # Loading angles
        angles_path = f"{parent_path}/{band_opt}/PowerCubeGeocod/AnglesUTM.mat"
        angles_data = loadmat(angles_path)
        IncAngleUTM = angles_data["IncAngleUTM"]
        easting_axis = angles_data["easting_axis"].squeeze()
        northing_axis = angles_data["northing_axis"].squeeze()

        # Loading LiDAR data
        lidar_path = f"{parent_path}/{band_opt}/PowerCubeGeocod/LidarUTM.mat"
        lidar_data = loadmat(lidar_path)
        LidarCHM = lidar_data["LidarCHM"]

        # Loading TomoDeRamped or TomoDeRampedDeBiased data
        pcube_norm_path = f"{parent_path}/{band_opt}/PowerCubeGeocod/PcubeNormUTM.mat"
        pcube_norm_data = loadmat(pcube_norm_path)
        PcubeNormUTM = pcube_norm_data["PcubeNormUTM"]
        z_axis = pcube_norm_data["z_axis"].squeeze()

    elif band_opt in ["L_Mono", "L_Bi"]:
        # Loading angles
        angles_path = f"{parent_path}/L/PowerCubeGeocod/AnglesUTM.mat"
        angles_data = loadmat(angles_path)
        IncAngleUTM = angles_data["IncAngleUTM"]
        easting_axis = angles_data["easting_axis"].squeeze()
        northing_axis = angles_data["northing_axis"].squeeze()

        # Loading LiDAR data
        lidar_path = f"{parent_path}/L/PowerCubeGeocod/LidarUTM.mat"
        lidar_data = loadmat(lidar_path)
        LidarCHM = lidar_data["LidarCHM"]

        # Loading TomoDeRamped or TomoDeRampedDeBiased data
        pcube_norm_path = f"{parent_path}/L/PowerCubeGeocod/PcubeNormUTM.mat"
        pcube_norm_data = loadmat(pcube_norm_path)
        PcubeNormUTM = pcube_norm_data["PcubeNormUTM"]
        z_axis = pcube_norm_data["z_axis"].squeeze()

    elif band_opt == "PL":
        # Loading angles
        angles_path_P = f"{parent_path}/P/PowerCubeGeocod/AnglesUTM.mat"
        angles_data_P = loadmat(angles_path_P)
        IncAngleUTM_P = angles_data_P["IncAngleUTM"]
        easting_axis_P = angles_data_P["easting_axis"].squeeze()
        northing_axis_P = angles_data_P["northing_axis"].squeeze()
        angles_path_L = f"{parent_path}/L/PowerCubeGeocod/AnglesUTM.mat"
        angles_data_L = loadmat(angles_path_L)
        IncAngleUTM_L = angles_data_L["IncAngleUTM"]
        easting_axis_L = angles_data_L["easting_axis"].squeeze()
        northing_axis_L = angles_data_L["northing_axis"].squeeze()

        # Loading LiDAR data (LiDAR data is same between P and L bands)
        lidar_path = f"{parent_path}/P/PowerCubeGeocod/LidarUTM.mat"
        lidar_data = loadmat(lidar_path)
        LidarCHM = lidar_data["LidarCHM"]

        pcube_norm_path_P = f"{parent_path}/P/PowerCubeGeocod/PcubeNormUTM.mat"
        pcube_norm_path_L = f"{parent_path}/L/PowerCubeGeocod/PcubeNormUTM.mat"
        pcube_norm_data_P = loadmat(pcube_norm_path_P)
        pcube_norm_data_L = loadmat(pcube_norm_path_L)
        PcubeNormUTM_P = pcube_norm_data_P["PcubeNormUTM"]
        PcubeNormUTM_L = pcube_norm_data_L["PcubeNormUTM"]
        z_axis_P = pcube_norm_data_P["z_axis"].squeeze()
        z_axis_L = pcube_norm_data_L["z_axis"].squeeze()

    # Loading SAR intensity data cube
    if band_opt == "P":
        sar_intensity_path = (
            f"{parent_path}/{band_opt}/PowerCubeGeocod/PcubeDeRampedUTM.mat"
        )
        sar_intensity_data = loadmat(sar_intensity_path)
        PcubeDeRampedUTM = sar_intensity_data["PcubeDeRampedUTM"]
    elif band_opt in ["L_Mono", "L_Bi"]:
        sar_intensity_path = (
            f"{parent_path}/L/PowerCubeGeocod/PcubeDeRampedDeBiasedUTM.mat"
        )
        sar_intensity_data = loadmat(sar_intensity_path)
        PcubeDeRampedDeBiasedUTM = sar_intensity_data["PcubeDeRampedDeBiasedUTM"]
    elif band_opt == "PL":
        sar_intensity_path_p = f"{parent_path}/P/PowerCubeGeocod/PcubeDeRampedUTM.mat"
        sar_intensity_path_l = (
            f"{parent_path}/L/PowerCubeGeocod/PcubeDeRampedDeBiasedUTM.mat"
        )
        sar_intensity_data_p = loadmat(sar_intensity_path_p)
        sar_intensity_data_l = loadmat(sar_intensity_path_l)
        PcubeDeRampedUTM = sar_intensity_data_p["PcubeDeRampedUTM"]
        PcubeDeRampedDeBiasedUTM = sar_intensity_data_l["PcubeDeRampedDeBiasedUTM"]

    head_ind, mono_bi_ind, pol_ind = None, None, None
    if heading is not None:
        head_ind = 0 if heading == "NW" else 1
    if acquisition_mode is not None:
        mono_bi_ind = 0 if acquisition_mode == "monostatic" else 1
    if polarimetric_channel is not None:
        pol_ind = {"HH": 0, "HV": 1, "VV": 2}[polarimetric_channel]
    print("Load complete")

    if band_opt == "P":
        return {
            "IncAngleUTM": IncAngleUTM,
            "easting_axis": easting_axis,
            "northing_axis": northing_axis,
            "LidarCHM": LidarCHM,
            "PcubeNormUTM": PcubeNormUTM,
            "z_axis": z_axis,
            "PcubeDeRampedUTM": PcubeDeRampedUTM,
            "head_ind": head_ind,
            "mono_bi_ind": mono_bi_ind,
            "pol_ind": pol_ind,
        }
    elif band_opt in ["L_Mono", "L_Bi"]:
        return {
            "IncAngleUTM": IncAngleUTM,
            "easting_axis": easting_axis,
            "northing_axis": northing_axis,
            "LidarCHM": LidarCHM,
            "PcubeNormUTM": PcubeNormUTM,
            "z_axis": z_axis,
            "PcubeDeRampedDeBiasedUTM": PcubeDeRampedDeBiasedUTM,
            "head_ind": head_ind,
            "mono_bi_ind": mono_bi_ind,
            "pol_ind": pol_ind,
        }
    elif band_opt == "PL":
        return {
            "IncAngleUTM_P": IncAngleUTM_P,
            "IncAngleUTM_L": IncAngleUTM_L,
            "easting_axis_P": easting_axis_P,
            "easting_axis_L": easting_axis_L,
            "northing_axis_P": northing_axis_P,
            "northing_axis_L": northing_axis_L,
            "LidarCHM": LidarCHM,
            "PcubeNormUTM_P": PcubeNormUTM_P,
            "PcubeNormUTM_L": PcubeNormUTM_L,
            "z_axis_P": z_axis_P,
            "z_axis_L": z_axis_L,
            "PcubeDeRampedUTM": PcubeDeRampedUTM,
            "PcubeDeRampedDeBiasedUTM": PcubeDeRampedDeBiasedUTM,
            "head_ind": head_ind,
            "mono_bi_ind": mono_bi_ind,
            "pol_ind": pol_ind,
        }
    else:
        raise ValueError(
            "Invalid value for band_opt: must be 'P', 'L_Mono', 'L_Bi', or 'PL'"
        )


def load_classification_tree_data(parent_path):
    """
    Load classification tree data from the file vm_btpolygonproved_target_clip.tif (processed)

    Parameters:
    parent_path (str): The parent path to the file.

    Returns:
    np.ndarray: The read data. Class 0 represents not known tree class for that pixel.
    """
    # Single precision size

    # command to download the data !gcloud storage cp gs://2024-esl-biomasssar-development-landing/tree_species/vm_btpolygonproved_target_clip.tif
    try:
        tree_classification_tif = tiff.imread(
            f"{parent_path}/vm_btpolygonproved_target_clip_mask_chm_rev_flipped.tif"
        )
        classification_tree_array = np.array(tree_classification_tif)
    except OSError:
        print(f"Error: Cannot open file {parent_path}")
        return None

    plot_classification_data(classification_tree_array)
    return classification_tree_array

def plot_classification_data(classification_tree_array):
    ''''
    Function to plot the classification data

    @param
    classification_tree_array: numpy array with the classification
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as mcolors

    # Plot Classification data
    cmap = plt.get_cmap('tab10', 10)
    cmap = mcolors.ListedColormap(['white'] + list(cmap.colors))
    bounds = np.arange(-0.5, 10.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    fig = plt.figure(figsize=(10, 6), dpi=100)
    img = plt.imshow(classification_tree_array, cmap=cmap, norm=norm)

    # Define custom labels for the colorbar
    labels = [
        'No data: unknown',  # Corresponding to 0 value
        'Aspen forest',
        'Pine forest',
        'Beech forest with deciduous woods',
        'Douglas fir forest',
        'Mixed spruce forest with native deciduous woods',
        'Oak-beech forest',
        'Oak forest',
        'Spruce forest',
        'Beech forest'
    ]

    # Create a colorbar with a custom size
    cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.6])  # [left, bottom, width, height]
    cbar = plt.colorbar(img, cax=cbar_ax, label='Tree Species')
    cbar.set_ticks(np.arange(0, 10))  # Set ticks to match the number of labels
    cbar.set_ticklabels(labels)  # Set tick labels
    plt.title('Classification Data: Ground Truth')
    plt.savefig('image_classification.png')


def merge_3d_arrays_with_max(arr1, arr2):
    # Convert inputs to numpy arrays if they aren't already
    arr1 = np.array(arr1, dtype=float)
    arr2 = np.array(arr2, dtype=float)

    # Check if arrays have the same shape
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape")

    # Create masks for non-NaN values
    mask1 = ~np.isnan(arr1)
    mask2 = ~np.isnan(arr2)

    # Combine the masks
    combined_mask = mask1 | mask2

    # Initialize the result array
    result = np.empty_like(arr1)

    # Where both arrays have values, take the maximum
    both_valid = mask1 & mask2
    result[both_valid] = np.maximum(arr1[both_valid], arr2[both_valid])

    # Where only arr1 has a value, use that
    only_arr1 = mask1 & ~mask2
    result[only_arr1] = arr1[only_arr1]

    # Where only arr2 has a value, use that
    only_arr2 = ~mask1 & mask2
    result[only_arr2] = arr2[only_arr2]

    # Where neither has a value, set to NaN
    result[~combined_mask] = np.nan

    return result


def merge_3d_arrays_with_mean(arr1, arr2):
    # Convert inputs to numpy arrays if they aren't already
    arr1 = np.array(arr1, dtype=float)
    arr2 = np.array(arr2, dtype=float)

    # Check if arrays have the same shape
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape")

    # Create masks for non-NaN values
    mask1 = ~np.isnan(arr1)
    mask2 = ~np.isnan(arr2)

    # Combine the masks
    combined_mask = mask1 | mask2

    # Initialize the result array
    result = np.empty_like(arr1)

    # Where both arrays have values, take the mean
    both_valid = mask1 & mask2
    result[both_valid] = (arr1[both_valid] + arr2[both_valid]) / 2

    # Where only arr1 has a value, use that
    only_arr1 = mask1 & ~mask2
    result[only_arr1] = arr1[only_arr1]

    # Where only arr2 has a value, use that
    only_arr2 = ~mask1 & mask2
    result[only_arr2] = arr2[only_arr2]

    # Where neither has a value, set to NaN
    result[~combined_mask] = np.nan

    return result


def merge_headings(df):

    head1 = df[:, :, :, :, 0]
    head2 = df[:, :, :, :, 1]

    return merge_3d_arrays_with_max(head1, head2)


def split_into_quadrants(array3d, array2d, array2d_coordinates=None):
    """Split two arrays into quadrants for two arrays with x,y,z and x,y shape
    array3d: power cube
    array2d: lidar data
    array2d_coordinates: coordinate_system data
    returns two tuples: one with 4 quadrants of array3d (intensity) and one with ground truth lidar data
    """
    print(array3d.shape[1], array2d.shape[1])
    if array3d.shape[0] != array2d.shape[0] or array3d.shape[1] != array2d.shape[1]:
        raise ValueError("shape mismatch between two arrays")

    x_mid = array3d.shape[0] // 2
    y_mid = array3d.shape[1] // 2

    # Define the quadrants
    q1 = array3d[:x_mid, :y_mid, :]
    q2 = array3d[:x_mid, y_mid:, :]
    q3 = array3d[x_mid:, :y_mid, :]
    q4 = array3d[x_mid:, y_mid:, :]

    gt1 = array2d[:x_mid, :y_mid]
    gt2 = array2d[:x_mid, y_mid:]
    gt3 = array2d[x_mid:, :y_mid]
    gt4 = array2d[x_mid:, y_mid:]

    if type(array2d_coordinates) != type(None):
        cs1 = array2d_coordinates[:x_mid, :y_mid]
        cs2 = array2d_coordinates[:x_mid, y_mid:]
        cs3 = array2d_coordinates[x_mid:, :y_mid]
        cs4 = array2d_coordinates[x_mid:, y_mid:]
        return (q1, q2, q3, q4), (gt1, gt2, gt3, gt4), (cs1, cs2, cs3, cs4)

    return (q1, q2, q3, q4), (gt1, gt2, gt3, gt4)


def crop_to_non_nan(array):
    """
    Crop a 2D numpy array to remove NaN values from edges.

    :param array: 2D numpy array
    :return: Cropped 2D numpy array
    """
    # Find rows and columns that don't contain NaN
    nan_rows = np.all(np.isnan(array), axis=1)
    nan_cols = np.all(np.isnan(array), axis=0)

    # Find the first and last non-NaN row and column
    first_valid_row = np.argmax(~nan_rows)
    last_valid_row = len(nan_rows) - np.argmax(~nan_rows[::-1]) - 1
    first_valid_col = np.argmax(~nan_cols)
    last_valid_col = len(nan_cols) - np.argmax(~nan_cols[::-1]) - 1

    # Crop the array
    cropped_array = array[
        first_valid_row : last_valid_row + 1, first_valid_col : last_valid_col + 1
    ]

    return cropped_array


def crop_lidar_edge_zone(data):
    """
    Crop the edge of zero value areas in LiDAR CHM dataset.

    :param array: 2D numpy array
    :return: Cropped 2D numpy array
    """
    # Define the coordinates of the polygon (example polygon coordinates)
    polygon_coords = [(50, 40), (350, 0), (710, 0), (710, 250), (50, 300)]
    polygon = Polygon(polygon_coords)

    # Access the correct coordinate variables (assuming they are named 'x' and 'y')
    rows, cols = data.shape
    mask = np.zeros(data.shape, dtype=bool)

    # Generate grid of coordinates
    x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))

    # Check if each point is inside the polygon
    for i in range(rows):
        for j in range(cols):
            point = Point(x_coords[i, j], y_coords[i, j])
            if polygon.contains(point):
                mask[i, j] = True

    # Create a new array by extracting only the data in the polygon
    clipped_data = np.full(data.shape, np.nan)
    clipped_data[mask] = data[mask]

    return clipped_data



