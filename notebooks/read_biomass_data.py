import numpy as np

def read_biomass_data(full_path, crop_struct, data_type):
    """
    Reads files produced by write_biomass_data.m.

    Parameters:
    full_path (str): Path to the file to be read.
    crop_struct (dict): Dictionary with the following keys:
        S: number of samples (dia only)
        L: number of lines (dia only)
        Sin: first sample to be read (starting from 1)
        Sfin: last sample to be read
        Lin: first line to be read (starting from 1)
        Lfin: last line to be read
    data_type (int): Type of data to read (1 for Complex, 2 for Real).

    Returns:
    np.ndarray: The read data as a NumPy array.
    """
    # Single
    N = 4

    try:
        fid = open(full_path, 'rb')  # Open the file in binary read mode
    except IOError:
        print('Error in read_biomass_data: File cannot be opened.')
        return None

    # Read the total number of rows and columns
    NrTot = np.fromfile(fid, dtype=np.float32, count=1)[0]
    NcTot = np.fromfile(fid, dtype=np.float32, count=1)[0]

    # If crop_struct is not provided, use the entire data
    if not crop_struct:
        crop_struct = {
            'S': NrTot,
            'Sin': 1,
            'Sfin': NrTot,
            'L': NcTot,
            'Lin': 1,
            'Lfin': NcTot
        }

    # Check for mismatches between provided crop_struct and actual dimensions
    if 'S' in crop_struct and 'L' in crop_struct:
        if NrTot != crop_struct['S'] or NcTot != crop_struct['L']:
            print('Warning in read_biomass_data:')
            print(f'Nr mismatch: {NrTot} vs {crop_struct["S"]}')
            print(f'Nc mismatch: {NcTot} vs {crop_struct["L"]}')

    # Calculate the number of rows and columns to read
    Nr = crop_struct['Sfin'] - crop_struct['Sin'] + 1
    Nc = crop_struct['Lfin'] - crop_struct['Lin'] + 1

    # Read complex data
    if data_type == 1:
        # Move file pointer to the start of the relevant data block
        fid.seek(int(2 * N + ((crop_struct['Lin'] - 1) * NrTot + crop_struct['Sin'] - 1) * N * 2), 0)
        
        if crop_struct['Sin'] == 1 and crop_struct['Sfin'] == NrTot:
            # Read the entire block of complex data
            I = np.fromfile(fid, dtype=np.float32, count=2 * Nr * Nc).reshape(2 * Nr, Nc)
        else:
            # Read the cropped block of complex data
            bytes_to_skip = int((NrTot - crop_struct['Sfin'] + crop_struct['Sin'] - 1) * N * 2)
            I = np.zeros((2 * Nr, Nc), dtype=np.float32)
            for i in range(Nc):
                I[:, i] = np.fromfile(fid, dtype=np.float32, count=2 * Nr)
                fid.seek(bytes_to_skip, 1)
        
        # Combine real and imaginary parts into complex numbers
        I = I[0::2, :] + 1j * I[1::2, :]
        fid.close()
    
    # Read real data
    elif data_type == 2:
        # Move file pointer to the start of the relevant data block
        fid.seek(int(2 * N + ((crop_struct['Lin'] - 1) * NrTot + crop_struct['Sin'] - 1) * N), 0)
        
        if crop_struct['Sin'] == 1 and crop_struct['Sfin'] == NrTot:
            # Read the entire block of real data
            I = np.fromfile(fid, dtype=np.float32, count=Nr * Nc).reshape(Nr, Nc)
        else:
            # Read the cropped block of real data
            bytes_to_skip = int((NrTot - crop_struct['Sfin'] + crop_struct['Sin'] - 1) * N)
            I = np.zeros((Nr, Nc), dtype=np.float32)
            for i in range(Nc):
                I[:, i] = np.fromfile(fid, dtype=np.float32, count=Nr)
                fid.seek(bytes_to_skip, 1)
        
        fid.close()
    
    else:
        print('Error in read_biomass_data: Unrecognized data type.')
        return None

    return I


"""
Explanation:
File Handling: The file is opened in binary read mode ('rb'), and a check is added to handle file opening errors.
Reading Dimensions: The total number of rows and columns (NrTot and NcTot) are read from the file.
Default Crop Struct: If crop_struct is not provided, the function sets it to read the entire data.
Dimension Mismatch Warning: A warning is printed if there is a mismatch between the provided crop_struct and the actual dimensions in the file.
Data Reading:
For complex data (data_type == 1), the function reads pairs of real and imaginary parts and combines them.
For real data (data_type == 2), the function reads the real values directly.
Seek and Skip: The function uses seek to position the file pointer correctly and skips bytes as needed to read the cropped data correctly.
Error Handling: Basic error handling is added for unrecognized data types.
"""