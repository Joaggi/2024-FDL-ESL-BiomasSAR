
import os 
import numpy  as np 
import cupy as cp 
import h5py 


def calculate_chunks(x, y, z, target_chunk_size=100):
    """
    Calculate chunk sizes for processing a 3D array.

    Args:
        x (int): Size of the first dimension.
        y (int): Size of the second dimension.
        z (int): Size of the third dimension.
        target_chunk_size (int, optional): Target size for each chunk. Defaults to 100.

    Returns:
        tuple: A tuple containing three elements:
            - tuple: Chunk sizes for the x dimension.
            - tuple: Chunk sizes for the y dimension.
            - tuple: Chunk size for the z dimension (always a single-element tuple).
    """
    nx = (x + target_chunk_size - 1) // target_chunk_size
    ny = (y + target_chunk_size - 1) // target_chunk_size
    print(nx, ny)

    xchunk_list, ychunk_list = [], []

    remaining_x, remaining_y = x, y
    for _ in range(nx):
        chunk_size = min(remaining_x, target_chunk_size)
        xchunk_list.append(chunk_size)
        remaining_x -= chunk_size

    for _ in range(ny):
        chunk_size = min(remaining_y, target_chunk_size)
        ychunk_list.append(chunk_size)
        remaining_y -= chunk_size

    return (tuple(xchunk_list), tuple(ychunk_list), (z,))


def compute_and_save_chunk(result, ground_truth, chunk_index, output_dir):
    """
    Compute a single chunk of the result and corresponding ground truth, and save them to an HDF5 file.

    Args:
        result (dask.array.Array): The full dask array result.
        ground_truth (dask.array.Array): The full dask array of ground truth.
        chunk_index (tuple): The index of the chunk to compute and save.
        output_dir (str): The directory where the chunk will be saved.

    Returns:
        str: The filepath of the saved chunk.
    """
    # Get and compute the specific partition (chunk) for result
    result_chunk = result.partitions[chunk_index].compute()
    
    # Get and compute the corresponding ground truth chunk
    # Note: ground truth might have fewer dimensions, so we use only the relevant indices
    gt_chunk_index = chunk_index[:ground_truth.ndim]
    gt_chunk = ground_truth.partitions[gt_chunk_index].compute()
    
    # Convert CuPy arrays to NumPy if necessary
    if isinstance(result_chunk, cp.ndarray):
        result_chunk = result_chunk.get()
    if isinstance(gt_chunk, cp.ndarray):
        gt_chunk = gt_chunk.get()
    
    # Create a filename and filepath for this chunk
    filename = f"chunk_{'_'.join(map(str, chunk_index))}.h5"
    filepath = os.path.join(output_dir, filename)
    
    # Save the chunks to an HDF5 file
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('data', data=result_chunk)
        f.create_dataset('ground_truth', data=gt_chunk)
    
    return filepath



def compute_and_save_all_chunks(result, ground_truth, output_dir):
    """
    Compute and save all chunks of the result and corresponding ground truth to separate HDF5 files.

    Args:
        result (dask.array.Array): The full dask array result.
        ground_truth (dask.array.Array): The full dask array of ground truth.
        output_dir (str): The directory where chunks will be saved.

    Returns:
        list: A list of filepaths for all saved chunks.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the number of chunks in each dimension of the result
    chunk_shape = tuple(len(c) for c in result.chunks)
    
    # Compute and save all chunks
    all_chunk_files = [
        compute_and_save_chunk(result, ground_truth, chunk_index, output_dir)
        for chunk_index in np.ndindex(chunk_shape)
    ]
    
    return all_chunk_files
