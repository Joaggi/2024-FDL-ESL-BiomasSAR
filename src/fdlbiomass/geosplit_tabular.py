
import numpy as np 
from sklearn.model_selection import train_test_split
import os 


def gen_square_swath(ml_data, ml_data_name, outpath, label):
    
    # create swath 
    print('Starting swath train split')
    _ = generate_train_test_geo(ml_data, ml_data_name, create_diagonal_swathes, split_by_swath, 0.2, 'swath',  x_column = 'x', y_column = 'y', test_size = 0.2, outputpath=outpath, label=label)

    # create sq 
    print('Starting square train split')
    _ = generate_train_test_geo(ml_data, ml_data_name, create_squares, split_by_squares, 0.05, 'square',  x_column = 'x', y_column = 'y', test_size = 0.2, outputpath=outpath, label=label)

    print('Complete')


def generate_train_test_geo(ml_data, ml_data_name, create_geo_fn, split_fn, thickness, geo_name,  x_column = 'x', y_column = 'y', test_size = 0.2, outputpath=None, label=None):
    """ generic fn to call either geo split method on tabular data 
    
    Inputs: 
    
    mldata: the flattened data, with seperate polarimetry 
    ml_data_name: unionpol or seperatepol 
    create_geo_fn: the fn to create the geo splits based on xy cols 
    split_fn: fn to split the geo data 
    thickness: thickenss of the geo split e.g. 0.2 for swath, 0.05 for squares 
    geo_name: square or swath 
    x_column: name of x col 
    y_column: name of y col 
    test_size: size of test split 
    outputpath: if not none, save to this folder 

    """
    coords_data = ml_data[[x_column, y_column]].drop_duplicates()

    # Create diagonal swathes
    df = create_geo_fn(coords_data, x_column, y_column, thickness)
    train_data, test_data = split_fn(df, test_size=test_size)

    train_data = train_data.merge(ml_data, on =[x_column, y_column], how='inner')
    test_data = test_data.merge(ml_data, on =[x_column, y_column], how='inner')

    if train_data.merge(test_data, on=[x_column, y_column], how='inner').empty is True: 
        print('No overlap train and test- passed :)  ')
    else:
        raise ValueError('Overlap in x and y between train and test ')

    if outputpath is not None:
        train_data.to_csv(os.path.join(outputpath, f'{geo_name}_{label}_mldata_{ml_data_name}__train_data__thick_{thickness}__testsize_{test_size}.csv') ) 
        test_data.to_csv(os.path.join(outputpath, f'{geo_name}_{label}_mldata_{ml_data_name}__test_data__thick_{thickness}__testsize_{test_size}.csv'))

    print('Train test shapes: ', train_data.shape , test_data.shape)
    return train_data, test_data 


def create_diagonal_swathes(data, x_col, y_col, swath_thickness):
    """
    Create swath labels for the data based on diagonal swathes.
    
    Parameters:
    - data: DataFrame with geographic coordinates.
    - x_col: Column name for x coordinates.
    - y_col: Column name for y coordinates.
    - swath_thickness: Thickness of each swath as a proportion of the x range.
    
    Returns:
    - DataFrame with an additional column 'swath' indicating the swath of each point.
    """
    x_min, x_max = data[x_col].min(), data[x_col].max()
    y_min, y_max = data[y_col].min(), data[y_col].max()
    x_range = x_max - x_min
    
    # Calculate the swath boundaries
    num_swathes = int(np.ceil(1 / swath_thickness))
    data['swath'] = ((data[x_col] + data[y_col] - x_min - y_min) / (x_range * swath_thickness)).astype(int) % num_swathes
    
    return data

def split_by_swath(data, test_size=0.2, max_iterations=100):
    """
    Split the data into train and test sets based on swathes, aiming for the specified test_size.
    
    Parameters:
    - data: DataFrame with geographic coordinates.
    - swath_col: Column name indicating the swath of each point.
    - test_size: Proportion of the dataset to include in the test split.
    - max_iterations: Maximum number of iterations to attempt to achieve the desired split.
    
    Returns:
    - train_data: Training set.
    - test_data: Test set.
    """
    swath_col ='swath'
    unique_swathes = data[swath_col].unique()
    n_swathes = len(unique_swathes)
    
    best_split = None
    best_diff = float('inf')
    
    for _ in range(max_iterations):
        n_test_swathes = max(1, int(n_swathes * test_size))
        test_swathes = np.random.choice(unique_swathes, size=n_test_swathes, replace=False)
        
        test_data = data[data[swath_col].isin(test_swathes)]
        train_data = data[~data[swath_col].isin(test_swathes)]
        
        actual_test_size = len(test_data) / len(data)
        diff = abs(actual_test_size - test_size)
        
        if diff < best_diff:
            best_diff = diff
            best_split = (train_data, test_data)
        
        if diff < 0.01:  # If within 1% of desired test_size, break
            break
    
    return best_split



def create_squares(data, x_col, y_col, square_size):
    x_min, x_max = data[x_col].min(), data[x_col].max()
    y_min, y_max = data[y_col].min(), data[y_col].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # # Calculate the square boundaries
    # num_squares_x = int(np.ceil(1 / square_size))
    # num_squares_y = int(np.ceil(y_range / (y_range * square_size)))  # Changed x_range to y_range
    
    data['x_square'] = ((data[x_col] - x_min) / (x_range * square_size)).astype(int)
    data['y_square'] = ((data[y_col] - y_min) / (y_range * square_size)).astype(int)  # Changed x_range to y_range
    
    return data


def split_by_squares(data, test_size=0.2, max_iterations=100):
    """
    Split the data into train and test sets based on squares, aiming for the specified test_size.
    
    Parameters:
    - data: DataFrame with 'x_square' and 'y_square' columns.
    - test_size: Proportion of the dataset to include in the test split.
    - max_iterations: Maximum number of iterations to attempt to achieve the desired split.
    
    Returns:
    - train_data: Training set.
    - test_data: Test set.
    """
    unique_squares = data[['x_square', 'y_square']].drop_duplicates()
    n_squares = len(unique_squares)
    
    best_split = None
    best_diff = float('inf')
    
    for _ in range(max_iterations):
        n_test_squares = max(1, int(n_squares * test_size))
        test_square_indices = np.random.choice(n_squares, size=n_test_squares, replace=False)
        test_squares = unique_squares.iloc[test_square_indices]
        
        test_mask = data.apply(lambda row: any((row['x_square'] == sq[0] and row['y_square'] == sq[1]) for sq in test_squares.values), axis=1)
        
        test_data = data[test_mask]
        train_data = data[~test_mask]
        
        actual_test_size = len(test_data) / len(data)
        diff = abs(actual_test_size - test_size)
        
        if diff < best_diff:
            best_diff = diff
            best_split = (train_data, test_data)
        
        if diff < 0.01:  # If within 1% of desired test_size, break
            break
    
    return best_split

