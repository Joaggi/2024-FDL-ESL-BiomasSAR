import pandas as pd 

def get_x_y_from_coordinate_array(x, y, coordinate_array=None):
    """
    function to extract the coordinates from the coordinate_array if defined

    Parameters:
    - x: current x coordinate
    - y: current y coordinate
    - coordinate_array: original coordinate_array

    Returns: 
    - x_coordinate
    - y_coordinate
    """

    x_coordinate = x
    y_coordinate = y
    if type(coordinate_array) != type(None):
        x_coordinate, y_coordinate = [int(x) for x in coordinate_array[x, y].split(" ")]
    return x_coordinate, y_coordinate



def lidar_cube_to_table(lidar_cube, coordinate_array=None):
    """
    Converts a LiDAR cube into a table with columns x, y, and lidar.

    Parameters:
    - lidar_cube: np.array of shape (x_dim, y_dim) where dimensions are (x, y, lidar)

    Returns:
    - pd.DataFrame with columns [x, y, lidar]
    """
    x_dim, y_dim = lidar_cube.shape
    records = []

    for x in range(x_dim):
        for y in range(y_dim):
            lidar_value = lidar_cube[x, y]
            x_coordinate, y_coordinate = get_x_y_from_coordinate_array(x, y, coordinate_array=coordinate_array)
            records.append([x_coordinate, y_coordinate, lidar_value])

    # Create DataFrame
    df = pd.DataFrame(records, columns=['x', 'y', 'lidar'])
    return df


def classification_tree_image_to_table(classification_tree_image, keep_class_0=False, coordinate_array=None):
    """
    Converts a classification tree image into a table with columns x, y, and type of tree.

    Parameters:
    - classification_image: np.array of shape (x_dim, y_dim) where dimensions are (x, y)
    - keep_class_0: class 0 is not a real class is only places where we do not have any class type.

    Returns:
    - pd.DataFrame with columns [x, y, class]
    """
    x_dim, y_dim = classification_tree_image.shape
    records = []

    for x in range(x_dim):
        for y in range(y_dim):
            lidar_value = classification_tree_image[x, y]
            x_coordinate, y_coordinate = get_x_y_from_coordinate_array(x, y, coordinate_array=coordinate_array)
            if(not(keep_class_0==False and lidar_value==0)):
                #print(keep_class_0, lidar_value)
                records.append([x_coordinate, y_coordinate, lidar_value])

    # Create DataFrame
    df = pd.DataFrame(records, columns=['x', 'y', 'class'])
    return df



def extract_z_vectors(data, band, coordinate_array = None):
    """
    Extracts z vectors for each (x, y) point in the given data and stores them in a table with additional information.

    Parameters:
    - data: np.array of shape (326, 840, 36, 3) where dimensions are (x, y, z, polarimetry)
    - coordinate_array: coordinate array with the size of the data

    Returns:
    - pd.DataFrame with columns [x, y, z_0, z_1, ..., z_n, polarimetry] where z_n is the max value of z.
    """
    x_dim, y_dim, z_dim, pol_dim = data.shape
    records = []

    for polarimetry in range(pol_dim):
        for x in range(x_dim):
            for y in range(y_dim):
                z_vector = data[x, y, :, polarimetry]
                x_coordinate, y_coordinate = get_x_y_from_coordinate_array(x, y, coordinate_array=coordinate_array)
                record = [x_coordinate, y_coordinate] + z_vector.tolist() + [polarimetry]
                records.append(record)

    # Generate column names: x, y, z_P_0, z_P_1, ..., z_P_n, polarimetry
    columns = ['x', 'y'] + [f'z_{band}_{i}' for i in range(z_dim)] + ['polarimetry']
    
    df = pd.DataFrame(records, columns=columns)
    return df

def generate_flattened_intensity_and_lidar(PcubeDeRampedUTM, band_opt, lidar_df, filename=None, coordinate_array=None): #apply this to quadrant (need to pass the coordinates) lidar_cube_to_table
    """ Create the tabular data with lidar and intensity, with seperate entrys for each polarimetry 
    Inputs:
    PcubeDeRampedUTM: loaded from load_biomass_data
    lidar_df: the flattedn lidar cube from lidar_cube_to_table
    class_df: the flattedn classification tree image from classification_tree_image_to_table
    coordinate_array: the coordinate array for the original image

    Outputs: cleaned ml ready tabular data 

    """
    if band_opt == 'P':
        df = extract_z_vectors(PcubeDeRampedUTM, "P", coordinate_array)
    elif band_opt == 'L_Mono':
        df = extract_z_vectors(PcubeDeRampedUTM, "L_Mono", coordinate_array)
    elif band_opt == 'L_Bi':
        df = extract_z_vectors(PcubeDeRampedUTM, "L_Bi", coordinate_array)
    
    # Identify the columns corresponding to the z values
    z_columns = [col for col in df.columns if col.startswith('z_')]

    # Filter out rows where all z values are NaN
    df_filtered = df.loc[~(df[z_columns].isna()).all(axis=1)]

    # Filter out rows where the value is less than -300dB
    threshold = -300
    df_filtered = df_filtered[(df_filtered > threshold).all(axis=1)]

    # merge filtered flattened data with the lidar data , remove any lidar nulls
    mldata = df_filtered.merge(lidar_df, on=['x', 'y'])
    
    cl_ml_data = mldata[~mldata['lidar'].isna()]

    if filename is not None:
        cl_ml_data.to_csv( filename , index=False )
    
    return cl_ml_data

def generate_flattened_intensity_and_lidar_PL(PcubeDeRampedUTM, PcubeDeRampedDeBiasedUTM_Mono, PcubeDeRampedDeBiasedUTM_Bi, lidar_df, filename=None):
    """ Create the tabular data with lidar and intensity, with separate entries for each polarimetry 
    Inputs:
    PcubeDeRampedUTM: loaded from load_biomass_data
    lidar_df: the flatten lidar cube from lidar_cube_to_table
    class_df: the flatten classification tree image from classification_tree_image_to_table

    Outputs: cleaned ml ready tabular data 

    """
    print('P band shape:', PcubeDeRampedUTM.shape)
    print('L band Monostatic shape:', PcubeDeRampedDeBiasedUTM_Mono.shape)
    print('L band Bistatic shape:', PcubeDeRampedDeBiasedUTM_Bi.shape)
    df_P = extract_z_vectors(PcubeDeRampedUTM, "P")
    df_L_Mono = extract_z_vectors(PcubeDeRampedDeBiasedUTM_Mono, "L_Mono")
    df_L_Bi = extract_z_vectors(PcubeDeRampedDeBiasedUTM_Bi, "L_Bi")
    df = pd.merge(df_P, df_L_Mono, on=['x', 'y', 'polarimetry'], how='inner')
    df = pd.merge(df, df_L_Bi, on=['x', 'y', 'polarimetry'], how='inner')
    print('P-band df:', df_P.shape)        
    print('L-band Mono df:', df_L_Mono.shape)   
    print('L-band Bi df:', df_L_Bi.shape) 
    print('Merged df', df.shape) 
    
    # Identify the columns corresponding to the z values
    z_columns = [col for col in df.columns if col.startswith('z_')]

    # Filter out rows where any z value is NaN
    df_filtered = df.loc[~(df[z_columns].isna().any(axis=1))]
    
    # Filter out rows where the value is less than -300dB
    threshold = -300
    df_filtered = df_filtered[(df_filtered > threshold).all(axis=1)]

    # merge filtered flattened data with the lidar data , remove any lidar nulls
    mldata = df_filtered.merge(lidar_df, on=['x', 'y'])
    
    cl_ml_data = mldata[~mldata['lidar'].isna()]

    if filename is not None:
        cl_ml_data.to_csv( filename , index=False )
    
    return cl_ml_data


def generate_flattened_intensity_and_class(PcubeDeRampedUTM, band_opt, class_df, filename=None):
    """ Create the tabular data with tree class and intensity, with seperate entrys for each polarimetry 
    Inputs:
    PcubeDeRampedUTM: loaded from load_biomass_data
    class_df: the flattedn classification tree image from classification_tree_image_to_table

    Outputs: cleaned ml ready tabular data 

    """
    if band_opt == 'P':
        df = extract_z_vectors(PcubeDeRampedUTM, "P")
    elif band_opt == 'L_Mono':
        df = extract_z_vectors(PcubeDeRampedUTM, "L_Mono")
    elif band_opt == 'L_Bi':
        df = extract_z_vectors(PcubeDeRampedUTM, "L_Bi")
    
    # Identify the columns corresponding to the z values
    z_columns = [col for col in df.columns if col.startswith('z_')]

    # Filter out rows where all z values are 0
    df_filtered = df.loc[~(df[z_columns].isna().any(axis=1))]

    # Filter out rows where the value is less than -300dB
    threshold = -300
    df_filtered = df_filtered[(df_filtered > threshold).all(axis=1)]

    # merge filtered flattened data with the class data , remove any class nulls
    mldata = df_filtered.merge(class_df, on=['x', 'y'])
    
    cl_ml_data = mldata[~mldata["class"].isna()]

    if filename is not None:
        cl_ml_data.to_csv( filename , index=False )
    
    return cl_ml_data

def generate_flattened_intensity_and_class_PL(PcubeDeRampedUTM, PcubeDeRampedDeBiasedUTM_Mono, PcubeDeRampedDeBiasedUTM_Bi, class_df, filename=None):
    """ Create the tabular data with tree class and intensity, with separate entries for each polarimetry 
    Inputs:
    PcubeDeRampedUTM: loaded from load_biomass_data
    class_df: the flatten classification tree image from classification_tree_image_to_table

    Outputs: cleaned ml ready tabular data 

    """
    df_P = extract_z_vectors(PcubeDeRampedUTM, "P")
    df_L_Mono = extract_z_vectors(PcubeDeRampedDeBiasedUTM_Mono, "L_Mono")
    df_L_Bi = extract_z_vectors(PcubeDeRampedDeBiasedUTM_Bi, "L_Bi")
    df = pd.merge(df_P, df_L_Mono, on=['x', 'y', 'polarimetry'], how='inner')
    df = pd.merge(df, df_L_Bi, on=['x', 'y', 'polarimetry'], how='inner')
    print('P-band df:', df_P.shape)        
    print('L-band Mono df:', df_L_Mono.shape)   
    print('L-band Bi df:', df_L_Bi.shape) 
    print('Merged df', df.shape) 

    # Identify the columns corresponding to the z values
    z_columns = [col for col in df.columns if col.startswith('z_')]

    # Filter out rows where all z values are 0
    df_filtered = df.loc[~(df[z_columns].isna().any(axis=1))]

    # Filter out rows where the value is less than -300dB
    threshold = -300
    df_filtered = df_filtered[(df_filtered > threshold).all(axis=1)]

    # merge filtered flattened data with the class data , remove any class nulls
    mldata = df_filtered.merge(class_df, on=['x', 'y'])
    
    cl_ml_data = mldata[~mldata["class"].isna()]

    if filename is not None:
        cl_ml_data.to_csv( filename , index=False )
    
    return cl_ml_data

def convert_mldata_union_polarimetry(df, label, filename=None):
    """ 
    Union the three polarimetries into one entry per x,y 
    Inputs: 
    df: the output of the flattening of the cube , generate_flattened_intensity_and_lidar
    label: lidar for regression task or class for classification task
    """
    # 1. Separate the z columns
    z_cols = [col for col in df.columns if col.startswith('z_')]
    
    # 2. Melt the dataframe to long format for z values
    df_melted = df.melt(id_vars=['x', 'y', 'polarimetry', label], 
                        value_vars=z_cols, 
                        var_name='z_index', 
                        value_name='z_value')
        
    # 3. Create a new column combining polarimetry and z_index
    df_melted['pol_z'] = 'pol_' + df_melted['polarimetry'].astype(str) + '_' + df_melted['z_index']
    
    # 4. Pivot the dataframe to create separate columns for each polarimetry-z combination
    df_pivot = df_melted.pivot_table(index=['x', 'y', label], 
                                     columns='pol_z', 
                                     values='z_value', 
                                     aggfunc='first')
    
    # 5. Reset the index to turn x, y, and label back into columns
    df_pivot.reset_index(inplace=True)
    
    # 6. Flatten the column names
    df_pivot.columns.name = None

    # print(df_pivot.duplicated(subset=['x', 'y']).sum())

    if filename is not None:
        df_pivot.to_csv(filename, index=False)
    
    return df_pivot