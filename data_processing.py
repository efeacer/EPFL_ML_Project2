import pandas as pd
import scipy.sparse as sp
from collections import defaultdict

def load_csv_df(dataset_path):
    """
    Loads a given csv file dataset (with Netflix Prize format) as a
    Pandas DataFrame.
    Args:
        dataset_path: The file path for the csv dataset
    Returns:
        df: Pandas DataFrame corresponding to the dataset
    """
    df = pd.read_csv(dataset_path)
    df['User'] = df['Id'].apply(lambda id: int(id.split('_')[0][1:]))
    df['Item'] = df['Id'].apply(lambda id: int(id.split('_')[1][1:]))
    df['Rating'] = df['Prediction'].astype('float')
    df = df.drop(columns=['Id', 'Prediction'])
    return df

def load_csv_sp(dataset_path):
    """
    Loads a given csv file dataset (with Netflix Prize format) as a
    SciPy lil_matrix.
    Args:
        dataset_path: The file path for the csv dataset
    Returns:
        data: SciPy lil_matrix corresponding to the dataset
    """
    with open(dataset_path) as file:
        lines = file.read().splitlines()
    lines = lines[1:] # discard the header line
    def process_line(line):
        position, rating = line.split(',')
        row, col = position.split('_')
        row = row[1:] # discard the beginning letter, 'r'
        col = col[1:] #discard the beginning letter, 'c'
        return int(row), int(col), float(rating)
    data_lines = [process_line(line) for line in lines] # process each line
    max_row = max(data_line[0] for data_line in data_lines)
    max_col = max(data_line[1] for data_line in data_lines)
    data = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data_lines:
        data[row - 1, col - 1] = rating
    return data

def load_csv_dict(dataset_path):
    """
    Loads a given csv file dataset (with Netflix Prize format) as
    two dictionaries of dictionaries, one with user keys and the other
    with item keys.
    Args:
        dataset_path: The file path for the csv dataset
    Returns:
        dict_user_key: User keyed dictionary for the dataset
        dict_item_key: Item keyed dictionary for the dataset
    """
    with open(dataset_path) as file:
        lines = file.read().splitlines()
    lines = lines[1:] # discard the header line
    def process_line(line):
        position, rating = line.split(',')
        row, col = position.split('_')
        row = row[1:] # discard the beginning letter, 'r'
        col = col[1:] #discard the beginning letter, 'c'
        return int(row), int(col), float(rating)
    data_lines = [process_line(line) for line in lines] # process each line
    dict_user_key = dict_item_key = defaultdict(dict)
    for row, col, rating in data_lines:
        dict_user_key[row - 1][col - 1] = rating
        dict_item_key[col - 1][row - 1] = rating
    return dict_user_key, dict_item_key

def create_submission(predictions: pd.DataFrame, filename):
    """
    Creates a csv file from the predictions given as a Pandas DataFrame.
    Args:
        predictions: Predictions of a model in Pandas DataFrame format
        filename: The filename for created csv
    """
    def prepare_id(row):
        return 'r' + str(int(row['User'])) + '_c' + str(int(row['Item']))
    def prepare_prediction(row):
        rating = row['Rating']
        rating = int(round(rating))
        rating = 5 if rating > 5 else rating
        rating = 1 if rating < 1 else rating
        return rating
    df = pd.DataFrame.copy(predictions)
    df['Id'] = df.apply(prepare_id, axis=1)
    df = df.drop(columns=['User', 'Item'])
    df['Prediction'] = df.apply(prepare_prediction, axis=1)
    df = df.drop(columns=['Rating'])
    df.to_csv(filename, index=False)

def df_to_dict(df):
    """
    Builds and returns two dictionaries of dictionaries, one with user 
    keys and the other with item keys using a Pandas Data Frame
    representing the dataset.
    Args:
        df: Pandas Data Frame representation of the data
    Returns:
        dict_user_key: User keyed dictionary for the dataset
        dict_item_key: Item keyed dictionary for the dataset
    """
    dict_user_key = dict_item_key = defaultdict(dict)
    for _, row in df.iterrows():
        user, item, rating = row['User'], row['Item'], row['Rating']
        dict_user_key[user - 1][item - 1] = rating
        dict_item_key[item - 1][user - 1] = rating
    return dict_user_key, dict_item_key

def df_to_sp(df):
    """ 
    Converts the Pandas Data Frame representation of the data to the
    corresponding SciPy Sparse lil_matrix.
    Args:
        df: Pandas Data Frame representation of the data
    Returns:
        matrix: Corresponding SciPy Sparse lil_matrix
    """
    num_users = df['User'].max()
    num_items = df['Item'].max()
    matrix = sp.lil_matrix((num_users, num_items))
    for row in df.values:
        matrix[row[0] - 1, row[1] - 1] = row[2]
    return matrix

def sp_to_df(matrix):
    """ 
    Converts the SciPy Sparse lil_matrix representation of the data to
    the corresponding Pandas Data Frame.
    Args:
        matrix: SciPy Sparse lil_matrix representation of the data
    Returns:
        df: Corresponding Pandas Data Frame
    """
    print('Converting lil_matrix to Data Frame ...')
    rows, cols, ratings = sp.find(matrix)
    rows += 1
    cols += 1
    df = pd.DataFrame({'User': rows, 'Item': cols, 'Rating': ratings})
    df = df[['User', 'Item', 'Rating']].sort_values(['Item', 'User'])
    print('... converted.')
    return df