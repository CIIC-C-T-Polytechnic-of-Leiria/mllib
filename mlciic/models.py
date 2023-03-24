import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlciic import functions

def beforeModel(df: pd.DataFrame, drop_columns: list,label: str, one_hot_encoder = True, label_encoder = True, valid_indices = False, test_size = 0.2, standard_scaler = True, smote = True, random_state = 1):
    """
    Function to preprocess data before model training.

    Parameters:
    df (pandas.DataFrame): Input dataframe.
    drop_columns (list): List of column names to drop.
    label (str): Label column name.
    one_hot_encoder (bool): Whether to use one-hot encoding. Defaults to True.
    label_encoder (bool): Whether to use label encoding. Defaults to True.
    valid_indices (bool): Whether to use validation set. Defaults to False.
    test_size (float): Size of the test set. Defaults to 0.2.
    standard_scaler (bool): Whether to use standard scaler. Defaults to True.
    smote (bool): Whether to use SMOTE for oversampling. Defaults to True.
    random_state (int): Random state for reproducibility. Defaults to 1.

    Returns:
    X_train (numpy.ndarray): Training features.
    y_train (numpy.ndarray): Training labels.
    X_test (numpy.ndarray): Test features.
    y_test (numpy.ndarray): Test labels.
    X_valid (numpy.ndarray): Validation features if valid_indices=True.
    y_valid (numpy.ndarray): Validation labels if valid_indices=True.
    le (LabelEncoder): Label encoder object.
    """
    
    # Check input types
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(drop_columns, list):
        raise TypeError("Input 'drop_columns' must be a list of column names to drop.")
    
    functions.display_information_dataframe(df,showCategoricals = True, showDetailsOnCategorical = True, showFullDetails = True)
    
    df.drop(drop_columns, axis=1, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    df.drop_duplicates(subset=None, keep="first", inplace=True)
    df = shuffle(df)
    
    categorical_columns = []
    for col in df.columns[df.dtypes == object]:
        if col != label:
            categorical_columns.append(col)
    features = [ col for col in df.columns if col not in [label]]
    
    if one_hot_encoder:
        df, colunas_one_hot = functions.categorical_get_dummies(df, categorical_columns)
        features = [ col for col in df.columns if col not in [label]] 
    
    if label_encoder:
        df, le = functions.encode_labels(df, label)
    
    n_total = len(df)
    train_val_indices, test_indices = train_test_split(range(n_total), test_size=test_size, random_state=random_state)

    if valid_indices:
        train_indices, valid_indices = train_test_split(train_val_indices, test_size=0.25, random_state=random_state) # 0.25 x 0.8 = 0.2
        X_valid = df[features].values[valid_indices]
        y_valid = df[label].values[valid_indices]
        
        X_train = df[features].values[train_indices]
        y_train = df[label].values[train_indices]
    else:
        X_train = df[features].values[train_val_indices]
        y_train = df[label].values[train_val_indices]
        

    X_test = df[features].values[test_indices]
    y_test = df[label].values[test_indices]
    
    if standard_scaler:
        standScaler = StandardScaler()
        model_norm = standScaler.fit(X_train)

        X_train = model_norm.transform(X_train)
        X_test = model_norm.transform(X_test)
        if valid_indices:
            X_valid = model_norm.transform(X_valid)

    if smote:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state,n_jobs=-1)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    
    results = (X_train, y_train, X_test, y_test)
    if 'X_valid' in locals():
        results += (X_valid, y_valid)
    if 'le' in locals():
        results += (le,)

    return results
    