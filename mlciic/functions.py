import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
import tracemalloc
import time
import pickle
import os

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#corrigir o erro '-'
def display_information_dataframe(df,showCategoricals = False, showDetailsOnCategorical = False, showFullDetails = False):
    """
    Displays information about the input pandas DataFrame `df`, including the data type, column name, and unique values
    for each column. Optionally, the function can also display information about categorical columns and their one-hot encoded
    representations.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame to display information for.

    showCategoricals : bool, optional (default=False)
        If True, displays a list of the names of all categorical columns in `df`.

    showDetailsOnCategorical : bool, optional (default=False)
        If True, displays the unique values for each categorical column in `df`.
        If showFullDetails=True, displays the entire list of unique values for each categorical column.

    showFullDetails : bool, optional (default=False)
        If True, displays the entire list of unique values for each categorical column, rather than just the first few.

    Returns:
    --------
    None
    """
    
    print(f"---\nLines: {len(df.index)}\nColumns: {len(df.columns)} \nMissing value or NaN: {df.isnull().sum().sum()}\n---")
    
    # Create an empty summary DataFrame with columns for data type, column name, and unique values.
    summary_df = pd.DataFrame(columns=['Data Type', 'Column Name', 'Unique Values'])
    
    # Iterate through the columns of the original dataframe.
    for col in df.columns:
        # Get the data type of the column.
        dtype = df[col].dtype
        # Get the column name.
        col_name = col
        # Get the unique values of the column.
        unique_values = df[col].unique()
        # Append a new row to the summary dataframe.
        summary_df = summary_df.append({'Data Type': dtype, 'Column Name': col_name, 'Unique Values': unique_values}, ignore_index=True)
    
    # Set display options to show all rows and columns of the summary DataFrame.
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    
    # Identify any columns in `df` with a data type of `object` (i.e., categorical columns) and store their names in a list.
    categorical_columns = []
    for col in df.columns[df.dtypes == object]:
        #if col != "Attack_type":
        categorical_columns.append(col)
    
    # If `showCategoricals` is True, display a list of the names of all categorical columns in `df`.
    if showCategoricals:
        print("Categorical columns: ")
        print(categorical_columns)
        print("")
    
    # If `showDetailsOnCategorical` is True, display the unique values and one-hot encoded representations for each categorical column in `df`.
    if showDetailsOnCategorical:
        print("--- Details for categorical columns ---")
        colunas_one_hot = {}
        for coluna in categorical_columns:
            codes, uniques = pd.factorize(df[coluna].unique())
            colunas_one_hot[coluna] = {"uniques": uniques, "codes":codes}
            if showFullDetails:
                with np.printoptions(threshold=np.inf):
                    print(coluna + ": ")
                    print(colunas_one_hot[coluna]["uniques"])
                    print("")
            else:
                
                print(coluna + ": ")
                print(colunas_one_hot[coluna]["uniques"])
                print("")

    # Display the summary DataFrame.
    try:
        return display(summary_df)
    except:
        print(summary_df)
    

def calculate_metrics(modelName, yTrue, yPred, average='binary'):
    """
    Calculate and print the performance metrics of a classification model.
    
    Parameters:
    modelName (str): The name of the classification model.
    yTrue (array-like): The true labels.
    yPred (array-like): The predicted labels.
    average (str or None, optional): The averaging method to use for multi-class classification. One of 
        {'micro', 'macro', 'weighted', 'binary'} or None (default: 'binary'). If None, only binary 
        classification metrics will be computed.
    
    Raises:
    ValueError: If `average` is not one of {'micro', 'macro', 'weighted', 'binary'} or None.
    
    """
    # Check if average parameter is valid
    if average != 'micro' and average != 'macro' and average != 'weighted' and average != 'binary' and average != None:
        print("Average must be one of this options: {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} or None, default=’binary’")
        return
    
    # Prints the name of the model and calculate accuracy and precision
    print(f"--- Performance of {modelName} ---")
    acc = accuracy_score(y_true = yTrue, y_pred = yPred)
    precision = precision_score(y_true = yTrue, y_pred = yPred, average = average)
    print(f'Accuracy : {np.round(acc*100,2)}%\nPrecision: {np.round(precision*100,2)}%')
    
    # Calculates and print recall and F1-score
    f1 = f1_score(y_true = yTrue, y_pred = yPred, average = average)
    recall = recall_score(y_true = yTrue, y_pred = yPred, average = average)
    print(f'Recall: {np.round(recall*100,2)}%\nF1-score: {np.round(f1*100,2)}%')
    
    #auc_sklearn = roc_auc_score(y_true = yTrue, y_score = yPred, average = average)
    #print(f'Roc auc: {np.round(auc_sklearn*100,2)}%')
    
    # Calculates and prints balanced accuracy and classification report
    print(f"Balanced accuracy: {np.round(balanced_accuracy_score(yTrue, yPred)*100,2)}%")
    print(f"Classification report:\n{classification_report(yTrue, yPred)}")
    
# -----------------------------------------------------------------------------
# def start_measures():
#     tracemalloc.start()
#     start_time = time.time()
#     return start_time

# def stop_measures(start_time):
#     print("(current, peak)",tracemalloc.get_traced_memory())
#     tracemalloc.stop()
#     print("--- %s segundos ---" % (time.time() - start_time))

# -----------------------------------------------------------------------------

def start_measures():
    """
    Starts the memory and time measures
    
    Returns:
    --------
    start_time: float
        The start time of the measurements
    tracemalloc_obj: tracemalloc.TraceMalloc
        The `tracemalloc` object that is used for measuring memory usage
    """
    try:
        tracemalloc_obj = tracemalloc.start()
        start_time = time.time()
    except Exception as e:
        print(f"An error occurred while starting memory and time measures: {e}")
        return None, None
    
    return start_time, tracemalloc_obj

def stop_measures(start_time, tracemalloc_obj):
    """
    Stops the memory and time measures
    
    Parameters:
    -----------
    start_time: float
        The start time of the measurements
    tracemalloc_obj: tracemalloc.TraceMalloc
        The `tracemalloc` object that is used for measuring memory usage
    
    Returns:
    --------
    memory_usage: tuple(int, int)
        A tuple containing the current and peak memory usage in bytes
    elapsed_time: float
        The elapsed time in seconds
    """
    try:
        memory_usage = tracemalloc_obj.get_traced_memory()
        tracemalloc_obj.stop()
        elapsed_time = time.time() - start_time
    except Exception as e:
        print(f"An error occurred while stopping memory and time measures: {e}")
        return None, None
    
    print("(current, peak)", memory_usage)
    print("--- {:.2f} segundos ---".format(elapsed_time))
    return memory_usage, elapsed_time


def __corrfunc(x, y, **kws):
    """
    Calculates the Pearson correlation coefficient and p-value between two arrays.
    
    Parameters:
    -----------
    x: array-like
        First array to calculate the correlation coefficient.
    y: array-like
        Second array to calculate the correlation coefficient.
    **kws: keyword arguments
        Additional arguments to pass to the matplotlib `annotate` method.
        
    Returns:
    --------
    None
        The function adds an annotation to the current axis with the correlation coefficient 
        and a star indicator of the level of significance of the correlation based on the p-value.
    """
    r, p = stats.pearsonr(x, y)
    p_stars = ''
    if p <= 0.05:
        p_stars = '*'
    if p <= 0.01:
        p_stars = '**'
    if p <= 0.001:
        p_stars = '***'
    ax = plt.gca()
    ax.annotate('r = {:.2f} '.format(r) + p_stars, xy=(0.05, 0.9), xycoords=ax.transAxes)

def __annotate_colname(x, **kws):
    """
    Adds an annotation to the current axis with the name of a dataframe column.
    
    Parameters:
    -----------
    x: pandas Series
        A column of a dataframe.
    **kws: keyword arguments
        Additional arguments to pass to the matplotlib `annotate` method.
        
    Returns:
    --------
    None
        The function adds an annotation to the current axis with the name of the column.
    """
    ax = plt.gca()
    ax.annotate(x.name, xy=(0.05, 0.9), xycoords=ax.transAxes,fontweight='bold')

def cor_matrix(df):
    """
    Creates a correlation matrix plot for a given dataframe.
    
    Parameters:
    -----------
    df: pandas DataFrame
        The dataframe to create the correlation matrix plot for.
        
    Returns:
    --------
    seaborn PairGrid object
        The function returns a seaborn PairGrid object containing the correlation matrix plot.
    """
    g = sns.PairGrid(df, palette=['red'])
    # Use normal regplot as `lowess=True` doesn't provide CIs.
    g.map_upper(sns.regplot, scatter_kws={'s':10})
    g.map_diag(sns.distplot)
    g.map_diag(__annotate_colname)
    g.map_lower(sns.kdeplot, cmap='Blues_d')
    g.map_lower(__corrfunc)
    # Remove axis labels, as they're in the diagonals.
    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')
    return g


def save_model(model, model_name, directory='./models', overwrite=False):
    """
    Save a trained machine learning model to disk.´

    Parameters:
    -----------
        model (object): The trained model object.
        model_name (str): The name to use when saving the model.
        directory (str): The directory where the model will be saved. Default is './models'.
        overwrite (bool): Whether to overwrite an existing model with the same name. Default is False.

    Raises:
        ValueError: If overwrite is False and a file with the same name already exists in the directory.

    """

    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Check if file already exists
    filename = os.path.join(directory, model_name + '.sav')
    if os.path.exists(filename) and not overwrite:
        raise ValueError(f"A file with the name '{model_name}.sav' already exists in the '{directory}' directory.")

    # Save the model to disk
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved as '{model_name}.sav' in the '{directory}' directory.")

#--------------------------------------------------------------

# def save_model(model, model_name): 
#     # guarda modelo no disco
#     filename = model_name + '.sav'
#     pickle.dump(model, open(filename, 'wb'))


# def load_model(model_name): 
#     # carrega modelo do disco
#     model = pickle.load(open(model_name + '.sav', 'rb'))    
#     return model

#--------------------------------------------------------------
    
def load_model(model_name):
    """
    Loads a machine learning model from disk.

    Args:
    - model_name (str): The name of the model to be loaded.

    Returns:
    - model: The loaded machine learning model.
    """
    # check if the model file exists
    filename = model_name + '.sav'
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Model file {filename} not found.")

    # load the model from disk
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    return model

# def load_model(model_name): 
#     # carrega modelo do disco
#     model = pickle.load(open(model_name + '.sav', 'rb'))    
#     return model

def heatmap(df: pd.DataFrame, size: int = 40, save_path: str = None): #-> pd.DataFrame:
    """
    Generate a heatmap of the correlation matrix for a given dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    size : int, optional
        The size of the plot in inches (default is 40).
    save_path : str, optional
        The path to save the heatmap image. If not provided, the image is not saved.

    Returns
    -------
    pd.DataFrame
        The correlation matrix for the input dataframe.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas dataframe.")

    corr = df.corr().round(2)

    plt.figure(figsize=(size, size))
    sns.heatmap(corr, annot=True)
    plt.title("Correlation Matrix Heatmap")

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

    return corr


def metrics_graphs(history, model, model_conf="not provided!", metrics=['loss'], y_log=False):
    """
    Plots training and validation metrics of a Keras model during training.
    Parameters:
    -----------
    history: tf.keras.callbacks.History
        Object returned by model.fit() containing training history.
    model: tf.keras.Model
        The trained machine learning model.
    conf_modelo: str, optional (default="")
        The configuration of the model (optional)
    metrics: list of str, optional (default=['loss'])
        The metrics to plot. Must be keys in the history object.
    y_log: bool, optional(default=False) 
        Whether to use logarithmic scale for the y-axis on the loss plot. Defaults to False.
    """

    if not isinstance(history, tf.keras.callbacks.History):
        raise TypeError("The 'history' argument must be a tf.keras.callbacks.History object.")
    if not isinstance(model, tf.keras.Model):
        raise TypeError("The 'model' argument must be a tf.keras.Model object.")
    if not isinstance(model_conf, str):
        raise TypeError("The 'model_conf' argument must be a string.")
        
    plt.style.use('fast') 

    n_metrics = len(metrics)
    fig, axs = plt.subplots(1, n_metrics, figsize=(25, 8))

    if n_metrics == 1:
        axs = [axs]

    for i, metric in enumerate(metrics):
        if metric not in history.history or f"val_{metric}" not in history.history:
            raise ValueError(f"{metric} not found in the history object.")

        axs[i].plot(history.history[metric], "limegreen", marker=".", alpha=0.7)
        axs[i].plot(history.history[f"val_{metric}"], "orangered", marker=".", alpha=0.7)
        axs[i].set_title(f"{metric.title()} of {model.name}\nConfiguration: {model_conf}")
        axs[i].set_ylabel(metric.title())
        axs[i].set_xlabel('Epoch')
        axs[i].legend(['train', 'validation'], loc='best')
        axs[i].grid(linestyle='--', linewidth=0.4)

        # Creates box indicating maximum validation metric
        xmax = np.argmax(history.history[f"val_{metric}"])
        ymax = max(history.history[f"val_{metric}"])
        text = f"{metric.title()} Val.: {ymax:.3f}"
        bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=0.5)
        arrowprops1 = dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
        kw = dict(xycoords='data', textcoords="offset points",
                  arrowprops=arrowprops1, bbox=bbox_props, ha="right", va="center")
        axs[i].annotate(text, xy=(xmax, ymax), xytext=(-15, -30), **kw)

        if metric == 'loss' and y_log:
            axs[i].set_yscale('log')

    plt.tight_layout()


"""
#Just some tests for cpu and ram bar with multiprocessing
#not working properly

import multiprocessing
from time import sleep
import psutil
from tqdm import tqdm

def monitor_usage():
    with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:
        cpu_process = multiprocessing.Process(target=usage, args=(cpubar,))
        ram_process = multiprocessing.Process(target=usage, args=(rambar,))
        cpu_process.start()
        ram_process.start()
        cpu_process.join()
        ram_process.join()

def usage(bar):
    while True:
        bar.n = psutil.cpu_percent() if bar.desc == 'cpu%' else psutil.virtual_memory().percent
        bar.refresh()
        sleep(0.5)
        
        
        
def categorical_get_dummies(df: pd.DataFrame, categorical_columns = []):
    colunas_one_hot = {}
    for coluna in categorical_columns:
        codes, uniques = pd.factorize(df[coluna].unique())
        colunas_one_hot[coluna] = {"uniques": uniques, "codes":codes}
        df[coluna] = df[coluna].replace(colunas_one_hot[coluna]["uniques"], colunas_one_hot[coluna]["codes"])
        print(coluna)
    df = pd.get_dummies(data=df, columns=categorical_columns)
"""


def categorical_get_dummies(df: pd.DataFrame, categorical_columns: list): #-> pd.DataFrame:
    """
    One-hot encode the categorical columns of a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    categorical_columns : list
        A list of the categorical column names.

    Returns
    -------
    pandas.DataFrame
        The dataframe with the categorical columns one-hot encoded.
    dict
        A dictionary with the unique values and codes for each categorical column.
    """

    # Check input types
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(categorical_columns, list):
        raise TypeError("Input 'categorical_columns' must be a list of column names.")

    # Create dictionary of unique values and codes for each categorical column
    colunas_one_hot = {coluna: {"uniques": df[coluna].unique(), "codes": pd.factorize(df[coluna].unique())[0]} 
                       for coluna in categorical_columns}

    # Replace categorical values with codes and one-hot encode the columns
    for coluna in categorical_columns:
        df[coluna] = df[coluna].replace(colunas_one_hot[coluna]["uniques"], colunas_one_hot[coluna]["codes"])
    df = pd.get_dummies(data=df, columns=categorical_columns)

    return df, colunas_one_hot


def encode_labels(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Encodes the labels in a DataFrame using LabelEncoder.

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing the label column to be encoded.
    label : str
        The name of the column containing the labels to be encoded.

    Returns:
    --------
    pandas DataFrame
        The original DataFrame with the label column encoded.
    LabelEncoder
        The fitted LabelEncoder object.
    """

    # Importing the LabelEncoder class from the scikit-learn library
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    le.fit(df[label].values)
    
    # Encoding the label column using the fitted LabelEncoder
    df[label] = le.transform(df[label].values)
    # Returning the DataFrame with the encoded label column
    return df, le

def apply_smotenc(df: pd.DataFrame, label: str, categorical_indices: list = [],sampling_strategy: str = "auto", random_state: int = 0) -> pd.DataFrame:
    """
    This function applies the SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features)
    algorithm to oversample a dataset with imbalanced classes. SMOTENC is a variation of the SMOTE algorithm that can
    handle datasets with both numerical and categorical features.

    Parameters:
        - df (pd.DataFrame): a Pandas DataFrame containing the dataset to be oversampled.
        - label (str): the name of the column containing the target variable.
        - categorical_indices (list): a list containing the indices of the columns in `df` that contain categorical
                                      variables. Default: [].
        - sampling_strategy (str): the sampling strategy to use when generating the synthetic samples. Default: "auto".
        - random_state (int): the random seed to use for reproducibility. Default: 0.

    Returns:
        - df_smote (pd.DataFrame): a new Pandas DataFrame containing the original data plus the synthetic minority class
                                   samples generated by the SMOTENC algorithm.

    """
    from imblearn.over_sampling import SMOTENC
    
    #check if the value sent in the sampling_strategy parameter is valid
    valid_strategies = ["minority", "not minority", "not majority", "all", "auto"]
    if sampling_strategy not in valid_strategies:
        raise ValueError("Invalid sampling strategy.")
    
    # Make a copy of the input dataframe and separate the target variable column
    X = df.copy()
    X = X.drop(columns=[label])
    y = df[label].copy()
    
    # Apply the SMOTENC algorithm to oversample the dataset
    print(f"Started SMOTENC; size of df - {df.size} ")
    smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    
    # Create a new dataframe with the oversampled dataset
    df_smote = pd.DataFrame(X_resampled, columns=X.columns)
    df_smote[label] = y_resampled
    
    # Print the size of the original and oversampled datasets, and return the oversampled dataset
    print(f"finished SMOTENC; size of df - {df_smote.size}")
    return df_smote


def apply_smotenc_bigdata(df: pd.DataFrame, label: str, categorical_indices: list = [], random_state: int = 0) -> pd.DataFrame:
    """
    This function applies the SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features)
    algorithm to oversample a dataset with imbalanced classes. SMOTENC is a variation of the SMOTE algorithm that can
    handle datasets with both numerical and categorical features. This function is doing a cicle to oversample the data 
    with the minor class. Because otherwise the SMOTENC algorithm is not working with big data.
    

    Parameters:
        - df (pd.DataFrame): a Pandas DataFrame containing the dataset to be oversampled.
        - label (str): the name of the column containing the target variable.
        - categorical_indices (list): a list containing the indices of the columns in `df` that contain categorical
                                      variables. Default: [].
        - random_state (int): the random seed to use for reproducibility. Default: 0.

    Returns:
        - df_smote (pd.DataFrame): a new Pandas DataFrame containing the original data plus the synthetic minority class
                                   samples generated by the SMOTENC algorithm.

    """
    from imblearn.over_sampling import SMOTENC
    # Make a copy of the input dataframe and separate the target variable column
    X_resampled = df.copy()
    X_resampled=X.drop(columns=[label,"Attack_label"])
    y_resampled=df[label].copy()
    
    # Apply the SMOTENC algorithm to oversample the dataset
    print(f"Started SMOTENC; size of df - {df.size} ")
    
    # Apply the SMOTENC algorithm to oversample the dataset
    smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=random_state,sampling_strategy="minority")
    for labels in np.unique(y_resampled):
        X_resampled, y_resampled= smote_nc.fit_resample(X_resampled, y_resampled)
    
    df_smote = pd.DataFrame(X_resampled, columns=X_resampled.columns)
    df_smote[label]=y_resampled
    
    # Print the size of the original and oversampled datasets, and return the oversampled dataset
    print(f"finished SMOTENC; size of df - {df_smote.size}")
    return df_smote