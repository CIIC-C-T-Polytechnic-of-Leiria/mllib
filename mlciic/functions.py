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

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



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
        if col != "Attack_type":
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
    return display(summary_df)




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
    
    Returns:
    None
    
    Raises:
    ValueError: If `average` is not one of {'micro', 'macro', 'weighted', 'binary'} or None.
    
    """
    # Check if average parameter is valid
    if average != 'micro' and average != 'macro' and average != 'weighted' and average != 'binary' and average != None:
        print("Average must be one of this options: {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} or None, default=’binary’")
        return
    
    # Print the name of the model and calculate accuracy and precision
    print(f"--- Performance of {modelName} ---")
    acc = accuracy_score(y_true = yTrue, y_pred = yPred)
    precision = precision_score(y_true = yTrue, y_pred = yPred, average = average)
    print(f'Accuracy : {np.round(acc*100,2)}%\nPrecision: {np.round(precision*100,2)}%')
    
    # Calculate and print recall and F1-score
    f1 = f1_score(y_true = yTrue, y_pred = yPred, average = average)
    recall = recall_score(y_true = yTrue, y_pred = yPred, average = average)
    print(f'Recall: {np.round(recall*100,2)}%\nF1-score: {np.round(f1*100,2)}%')
    
    #auc_sklearn = roc_auc_score(y_true = yTrue, y_score = yPred, average = average)
    #print(f'Roc auc: {np.round(auc_sklearn*100,2)}%')
    
    # Calculate and print balanced accuracy and classification report
    print(f"Balanced accuracy: {np.round(balanced_accuracy_score(yTrue, yPred)*100,2)}%")
    print(f"Classification report:\n{classification_report(yTrue, yPred)}")
    

def start_measures():
    tracemalloc.start()
    start_time = time.time()
    return start_time
    
def stop_measures(start_time):
    print("(current, peak)",tracemalloc.get_traced_memory())
    tracemalloc.stop()
    print("--- %s segundos ---" % (time.time() - start_time))



#Functions removed from Tiago code
#The original come from SO (https://stackoverflow.com/questions/48139899/correlation-matrix-plot-with-coefficients-on-one-side-scatterplots-on-another)
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


def save_model(model, model_name): 
    # guarda modelo no disco
    filename = model_name + '.sav'
    pickle.dump(model, open(filename, 'wb'))
    
def load_model(model_name): 
    # carrega modelo do disco
    model = pickle.load(open(model_name + '.sav', 'rb'))    
    return model

def heatmap(df,size=40):
    corr = df.corr().round(2)
    plt.figure(figsize=(size, size))
    sns.heatmap(corr, cmap="Blues", annot=True)
    plt.show()

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
"""