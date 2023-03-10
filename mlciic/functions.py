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
    
    Raises:
    ValueError: If `average` is not one of {'micro', 'macro', 'weighted', 'binary'} or None.
    
    """
    # Check if average parameter is valid
    if average != 'micro' and average != 'macro' and average != 'weighted' and average != 'binary' and average != None:
        print("Average must be one of this options: {???micro???, ???macro???, ???samples???, ???weighted???, ???binary???} or None, default=???binary???")
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
        memory_usage = tracemalloc.get_traced_memory()
        tracemalloc.stop()
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
    Save a trained machine learning model to disk.??

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

def heatmap(df,size=40):
    corr = df.corr().round(2)
    plt.figure(figsize=(size, size))
    sns.heatmap(corr, cmap="Blues", annot=True)
    plt.show()

    

def metrics_graphs(history, model, model_conf = "not provided!", iou = False, y_log = False):
    """
    Plots training and validation metrics (accuracy, loss, and IoU) of a Keras model during training.

    Parameters:
    -----------
    history: tf.keras.callbacks.History
        Object returned by model.fit() containing training history.
    model: tf.keras.Model
        The trained machine learning model.
    conf_modelo: str, optional (default="")
        The configuration of the model (optional)
    iou: bool, optional (default=False)
        Whether to plot the IoU score or not. If True, the function plots the IoU score over the epochs.
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

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,8))
    
    if iou:
        if 'iou_score' not in history.history or 'val_iou_score' not in history.history:
            raise ValueError("IoU score not found in the history object.")
        
        ax1.plot(np.array(history.history['iou_score']) * 100, "limegreen", marker=".")
        ax1.plot(np.array(history.history['val_iou_score']) * 100, "orangered", marker=".")
        ax1.set_title('Intersection over Union (IoU) of ' + model.name + '\n' + "Configuration: " + model_conf)
        ax1.set_ylabel('IoU (%)')
        ax1.set_xlabel('Epoch')
        ax1.legend(['train', 'validation'], loc='best')
        ax1.grid(linestyle='--', linewidth=0.4)
        
        # Creates box indicating maximum validation IoU
        xmax = np.argmax(history.history['val_iou_score'])
        ymax = max(history.history['val_iou_score']) * 100
        text = "IoU Val.:{:.3f} %".format(ymax)
        bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=0.5)   
        arrowprops1 = dict(arrowstyle="->",connectionstyle="arc3,rad=0.3")
        kw = dict(xycoords='data',textcoords="offset points",
                  arrowprops=arrowprops1, bbox=bbox_props, ha="right", va="center")
        ax1.annotate(text, xy=(xmax, ymax), xytext=(-15,-30), **kw)
        ax1.set_ylim(top=max(history.history['val_iou_score'] + 
                             history.history['iou_score']) * 100 + 1)
    else:
        if 'accuracy' not in history.history or 'val_accuracy' not in history.history:
            raise ValueError("Accuracy not found in the history object.")
        
        
        ax1.plot(np.array(history.history['accuracy'])*100,"limegreen",  marker =".")
        ax1.plot(np.array(history.history['val_accuracy'])*100, "orangered" ,  marker =".")
        ax1.set_title('Accuracy of ' + model.name + '\n' + "Configuration: " + model_conf)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_xlabel('Epoch')
        ax1.legend(['train', 'validation'], loc='best')
        ax1.grid(linestyle='--', linewidth=0.4)

        # Creates box indicating maximum validation accuracy
        xmax = np.argmax(history.history['val_accuracy'])
        ymax = max(history.history['val_accuracy'])*100
        text= "Acur. Val.:{:.3f} %".format(ymax)
        # lw, linewidth; fc, facebolor; ec, edgecolor
        bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=0.5)   
        arrowprops1 = dict(arrowstyle="->",connectionstyle="arc3,rad=0.3")
        kw = dict(xycoords='data',textcoords="offset points",
                  arrowprops=arrowprops1, bbox=bbox_props, ha="right", va="center")
        ax1.annotate(text, xy=(xmax, ymax), xytext=(-15,-30), **kw)
        ax1.set_ylim(top = max(history.history['val_accuracy'] + 
                               history.history['accuracy'])*100 + 1)  
    
        # Plots Cost Graph
        ax2.plot(history.history['loss'], "limegreen",  marker =".")
        ax2.plot(history.history['val_loss'],"orangered" ,  marker =".")
        ax2.set_title('Cost of '+ model.name +'\n'+ "Configuration: " + model_conf)
        ax2.set_ylabel('Cost')
        ax2.set_xlabel('Epoch')
        ax2.legend(['train', 'validation'], loc='best')
        ax2.grid(linestyle = '--', linewidth = 0.5)
        ax2.set_ylim(ymin=0)
        if y_log == True:
            ax2.set_yscale('log')
            ax2.set_ylim(None)
        
        # Creates box indicating minimum validation cost value
        xmin = np.argmin(history.history['val_loss'])
        ymin = min(history.history['val_loss'])
        text2= "Val. Cost:{:.3f}".format(ymin)
        bbox_props2 = dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw = 0.5)
        arrowprops2 = dict(arrowstyle="->",connectionstyle="arc3,rad=-0.3")
        kw2 = dict(xycoords='data',textcoords="offset points",
                   arrowprops=arrowprops2, bbox=bbox_props2, ha="right", va="center")
        ax2.annotate(text2, xy=(xmin, ymin), xytext=(70, 35), **kw2)
        
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
"""

