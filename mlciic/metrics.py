import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

def dice_coef(y_true, y_pred, smooth):
    """
    Computes the dice coefficient between the ground truth and predicted segmentation masks.
    The dice coefficient is a similarity metric between two sets of binary data, which measures
    the overlap between the sets.

    Parameters:
        - y_true: Ground truth segmentation mask.
        - y_pred: Predicted segmentation mask.
        - smooth: A smoothing parameter to prevent division by zero.

    Returns:
        The dice coefficient between the ground truth and predicted segmentation masks.
    """
    
    y_true_f = np.flatten(y_true)
    y_pred_f = np.flatten(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return dice

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


def keras_metrics_graphs(history, model, model_conf="not provided!", metrics=['loss'], y_log=False):
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
    