import pandas as pd
#from pprint import pprint
#import numpy as np

def displayInformationDataFrame(df,showCategoricals = False, showDetailsOnCategorical = False, showFullDetails = False):
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

