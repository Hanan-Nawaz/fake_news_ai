import pandas as pd

def load_file(file_path: str) -> pd.DataFrame:
    """Load csv file and convert it to DataFrame

    Parameters
    ----------
    file_path : str
        File path for the raw file

    Returns
    -------
    pd.DataFrame
        DataFrame that is created
    """

    df = pd.read_csv(file_path, index_col = 0)
    return df