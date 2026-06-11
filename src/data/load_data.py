import pandas as pd

def load_file(file_path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Load csv file and convert it to DataFrame

    Parameters
    ----------
    file_path : str
        File path for the raw file
    encoding : str, optional
        Encoding of the csv file, by default "utf-8"

    Returns
    -------
    pd.DataFrame
        DataFrame that is created
    """

    df = pd.read_csv(file_path, encoding=encoding)
    return df