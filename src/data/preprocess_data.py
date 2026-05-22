import pandas as pd
from src.data.load_data import load_file

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by dropping duplicates and null values
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be cleaned
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    
    return df

def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop specified columns from the DataFrame
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to drop columns from
    columns : list
        List of column names to be dropped
        
    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns dropped
    """
    
    df = df.drop(columns=columns, errors="ignore")
    return df

def rename_columns(df: pd.DataFrame, columns_mapping: dict) -> pd.DataFrame:
    """Rename columns in the DataFrame based on a mapping dictionary
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to rename columns in
    columns_mapping : dict
        Dictionary mapping old column names to new column names
        
    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
    """
    
    df = df.rename(columns=columns_mapping)
    return df

def prepare_text(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    """Prepare the text data by converting it to lowercase and removing punctuation
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to prepare text in
    text_cols : list
        List of column names containing the text data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with prepared text data
    """
    
    for col in text_cols:
        df[col] = df[col].astype(str).str.lower()
        df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
    
    return df

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the entire preprocessing pipeline on the DataFrame
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be preprocessed
        
    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame
    """
    
    cols_to_drop = ["text", "Label", "date", "subject", "char_count", "word_count", "avg_word_length"]
    
    cols_to_rename = {
        "label_number": "label",
        "clean_text": "text"
    }
    
    text_cols = ["text", "title"]
    
    df = clean_df(df)
    df = drop_columns(df, cols_to_drop)
    df = rename_columns(df, cols_to_rename)
    df = prepare_text(df, text_cols)
    return df

def save_file(df: pd.DataFrame, file_path: str) -> None:
    """Save the preprocessed DataFrame to a csv file
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be saved
    file_path : str
        File path for the preprocessed file
    """
    
    df.to_csv(file_path, index=False)

def main():
    raw_file_path = "data/raw/news_full.csv"
    processed_file_path = "data/processed/news_preprocessed.csv"
    
    df = load_file(file_path=raw_file_path)
    df = preprocess_pipeline(df)
    save_file(df, file_path=processed_file_path)
    print(df.head())
    
    return df

if __name__ == "__main__":
    main()