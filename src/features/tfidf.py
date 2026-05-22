from sklearn.feature_extraction.text import TfidfVectorizer 

def vectorize_text(X_train, X_test): 
    vectorizer = TfidfVectorizer( 
                    stop_words="english", 
                    max_df=0.7, 
                    min_df=5, 
                    ngram_range=(1, 2) 
                ) 
    
    X_train_vec = vectorizer.fit_transform(X_train) 
    X_test_vec = vectorizer.transform(X_test) 
    
    return X_train_vec, X_test_vec, vectorizer