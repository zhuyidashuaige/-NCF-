import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

# Load the dataset with GBK encoding
data = pd.read_csv('4399.csv', encoding='GBK')


# Data Preprocessing
def preprocess_data(data):
    # Encode game types as numerical values
    game_type_mapping = {label: idx for idx, label in enumerate(data['游戏类型'].unique())}
    data['游戏类型编码'] = data['游戏类型'].map(game_type_mapping)

    # Combine all text features for TF-IDF vectorization
    data['所有评论'] = data[['评论1', '评论2']].agg(' '.join, axis=1)

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=100)  # Limit features to 100 for simplicity
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['所有评论'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Combine TF-IDF features with other numerical features
    features = pd.concat([data[['评分', '游戏类型编码']], tfidf_df], axis=1)

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, data['游戏名字'], tfidf_vectorizer, scaler, game_type_mapping


# Execute preprocessing
features, game_names, tfidf_vectorizer, scaler, game_type_mapping = preprocess_data(data)

# Save the preprocessed data and related objects for later use
np.save('features.npy', features)
game_names.to_csv('game_names.csv', index=False, encoding='utf-8')
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('game_type_mapping.pkl', 'wb') as f:
    pickle.dump(game_type_mapping, f)

print("Data preprocessing completed and saved.")
