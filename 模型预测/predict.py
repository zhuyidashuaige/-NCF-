import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from sklearn.metrics.pairwise import cosine_similarity


# 定义简单的神经网络模型（需要与训练时相同）
class NCFmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NCFmodel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 加载数据和模型
features = np.load('features.npy')
game_names = pd.read_csv('game_names.csv')['游戏名字'].tolist()

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('game_type_mapping.pkl', 'rb') as f:
    game_type_mapping = pickle.load(f)

# 初始化并加载模型
input_dim = features.shape[1]
hidden_dim = 128  # 与训练时相同
model = NCFmodel(input_dim, hidden_dim)
model.load_state_dict(torch.load('game_recommendation_model.pth'))
model.eval()


def recommend_games(game_name,game_commend,star_count, top_k=5):
    # 获取游戏索引
    if game_name not in game_names:
        print("Game not found in the dataset.")
        return []

    feature = game_commend,star_count

    game_index = game_names.index(game_name)

    # 获取游戏特征向量
    game_feature = features[game_index]
    game_feature_tensor = torch.tensor(game_feature, dtype=torch.float32).unsqueeze(0)

    # 通过模型获取嵌入向量
    with torch.no_grad():
        game_embedding = model(game_feature_tensor).numpy()

    # 计算所有游戏的相似度
    all_embeddings = []
    for feature in features:
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            embedding = model(feature_tensor).numpy()
        all_embeddings.append(embedding)
    all_embeddings = np.array(all_embeddings).squeeze(1)

    similarities = cosine_similarity(game_embedding, all_embeddings).flatten()

    # 获取最相似的游戏
    similar_indices = similarities.argsort()[::-1][1:top_k + 1]  # 排除自身
    recommended_games = [game_names[i] for i in similar_indices]

    return recommended_games


# 输入游戏名，推荐其他游戏
input_game_name = "贪吃鱼进化"  # 示例游戏名
input_commend = "这个游戏我太爱了，简直爱的不要不要的，摸摸大"
input_star_count = "5"
recommended_games = recommend_games(input_game_name,input_commend,input_star_count)

print(f"Games similar to '{input_game_name,input_commend,input_star_count}':")
for game in recommended_games:
    print(game)
