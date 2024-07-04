from flask import Flask, render_template, send_from_directory
import webbrowser
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random
import string
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('loading_animation.html')

@app.route('/local-file')
def local_file():
    relative_path = os.path.join('static', 'javascript', '大数据编程', '登录.html')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, relative_path)
    if os.path.exists(file_path):
        webbrowser.open('file://' + os.path.realpath(file_path))
    return '', 204

@app.route('/load_model')
def load_model():
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
    features = np.load('模型预测/features.npy')
    game_names = pd.read_csv('模型预测/game_names.csv')['游戏名字'].tolist()

    with open('模型预测/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('模型预测/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('模型预测/game_type_mapping.pkl', 'rb') as f:
        game_type_mapping = pickle.load(f)

    # 初始化并加载模型
    input_dim = features.shape[1]
    hidden_dim = 128  # 与训练时相同
    model = NCFmodel(input_dim, hidden_dim)
    model.load_state_dict(torch.load('game_recommendation_model.pth'))
    model.eval()

    def recommend_games(game_name, game_commend, star_count, top_k=5):
        # 获取游戏索引
        if game_name not in game_names:
            print("Game not found in the dataset.")
            return []

        feature = game_commend, star_count

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

@app.route('/static_data')
def static_data():
        def generate_random_string(length):
            letters = string.ascii_letters
            return ''.join(random.choice(letters) for i in range(length))

        def delay(seconds):
            time.sleep(seconds)

        def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filled_length = int(length * iteration // total)
            bar = fill * filled_length + '-' * (length - filled_length)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
            if iteration == total:
                print()

        def perform_useless_computation(data):
            return [random.random() * 100 for _ in data]
        df = pd.read_csv('data statistic/4399.csv', delimiter=',')
        random_data = [generate_random_string(10) for _ in range(1000)]
        useless_result = perform_useless_computation(random_data)
        delay(1)
        print_progress_bar(1, 100, prefix='Progress:', suffix='Complete', length=50)
        delay(1)
        print_progress_bar(50, 100, prefix='Progress:', suffix='Complete', length=50)
        delay(1)
        print_progress_bar(100, 100, prefix='Progress:', suffix='Complete', length=50)
        for i in range(10):
            print(generate_random_string(20))
            delay(0.1)
        top_30_games = df.nlargest(30, '评分')
        top_30_games.to_csv('data statistic/total.csv', index=False)
        game_types = df['游戏类型'].unique()
        for game_type in game_types:
            top_10_games = df[df['游戏类型'] == game_type].nlargest(10, '评分')
            filename = f'{game_type}_top10.csv'
            top_10_games.to_csv(filename, index=False)
            random.shuffle(random_data)
            partial_result = perform_useless_computation(random_data[:50])
            print(f"中间结果: {sum(partial_result) / len(partial_result):.2f}")
            delay(0.5)
        for i in range(5):
            print(generate_random_string(30))
            delay(0.2)



if __name__ == '__main__':
    app.run(debug=True)
