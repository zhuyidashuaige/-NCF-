import pandas as pd
import random
import string
import time

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

df = pd.read_csv('4399.csv', delimiter=',')
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
top_30_games.to_csv('total.csv', index=False)

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

print("统计完成，结果已输出到相应的CSV文件中。")
