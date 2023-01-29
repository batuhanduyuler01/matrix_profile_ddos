import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from matplotlib.patches import Rectangle


def upload_dataset_with_time(path:str):
    startingT = time.perf_counter()
    if 'pkl' in path:
        veriseti = pd.read_pickle(path)
    else:
        veriseti = pd.read_csv(path, low_memory=True)
    endingT = time.perf_counter()
    print(f"Dataset is loaded in {endingT - startingT} seconds")
    return veriseti


def plot_ddos(df: pd.DataFrame, attack_list:list[tuple]):
    attack_color_dict = {'syn' : 'lightgreen', 'ntp' : 'red', 'udp' : 'blue'}
    
    
    xAxis = list(range(len(df)))
    yAxis = df["Label"].to_list()
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    ax.plot(xAxis, yAxis)
    
    for attack in attack_list:
        face_color = attack_color_dict[attack[0]]
        attack_duration = attack[1]
        duration_before_attack = attack[2]
        
        rect = Rectangle((duration_before_attack * 60, 0), attack_duration * 60, 1, facecolor=face_color)
        ax.add_patch(rect)
    
    plt.ylabel('Label')
    plt.xlabel('Seconds')
    plt.title('Network Traffic')
    plt.show()