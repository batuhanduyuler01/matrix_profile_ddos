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
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='lightgreen', lw=3),
                    Line2D([0], [0], color='red', lw=3),
                    Line2D([0], [0], color='blue', lw=3),
                    Line2D([0], [0], color='black', lw=3),
                    Line2D([0], [0], color='brown', lw=3)]
    
    attack_color_dict = {'syn' : 'lightgreen', 'ntp' : 'red', 'udp' : 'blue', 'udp_lag' : 'black', 'ldap' : 'brown'}
    
    total_duration = len(df) / 60
    print(f"Total Duration Of Traffic is: {total_duration} minutes")
    xAxis = list(range(len(df)))
    yAxis = df["Label"].to_list()
    fig = plt.figure(figsize=(25,15))
    ax = fig.add_subplot()
    ax.plot(xAxis, yAxis)
    
    legend_custom_lines = []
    legend_custom_names = []
    for attack in attack_list:
        face_color = attack_color_dict[attack[0]]
        attack_duration = attack[1]
        duration_before_attack = attack[2]
        
        attack_index = list(attack_color_dict.keys()).index(attack[0])
        if attack[0] not in legend_custom_names:
            legend_custom_lines.append(custom_lines[attack_index])
            legend_custom_names.append(attack[0])
        
        rect = Rectangle((duration_before_attack * 60, 0), attack_duration * 60, 1, facecolor=face_color)
        ax.add_patch(rect)

    
    ax.legend(legend_custom_lines, legend_custom_names,  prop={'size': 20})
    
    plt.ylabel('Label')
    plt.xlabel('Seconds')
    plt.title('Network Traffic')
    plt.show()