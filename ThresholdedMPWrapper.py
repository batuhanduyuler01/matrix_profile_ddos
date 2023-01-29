from MatrixProfileWrapper import MatrixProfileManager
import numpy as np
import pandas as pd

class ThresholdMatrixProfile(MatrixProfileManager):
    def __init__(self, df:pd.DataFrame, window_size:int = 60, method = 'mpx', threshold:float=2.0):
        super().__init__(df, window_size, method)
        self.threshold = threshold
        
    def calculate_discords(self):
        curr_mps_dict = {f_idx: np.where(self.mps[idx] > self.threshold)[0].tolist()
                         for idx, f_idx in enumerate(self.df.columns)}
        
        
        for idx, indices in curr_mps_dict.items():
            indice_list = []
            for indice in indices:
                #get mp point window
                indice_list.extend(list(range(indice, indice + self.window_size - 1)))

            self.discord_dict[idx] = list(set(indice_list))
            
        self.curr_mp_dict = curr_mps_dict
            
    def majority_vote_discords(self):
        overall_list = []
        for ft, ids_list in self.discord_dict.items():
            overall_list.extend(ids_list)

        self.discords = list(set(overall_list))
        

        

