from MatrixProfileWrapper import MatrixProfileManager
import pandas as pd
import numpy as np


class PredefinedMatrixProfile(MatrixProfileManager):
    def __init__(self, df:pd.DataFrame, window_size:int = 60, method = 'mpx', discord_number:int=0):
        if (discord_number < 1):
            raise ValueError("Discord Number Can Not be Lower than 1")
        super().__init__(df, window_size, method)
        self.discord_number = discord_number
        
    def calculate_discords(self):
        from collections import Counter
        curr_mps_dict = {f_idx: np.argsort(self.mps[idx])[::-1][:1000] for idx, f_idx in enumerate(self.df.columns)}
        self.curr_mp_dict = {f_idx: np.sort(self.mps[idx])[::-1][:self.discord_number]
                             for idx, f_idx in enumerate(self.df.columns)}

        for idx, indices in curr_mps_dict.items():
            indice_list = []
            for indice in indices:
                indice_list.extend(list(range(indice, indice + self.window_size - 1)))

            sorted_discords = sorted(Counter(indice_list).items(), key=lambda t:t[1], reverse=True)
            sorted_discord_indexes = [elem[0] for elem in sorted_discords[:self.discord_number]]

            self.discord_dict[idx] = sorted_discord_indexes
        
        
    def majority_vote_discords(self):
        from collections import Counter
        overall_list = []
        for ft, ids_list in self.discord_dict.items():
            overall_list.extend(ids_list)

        sorted_overall = (sorted(Counter(overall_list).items(), key=lambda t:t[1], reverse=True))
        self.discords = [elem[0] for elem in sorted_overall[:self.discord_number]]
        