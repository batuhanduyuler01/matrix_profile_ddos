from MatrixProfileWrapper import MatrixProfileManager
import pandas as pd


class IsolationMatrixProfile(MatrixProfileManager):
    def __init__(self, df:pd.DataFrame, window_size:int = 60, method = 'mpx',  n_estimators = None, contamination = None):
        super().__init__(df, window_size, method)
        self.n_estimators = n_estimators or None
        self.contamination = contamination or None
        
    def calculate_discords(self):
        import numpy as np
        from sklearn.ensemble import IsolationForest
        curr_mps_dict = {f_idx: np.log10((1 + self.mps[idx])).tolist()
                            for idx, f_idx in enumerate(self.df.columns)}
        
        self.mps_df= pd.DataFrame(curr_mps_dict)
        n_est = 100
        cont = "auto"

        if self.n_estimators is not None:
            n_est = self.n_estimators
        if self.contamination is not None:
            cont = self.contamination

        model=IsolationForest(n_estimators = n_est, contamination = cont)
        model.fit(self.mps_df.iloc[:,:])
        
        self.mps_df['scores']=model.decision_function(self.mps_df.iloc[:,:])
        self.mps_df['anomaly']=model.predict(self.mps_df.iloc[:, :-1])
        
    def majority_vote_discords(self):
        self.discords = self.mps_df[self.mps_df["anomaly"] == -1].index.to_list()
        # discord_window_list = self.mps_df[self.mps_df["anomaly"] == -1].index.to_list()
        # indice_list = []
        # for indice in discord_window_list:
        #     indice_list.extend(list(range(indice, indice + self.window_size - 1)))

        # self.discords = list(set(indice_list))
