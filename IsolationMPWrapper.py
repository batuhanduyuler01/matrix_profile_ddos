from MatrixProfileWrapper import MatrixProfileManager
import pandas as pd


class IsolationMatrixProfile(MatrixProfileManager):
    def __init__(self, df:pd.DataFrame, window_size:int = 60, method = 'mpx'):
        super().__init__(df, window_size, method)
        
    def calculate_discords(self):
        import numpy as np
        from sklearn.ensemble import IsolationForest
        curr_mps_dict = {f_idx: np.log10((1 + self.mps[idx])).tolist()
                            for idx, f_idx in enumerate(self.df.columns)}
        
        self.mps_df= pd.DataFrame(curr_mps_dict)
        model=IsolationForest()
        model.fit(self.mps_df.iloc[:,:])
        
        self.mps_df['scores']=model.decision_function(self.mps_df.iloc[:,:])
        self.mps_df['anomaly']=model.predict(self.mps_df.iloc[:, :-1])
        
    def majority_vote_discords(self):
        self.discords = self.mps_df[self.mps_df["anomaly"] == -1].index.to_list()
