import pandas as pd
import numpy as np


class MatrixProfileManager:
    if 'global_df' not in dir():
        global_df = None
        
    def __init__(self, df:pd.DataFrame, window_size:int = 60, method = 'mpx'):
        if (type(df) == pd.Series):
            df = pd.DataFrame(df)
            
        if (type(df) != pd.DataFrame):
            raise ValueError("df must be pd.DataFrame")
            
        self.df = df
        self.window_size = window_size
        self.mp_method = method
        self.discord_dict = {}
        self.discords = []
        self.curr_mp_dict = {}
        self.mps = None
        self.pred_df = pd.DataFrame()
        self.method_func_dict = {'mpx'     : self.calculate_mp_seperately_mpx,
                                 'mstump'  : self.calculate_mp_multivariate_stumpy}
        
    def calculate_mp_multivariate_stumpy(self):
        from stumpy import mstump
        curr_mps, curr_indices = mstump(self.df, self.window_size)
        self.mps = curr_mps
        
    def calculate_mp_seperately_mpx(self):
        import matrixprofile as mp
        mp_list = []
        
        for ft in self.df.columns:
            inputSignal = self.df[ft].to_list()
            matrix_profile = mp.compute(inputSignal, windows=self.window_size, threshold=0.95, n_jobs=4)
            mp_list.append(matrix_profile['mp'])

        self.mps = np.array(mp_list)
        
        
        
    def calculate_discords(self):
        raise NotImplementedError("Base Class Env")
        
    def majority_vote_discords(self):
        raise NotImplementedError("Base Class Env")
        
    def obtain_y_vals(self):
        if (MatrixProfileManager.global_df is None):
            raise ValueError("global df is none")
            
        df_idxs = list(range(0, len(MatrixProfileManager.global_df)))
        benign_preds = [idx for idx in df_idxs if idx not in self.discords]
  
        self.pred_df['y_true'] = MatrixProfileManager.global_df["Label"].copy()
        self.pred_df["y_pred"] = MatrixProfileManager.global_df["Label"].copy()
        
        self.pred_df.iloc[df_idxs, 1] = 0
        self.pred_df.iloc[self.discords, 1] = 1
        
    def calculate_classification_report(self):
        from sklearn.metrics import classification_report
        if 'y_true' not in self.pred_df.columns:
            raise ValueError('true vals not included in df')

        if 'y_pred' not in self.pred_df.columns:
            raise ValueError('pred vals not included in df')

        out_dict = classification_report(self.pred_df["y_true"].to_list(),
                                             self.pred_df["y_pred"].to_list(), output_dict=True)
        
        self.creport = out_dict["1"]
        self.creport["accuracy"] = out_dict["accuracy"]
        
    def get_f1_score(self):
        if self.creport is None:
            raise ValueError('Classification Report is not ready!')
            
        return self.creport['f1-score']

    def get_mp_score(self):
        #maximize this
        import numpy as np
        return 0
        #return sum([sum(np.log10(1+mp_score)) for mp_score in self.curr_mp_dict.values()]) / len(self.curr_mp_dict.keys())
        
    def calculate_cost(self):
        self.method_func_dict[self.mp_method.lower()]()
        
        self.calculate_discords()
        self.majority_vote_discords()
        self.obtain_y_vals()
        self.calculate_classification_report()

        f1_score = self.get_f1_score()
        mp_score = self.get_mp_score()
        return mp_score, f1_score
        