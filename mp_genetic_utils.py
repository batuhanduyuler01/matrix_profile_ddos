import pandas as pd
import numpy as np
import stumpy
import random

class MatrixProfileManager:
    if 'global_df' not in dir():
        global_df = pd.read_csv('../verisetleri/ddos_dataset_on_seconds.csv', low_memory=True)

    THRESHOLD_BASE_ACTIVE = False
    threshold = 1.5

    def __init__(self, df:pd.DataFrame, window_size:int = 60, discord_number = 476, method='mpx', measure='acc'):
        self.measurement = measure
        self.window_size = window_size
        self.discord_number = discord_number
        self.discord_dict = {}
        self.discords = []
        self.df = df
        self.mp_method = method
        self.curr_mp_dict = {}

    def calculate_mp_multivariate_stumpy(self):
        curr_mps, curr_indices = stumpy.mstump(self.df, self.window_size)
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
        from collections import Counter
        curr_mps_dict = dict()
        curr_mps_dict = {f_idx: np.argsort(self.mps[idx])[::-1][:1000] for idx, f_idx in enumerate(self.df.columns)}
        self.curr_mp_dict = {f_idx: np.sort(self.mps[idx])[::-1][:self.discord_number] for idx, f_idx in enumerate(self.df.columns)}

        for idx, indices in curr_mps_dict.items():
            # print(f'now processing current idx: {idx}') 
            indice_list = []
            for indice in indices:
                #get mp point window
                indice_list.extend(list(range(indice, indice + self.window_size - 1)))
            #sort the indices by count
            sorted_discords = sorted(Counter(indice_list).items(), key=lambda t:t[1], reverse=True)
            sorted_discord_indexes = [elem[0] for elem in sorted_discords[:self.discord_number]]

            self.discord_dict[idx] = sorted_discord_indexes

    def majority_vote_discords(self):
        from collections import Counter
        overall_list = []
        for ft, ids_list in self.discord_dict.items():
            overall_list.extend(ids_list.copy())

        if (MatrixProfileManager.THRESHOLD_BASE_ACTIVE):
            self.discords = list(set(overall_list)).copy()
            return

        sorted_overall = (sorted(Counter(overall_list).items(), key=lambda t:t[1], reverse=True))
        self.discords = [elem[0] for elem in sorted_overall[:self.discord_number]]


    def obtain_y_vals(self):
        df_idxs = list(range(0, len(MatrixProfileManager.global_df)))
        for idx in self.discords:
            df_idxs.remove(idx)
  
        
        self.pred_df = pd.DataFrame()
        self.pred_df['y_true'] = MatrixProfileManager.global_df["Label"].copy()
        self.pred_df["y_pred"] = MatrixProfileManager.global_df["Label"].copy()
        
        self.pred_df.iloc[df_idxs, 0] = 0
        self.pred_df.iloc[self.discords, 0] = 1

    def calculate_classification_report(self):
        from sklearn.metrics import classification_report
        if 'y_true' not in self.pred_df.columns:
            raise ValueError('true vals not included in df')

        if 'y_pred' not in self.pred_df.columns:
            raise ValueError('pred vals not included in df')

        self.creport = classification_report(self.pred_df["y_true"].to_list(),
                                             self.pred_df["y_pred"].to_list(), output_dict=True)["1"]

    def get_f1_score(self):
        if self.creport is None:
            raise ValueError('Classification Report is not ready!')
            
        return self.creport['f1-score']

    def get_mp_score(self):
        #maximize this
        return sum([sum(mp_score) for mp_score in self.curr_mp_dict.values()]) / len(self.curr_mp_dict.keys())

    def calculate_cost(self):
        if self.mp_method.lower() == 'mpx':
            self.calculate_mp_seperately_mpx()
        else:
            self.calculate_mp_multivariate_stumpy()
        
        self.calculate_discords()
        self.majority_vote_discords()
        self.obtain_y_vals()
        self.calculate_classification_report()

        f1_score = self.get_f1_score()
        mp_score = self.get_mp_score()
        return mp_score, f1_score

    def calculate_thresholded_discords(self):
        from collections import Counter
        curr_mps_dict = dict()
        threshold = MatrixProfileManager.threshold
        curr_mps_dict = {f_idx: np.where(self.mps[idx] > threshold)[0].tolist() for idx, f_idx in enumerate(self.df.columns)}
        self.curr_mp_dict = {f_idx: np.sort(self.mps[idx])[::-1][:10] for idx, f_idx in enumerate(self.df.columns)}

        for idx, indices in curr_mps_dict.items():
            # print(f'now processing current idx: {idx}') 
            indice_list = []
            for indice in indices:
                #get mp point window
                indice_list.extend(list(range(indice, indice + self.window_size - 1)))

            if (MatrixProfileManager.THRESHOLD_BASE_ACTIVE):
                self.discord_dict[idx] = indice_list.copy()
            else:
                AssertionError("wrong func!")

    def calculate_threshold_based_cost(self):    
        if self.mp_method.lower() == 'mpx':
            self.calculate_mp_seperately_mpx()
        else:
            self.calculate_mp_multivariate_stumpy()

        self.calculate_thresholded_discords()
        self.majority_vote_discords()
        self.obtain_y_vals()
        self.calculate_classification_report()

        f1_score = self.get_f1_score()
        mp_score = self.get_mp_score()
        return mp_score, f1_score


class GeneticAlgo:
    verbosity_level = 0
    thresholded_mp = False
    def __init__(self, df:pd.DataFrame, max_features:int, population_bag_size:int = 3, fitness = 'MP'):
        print('Genetic Algorithm Process is ready to start')
        self.df = df.copy()
        self.y = df[["Label"]]
        self.X = df.drop(["Label"], axis = 1)
        self.feature_map = {i : feat_name for i, feat_name in enumerate(self.X.columns)}
        self.X.columns = list(range(0, len(self.X.columns)))
        self.feature_number = max_features
        self.pop_bag_size = population_bag_size
        self.creport = None
        self.eval_result = None
        self.fitness_type = fitness
        

    def initialize_population(self):
        self.population_bag = []
        for _ in range(self.pop_bag_size):
            #0 veya 1 atiyoruz feature pick or not pick, 1 olanlari appendliyoruz.
            genes = [random.randrange(0,2) for _ in range(self.feature_number)]
            gene_indexes = [idx for idx, f in enumerate(genes) if f == 1]
            if (len(gene_indexes) == 0):
                gene_indexes.append(random.randint(1,self.feature_number))

            self.population_bag.append(self.X.iloc[:, gene_indexes])

        return self.population_bag

    def create_population(self, pop_bag) -> pd.DataFrame:
        self.population_bag.clear()
        for elem in pop_bag:
            self.population_bag.append(self.X.iloc[:, elem])
            
        return self.population_bag

    def fitness_function(self, individual:pd.DataFrame):
        mp_manager = MatrixProfileManager(individual, window_size=60, discord_number=1000, method='mpx', measure='mp')
        if (GeneticAlgo.thresholded_mp == False):
            cost, f1_score = mp_manager.calculate_cost()
        elif (GeneticAlgo.thresholded_mp == True):
            cost, f1_score = mp_manager.calculate_threshold_based_cost()
            
        if (GeneticAlgo.verbosity_level < 2):
            print(f'processing solution: {individual.columns.to_list()}')
            print(f"f1-score is: {mp_manager.get_f1_score()}")
        #return f1score instead of cost in order to maximize f1-score:
        
        # return cost, f1_score
        return cost, f1_score

    def eval_fit_population(self, pop_bag):
        #This evaluation is based on minimizing the cost!
        result = {}
        fit_vals_lst = []
        f1_score_lst = []
        solutions = []
        for individual in pop_bag:
            if (type(individual) != pd.DataFrame):
                assert(True)

            cost, f1_sc = self.fitness_function(individual.copy())
            fit_vals_lst.append(cost)
            f1_score_lst.append(f1_sc)
            solutions.append(individual.columns.to_list())
            
        result["fit_vals"] = fit_vals_lst
        result["f1-scores"] = f1_score_lst 
        if self.fitness_type == "MP":
            min_wgh = [abs(np.min(list(result['fit_vals'])) - i) for i in list(result['fit_vals'])]
        else:
            min_wgh = [abs(np.min(list(result['f1-scores'])) - i) for i in list(result['f1-scores'])]
        
        from scipy.special import logsumexp
        result["fit_wgh"]  = [i/logsumexp(min_wgh) for i in min_wgh]
        result["solution"] = np.array(solutions, dtype=list).tolist()
        
        self.eval_result = result.copy()
        return result

    def find_best(self, eval_result:dict)->dict:
        # Best individual so far
        best_fit = np.max(eval_result["fit_vals"])
        best_fit_index = eval_result["fit_vals"].index(best_fit)
        best_solution  = eval_result["solution"][best_fit_index]
        f1_sc = eval_result["f1-scores"][best_fit_index]
        print(f'best fit: {best_fit}\nsolution: {best_solution}\nf1Score: {f1_sc}')
        return {'best_fit': best_fit, 'index' : best_fit_index,
                 'solution': best_solution, 'f1-score' : f1_sc}

    def pick_one(self, pop_bag):
        
        if self.eval_result is None:
            eval_result = self.eval_fit_population(pop_bag)
        else:
            eval_result = self.eval_result

        notPicked=True
        cnt = 0
        pickedSol = list()
        while (notPicked == True):
            rnIndex = random.randint(0, len(pop_bag)-1)
            rnPick  = eval_result["fit_wgh"][rnIndex]
            r = random.random()
            if  r <= rnPick:
                pickedSol = eval_result["solution"][rnIndex]
                notPicked = False
            if (cnt > 250):
                pickedSol = eval_result["solution"][rnIndex]
                notPicked = False
            cnt += 1

        return pickedSol

    def crossover(self, solA, solB):
        
        n     = len(solA)
        child: list = []

        num_els = random.randint(0, self.feature_number)
        str_pnt = random.randint(0, max(0,n-3))
        end_pnt = n if int(str_pnt+num_els) > n else int(str_pnt+num_els)

        blockA = list(solA[str_pnt:end_pnt])
        child = blockA.copy()

        for elem in solB:
            if len(child) >= num_els:
                break
            if elem not in blockA:
                child.append(elem)  

        if (len(child) < 1):
            return solA

        return child

    def mutation(self,sol):
        
        n = len(sol)
        pos_1 = random.randint(0,n-1)
        pos_2 = random.randint(0,n-1)
        result = self.swap(sol, pos_1, pos_2)
        return result

    def swap(self,sol, posA, posB):
        result = sol.copy()
        elA = sol[posA]
        elB = sol[posB]
        result[posA] = elB
        result[posB] = elA
        return result