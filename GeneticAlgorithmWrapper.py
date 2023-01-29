from mp_utils import *
import pandas as pd
import random
from enum import Enum


class MPMethod(Enum):
    ThresholdBase = 0
    IsolationForest = 1
    PredefinedDiscords = 2

class FitnessType(Enum):
    F1_Score = 0
    LOG_MP_SUM = 1

class Verbosity(Enum):
    NoLog  = 0
    HalfLog = 1
    FullLog = 2
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplementedError("!!!")


class GeneticAlgo:
    verbosity_level = Verbosity.NoLog
    mp_map = {  MPMethod.ThresholdBase : ThresholdMatrixProfile,
                MPMethod.IsolationForest : IsolationMatrixProfile,
                MPMethod.PredefinedDiscords : PredefinedMatrixProfile}

    def __init__(self, df:pd.DataFrame,
                 max_features:int,
                 population_bag_size:int = 3,
                 fitness : FitnessType = FitnessType.F1_Score,
                 mp_method : MPMethod = MPMethod.ThresholdBase, **kwargs):
        
        #mp_kwargs : TH{'threshold' : 2.0} , PR{'discord_number' : 500} olabilir.
        
        print('Genetic Algorithm Process is ready to start')
        self.df = df.copy()
        self.feature_number = max_features
        self.pop_bag_size = population_bag_size
        self.fitness_type = fitness
        self.mp_method = mp_method
        self.mp_kwargs = kwargs
        
        self.y = df[["Label"]]
        self.X = df.drop(["Label"], axis = 1)
        self.feature_map = {i : feat_name for i, feat_name in enumerate(self.X.columns)}
        self.X.columns = list(range(0, len(self.X.columns)))

        self.creport = None
        self.eval_result = None


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
            mp_manager = GeneticAlgo.mp_map[self.mp_method](df=individual, **(self.mp_kwargs))
            cost, f1_score = mp_manager.calculate_cost()
                
            if (GeneticAlgo.verbosity_level > Verbosity.NoLog):
                print(f'processing solution: {individual.columns.to_list()}')
                print(f"f1-score is: {mp_manager.get_f1_score()}")
                            
            del mp_manager
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
            if self.fitness_type == FitnessType.LOG_MP_SUM:
                min_wgh = [abs(np.min(list(result['fit_vals'])) - i) for i in list(result['fit_vals'])]
            elif (self.fitness_type == FitnessType.F1_Score):
                min_wgh = [abs(np.min(list(result['f1-scores'])) - i) for i in list(result['f1-scores'])]

            from scipy.special import logsumexp
            #TODO: find a way instead of logsumexp
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
        if (GeneticAlgo.verbosity_level > Verbosity.NoLog):
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
        if (len(sol) > 2):
            rd_idx = random.randint(0, len(sol) - 1)
            del sol[rd_idx]
        return sol