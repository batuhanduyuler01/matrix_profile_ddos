from GeneticAlgorithmWrapper import *
from mp_utils import * 
import utils
import pandas as pd
import random


df_necessary_cols = ['Date_Second',
 ' Flow Duration',
 ' Total Fwd Packets',
 ' Total Backward Packets',
 'Total Length of Fwd Packets',
 ' Total Length of Bwd Packets',
 ' Fwd Packet Length Max',
 ' Fwd Packet Length Min',
 ' Fwd Packet Length Mean',
 ' Fwd Packet Length Std',
 'Bwd Packet Length Max',
 ' Bwd Packet Length Mean',
 ' Bwd Packet Length Std',
 'Flow Bytes/s',
 ' Flow Packets/s',
 ' Flow IAT Mean',
 ' Flow IAT Std',
 ' Flow IAT Max',
 ' Flow IAT Min',
 'Fwd IAT Total',
 ' Fwd IAT Mean',
 ' Fwd IAT Std',
 ' Fwd IAT Max',
 ' Fwd IAT Min',
 'Bwd IAT Total',
 ' Bwd IAT Mean',
 ' Bwd IAT Std',
 ' Bwd IAT Max',
 'Label']


if __name__ == '__main__':
    data_paths = {  'ntp'    : 'ntp_ddos_14_minutes.csv',
                'udp'    : 'udp_ddos_2_minutes.csv',
                'syn'    : 'syn_ddos_3_minutes.csv',
                'benign' : 'ntp_benign_10_minutes.csv'}

    dataset_dict = {    'ntp' : None, 'udp' : None,
                    'syn' : None, 'benign' : None}

    for data_name, path in data_paths.items():
        data = utils.upload_dataset_with_time(path)
        if 'benign' in data_name:
            #manipulation
            data["Label"] = data["Label"].apply(lambda x: 0)
        
        dataset_dict[data_name] = data

    mixed_df = pd.concat([  dataset_dict["benign"], dataset_dict["ntp"].iloc[:60,:],
                                dataset_dict["benign"], dataset_dict["udp"].iloc[:60, :], dataset_dict["benign"].iloc[:300,:],
                                dataset_dict["syn"].iloc[:30,:], dataset_dict["ntp"].iloc[:30,:],
                                dataset_dict["benign"],  dataset_dict["benign"],
                                dataset_dict["udp"].iloc[:30, :],  dataset_dict["benign"]], axis=0).reset_index(drop=True)

    utils.plot_ddos(mixed_df, attack_list=[])

    mixed_df = mixed_df[[*df_necessary_cols]]
    MatrixProfileManager.global_df = mixed_df.copy()

    GA = GeneticAlgo(mixed_df.iloc[:,1:].copy(), max_features=20, population_bag_size=10,
                    fitness=FitnessType.F1_Score,
                    mp_method=MPMethod.ThresholdBase,  window_size=10)


    pop_bag = GA.initialize_population()
    generation_number = 20

    for generation in range(generation_number):
            print(f"Generation {generation} is started!")
            
            res = GA.eval_fit_population(pop_bag)
            best_fit, _, best_solution, best_f1_score = GA.find_best(res).values()
            
            if (generation == 0):
                best_fit_global      = best_fit
                best_solution_global = best_solution
                best_f1_global       = best_f1_score
            else:
                if (best_f1_score >= best_f1_global):
                    best_fit_global      = best_fit
                    best_f1_global       = best_f1_score
                    best_solution_global = best_solution

            new_pop_bag = []
            for i in range(len(GA.population_bag)):
                    # Pick 2 parents from the bag
                pA = GA.pick_one(pop_bag)
                pB = GA.pick_one(pop_bag)
                new_element = pA
                # Crossover the parents
                if random.random() <= 0.87:
                    new_element = GA.crossover(pA, pB)
                #Mutate the child
                if random.random() <= 0.5:
                    new_element = GA.mutation(new_element) 
                
                # Append the child to the bag
                new_pop_bag.append(new_element)
                # Set the new bag as the population bag
            pop_bag = GA.create_population(new_pop_bag)
        
    print("\n\n**** Generations Over ****\n")
    print(f"Best Fitness: {best_fit_global}")
    print(f"Best Solution: {best_solution_global}")
    print(f"F1-Score: {best_f1_score}")


