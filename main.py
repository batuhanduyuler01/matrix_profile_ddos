import mp_genetic_utils
import pandas as pd




if __name__ == '__main__':
    import random as rnd
    df = pd.read_csv("../verisetleri/ddos_dataset_on_seconds.csv", low_memory=True)

    mp_genetic_utils.GeneticAlgo.verbosity_level = 4
    genetic_algo = mp_genetic_utils. GeneticAlgo(df.copy(), max_features=38, population_bag_size=10)
    pop_bag = genetic_algo.initialize_population()
    generation_number = 10
    for generation in range(generation_number):
        print(f"Generation {generation} is started!")
        
        res = genetic_algo.eval_fit_population(pop_bag)
        best_fit, _, best_solution, f1_score = genetic_algo.find_best(res).values()
        
        if (generation == 0):
            best_fit_global      = best_fit
            best_solution_global = best_solution
        else:
            if (best_fit >= best_fit_global):
                best_fit_global      = best_fit
                best_solution_global = best_solution

        new_pop_bag = []
        for i in range(len(genetic_algo.population_bag)):
                # Pick 2 parents from the bag
            pA = genetic_algo.pick_one(pop_bag)
            pB = genetic_algo.pick_one(pop_bag)
            new_element = pA
            # Crossover the parents
            if rnd.random() <= 0.87:
                new_element = genetic_algo.crossover(pA, pB)
            # Mutate the child
            # if rnd.random() <= 0.7:
            #     new_element = mutation(new_element) 
            
            # Append the child to the bag
            new_pop_bag.append(new_element)
            # Set the new bag as the population bag
        pop_bag = genetic_algo.create_population(new_pop_bag)

    print("\n\n**** Generations Over ****\n")
    print(f"Best Fitness: {best_fit_global}")
    print(f"Best Solution: {best_solution_global}")
    print(f"F1-Score: {f1_score}")
