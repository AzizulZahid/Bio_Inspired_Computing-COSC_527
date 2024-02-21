import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
import time
import shutil

class EaCalculator():
    def __init__(self, pop_size, mut_prob, cross_prob, tour_size, iteration, fn="Main_data.csv", exp_dir="experiments"):
        self.fn = fn
        self.exp_dir = 'D:/PhD/Spring_24/BIC_EEE_527/Programing_Assignment/Evolutionary_Algorithm/experiments/data/'
        self.main = open(self.exp_dir + self.fn, 'a')
        self.main.write("N,P_m,P_c,Tournament Size,Iteration,Generation,Average Fitness,Best Fitness,Best Genome, Solution Found (0 or 1), Total Number of Solutions Found, Diversity Metric\n")
        self.df = pd.read_csv('D:\PhD\Spring_24\BIC_EEE_527\Programing_Assignment\Evolutionary_Algorithm\experiments\data\m_temp_file.csv')
        self.pop_size = pop_size
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob
        self.tour_size = tour_size
        self.iteration = iteration

    def calculate(self):
        # f = open('temp_cal.csv', "w")

        # if not os.path.exists(self.exp_dir):
        #     os.makedirs(self.exp_dir)

        # f.write("N,P_m,P_c,Tournament Size,Iteration,Generation,Average Fitness,Best Fitness,Best Genome, Solution Found (0 or 1), Total Number of Solutions Found, Diversity Metric\n")
            
        # f.write("Generation,Average Fitness,Best Fitness,Best Genome, Solution Found (0 or 1), Total Number of Solutions Found, Diversity Metric\n")

        generations = 30
        for generation in range(generations):

            Avg_fitness, Best_fitness, Best_genome, Solution, Total_num_solution = self.Calculate_avg_best_fitness(str(generation))
            Diversity_metric = self.Calculate_diversity_metric(str(generation))

            self.main.write(str(self.pop_size) + "," + str(self.mut_prob) + "," + str(self.cross_prob) + "," + str(self.tour_size) + "," + str(self.iteration) + "," + str(generation) + "," + str(Avg_fitness) + "," + str(Best_fitness) + "," + str(Best_genome) + "," + str(Solution) + "," + str(Total_num_solution) + "," + str(Diversity_metric) + "\n")
        
    def Calculate_avg_best_fitness(self, generation):

        df = self.df[self.df['step'] == generation]
        fitness_values = df['fitness']
        fitness_values = pd.to_numeric(fitness_values, errors='coerce')
        best_genome = df['genome'].max()
        binary = self.string_to_binary(best_genome)
        occurrences = df[df['fitness'] == '1.0'].shape[0]

        if occurrences>0:
            solution = 'yes'
        else:
            solution = 'no'

        return fitness_values.mean(), fitness_values.max(), binary, solution, occurrences
    
    def string_to_binary(self, string):
        boolean_list = eval(string)
        binary = ''.join('1' if boolean else '0' for boolean in boolean_list)
        return binary
    
    def Calculate_diversity_metric(self, generation):
        df = self.df[self.df['step'] == generation]
        list = df['genome'].tolist()
        num = len(list)
        avg_distance = []

        for i in range(num):
            distances = []
            par_gen = self.string_to_1D_array(list[i])

            for j in range(num):
                if j != i:
                    next_gen = self.string_to_1D_array(list[j])
                    distances.append(np.linalg.norm(par_gen-next_gen))
            avg_distance.append(sum(distances)/ (num-1))

        return sum(avg_distance)/num
    
    def string_to_1D_array(self, string):
        boolean_list = eval(string)
        list = [1 if boolean else 0 for boolean in boolean_list]
        return np.array(list)


# EA_cal = EaCalculator()
# EA_cal.calculate()