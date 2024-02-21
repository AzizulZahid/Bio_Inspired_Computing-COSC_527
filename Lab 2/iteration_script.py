import subprocess
import shutil
from EA_calculator import EaCalculator
import time
import os
import numpy as np
from toolz import pipe

from leap_ec import Individual, context, test_env_var
from leap_ec import ops, probe, util
from leap_ec.decoder import IdentityDecoder
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.binary_rep.problems import ScalarProblem
import argparse
import sys

class Lab2Problem(ScalarProblem):
    def __init__(self):
        super().__init__(maximize=True)
        
    def evaluate(self, ind):
        binary_array = ind.astype(int)
        ind = ''.join(map(str, binary_array))
        f = (int(ind,2) / ((2**len(ind))-1))**10
        return f

pop_sizes = [25, 50, 75, 100]
mut_probs = [0, 0.01, 0.03, 0.05]
cross_probs = [0, 0.1, 0.3, 0.5]
tour_sizes = [2, 3, 4, 5]
iterations = 20

max_generation = 30
l = 40
start_time = time.time()
path = 'D:/PhD/Spring_24/BIC_EEE_527/Programing_Assignment/Evolutionary_Algorithm/experiments/data/'

for pop_size in pop_sizes:
    for mut_prob in mut_probs:
        for cross_prob in cross_probs:
            for tour_size in tour_sizes:
                # command = ["python", ".\LAB_2\lab2-1.py", "--n", str(pop_size), "--p_m", str(mut_prob), "--p_c", str(cross_prob), "--trn_size", str(tour_size), "--csv_output", str(pop_size)+str(mut_prob)+str(cross_prob)+str(tour_size)+".csv"]
                for iteration in range(iterations):
                    # subprocess.run(command, shell=False)
                    # ea = EaCalculator(pop_size, mut_prob, cross_prob, tour_size, iteration)
                    # ea.calculate()
                    parents = Individual.create_population(pop_size,
                                           initialize=create_binary_sequence(
                                               l),
                                           decoder=IdentityDecoder(),
                                           problem=Lab2Problem())

                        # Evaluate initial population
                    parents = Individual.evaluate_population(parents)

                    generation_counter = util.inc_generation()
                    # out_f = open(path + str(pop_size)+ '_'+str(mut_prob)+'_'+str(cross_prob)+'_'+str(tour_size)+'_'+str(iteration)+".csv", "w")
                    out_f = open(path + "m_temp_file.csv", "w")
                    while generation_counter.generation() < max_generation:
                        offspring = pipe(parents,
                                        ops.tournament_selection(k=tour_size),
                                        ops.clone,
                                        mutate_bitflip(probability=mut_prob),
                                        ops.UniformCrossover(p_xover=cross_prob),
                                        ops.evaluate,
                                        ops.pool(size=len(parents)),  # accumulate offspring
                                        probe.AttributesCSVProbe(stream=out_f, do_fitness=True, do_genome=True)
                                        )
                        
                        parents = offspring
                        generation_counter()  # increment to the next generation
                    out_f.close()
                    ea = EaCalculator(pop_size, mut_prob, cross_prob, tour_size, iteration)
                    ea.calculate()

end_time = time.time()
print(end_time-start_time)                   