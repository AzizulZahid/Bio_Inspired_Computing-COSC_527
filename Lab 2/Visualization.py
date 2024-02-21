import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class plot():
    def __init__(self):
        self.df = pd.read_csv('D:\PhD\Spring_24\BIC_EEE_527\Programing_Assignment\Evolutionary_Algorithm\experiments\data\Main_data.csv', skipinitialspace=True)
        self.columns_to_convert = ['N','P_m', 'P_c', 'Tournament Size', 'Generation', 'Average Fitness', 'Best Fitness', 'Total Number of Solutions Found', 'Diversity Metric'] 
        self.df[self.columns_to_convert] = self.df[self.columns_to_convert].apply(pd.to_numeric, errors='coerce')
        
        self.path = "D:\PhD\Spring_24\BIC_EEE_527\Programing_Assignment\Evolutionary_Algorithm\experiments\Images/"
        self.gen_num = 30

    def plot_all(self):
        # self.population_avg_fitness()
        # self.population_best_fitness()
        # self.mutation_avg_fitness()
        # self.mutation_best_fitness()
        # self.crossover_avg_fitness()
        # self.crossover_best_fitness()
        # self.tournament_avg_fitness()
        # self.tournament_best_fitness()
        # self.population_diversity()
        # self.mutation_diveristy()
        # self.crossover_diversity()
        # self.tournament_diversity()
        # self.selection_pressure_solutions()
        # self.mutation_prob_solutions()
        # self.population_solutions()
        # self.crossover_solutions()
        # self.crossover_mutation_solutions()
        # self.population_no_mutation_avg_fitness()
        # self.crossover_no_mutation_solutions()
        self.mutation_no_crossover_solutions()

    def population_avg_fitness(self):

        pop = [25, 50, 75, 100]
        colors = ['k', 'g', 'b', 'm']
        colors_dash = ['b-', 'g-', 'r-', 'y-']


        we = self.df[(self.df['P_m']==0.01) & (self.df['P_c']==0.3) & (self.df['Tournament Size']==2)]
        g = 0
        x = np.arange(self.gen_num)
        for i in pop:
            er = we[we['N'] == i]
            mean_1 = []
            std_1 = []
            for k in range(self.gen_num):
                op = er[er['Generation'] == k]
                mean_1.append((op['Average Fitness'].mean()))
                std_1.append((op['Average Fitness'].std()))
            mean_1 = np.array(mean_1)
            std_1 = np.array(std_1)
            plt.plot(x, mean_1, colors[g], label=str(i), linewidth=1)
            plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=colors[g], alpha=0.3)
            plt.rcParams['legend.loc'] = 4
            plt.legend(title='Population size')
            plt.title('Population vs Average Fitness', fontsize=12)
            plt.xlabel('generation', fontsize=10)
            plt.ylabel('avg fitness', fontsize=10)
            g+=1
        plt.savefig(self.path + 'Population vs Average Fitness.png')
        plt.show()

    def population_best_fitness(self):
        pop = [25, 50, 75, 100]
        colors = ['k', 'g', 'b', 'm']
        colors_dash = ['b-', 'g-', 'r-', 'y-']
        # fig, axes = plt.subplots(4, figsize=(10,14))

        we = self.df[(self.df['P_m']==0.01) & (self.df['P_c']==0.3) & (self.df['Tournament Size']==2)]
        g = 0
        x = np.arange(self.gen_num)
        for i in pop:
            er = we[we['N'] == i]
            mean_1 = []
            std_1 = []
            for k in range(self.gen_num):
                op = er[er['Generation'] == k]
                mean_1.append((op['Best Fitness'].mean()))
                std_1.append((op['Best Fitness'].std()))
            mean_1 = np.array(mean_1)
            std_1 = np.array(std_1)
            plt.plot(x, mean_1, colors[g], label=str(i), linewidth=1)
            plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=colors[g], alpha=0.3)
            plt.legend(title='Population size')
            # plt.set_title('Population '+str(i)+', Average Fitness', fontsize=10)
            # plt.set_xlabel('generation', fontsize=8)
            # plt.set_ylabel('avg fitness', fontsize=8)
            plt.title('Population vs Best Fitness', fontsize=12)
            plt.xlabel('generation', fontsize=10)
            plt.ylabel('best fitness', fontsize=10)
            g+=1
        plt.savefig(self.path + 'Population vs best Fitness.png')
        plt.show()

    def mutation_avg_fitness(self):

        pop = [0, 0.01, 0.03, 0.05]
        colors = ['k', 'g', 'b', 'm']
        colors_dash = ['b-', 'g-', 'r-', 'y-']
        # fig, axes = plt.subplots(4, figsize=(10,14))

        we = self.df[(self.df['N']==50) & (self.df['P_c']==0.3) & (self.df['Tournament Size']==2)]
        g = 0
        x = np.arange(self.gen_num)
        for i in pop:
            er = we[we['P_m'] == i]
            mean_1 = []
            std_1 = []
            for k in range(self.gen_num):
                op = er[er['Generation'] == k]
                mean_1.append((op['Average Fitness'].mean()))
                std_1.append((op['Average Fitness'].std()))
            mean_1 = np.array(mean_1)
            std_1 = np.array(std_1)
            plt.plot(x, mean_1, colors[g], label=str(i), linewidth=1)
            plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=colors[g], alpha=0.3)
            plt.legend(title='Mutation prob')
            # plt.set_title('Population '+str(i)+', Average Fitness', fontsize=10)
            # plt.set_xlabel('generation', fontsize=8)
            # plt.set_ylabel('avg fitness', fontsize=8)
            plt.title('Mutation vs Average Fitness', fontsize=12)
            plt.xlabel('generation', fontsize=10)
            plt.ylabel('avg fitness', fontsize=10)
            g+=1
        plt.savefig(self.path + 'Mutation vs Average Fitness.png')
        plt.show()

    def mutation_best_fitness(self):

        pop = [0, 0.01, 0.03, 0.05]
        colors = ['k', 'g', 'b', 'm']
        colors_dash = ['b-', 'g-', 'r-', 'y-']
        # fig, axes = plt.subplots(4, figsize=(10,14))

        we = self.df[(self.df['N']==50) & (self.df['P_c']==0.3) & (self.df['Tournament Size']==2)]
        g = 0
        x = np.arange(self.gen_num)
        for i in pop:
            er = we[we['P_m'] == i]
            mean_1 = []
            std_1 = []
            for k in range(self.gen_num):
                op = er[er['Generation'] == k]
                mean_1.append((op['Best Fitness'].mean()))
                std_1.append((op['Best Fitness'].std()))
            mean_1 = np.array(mean_1)
            std_1 = np.array(std_1)
            plt.plot(x, mean_1, colors[g], label=str(i), linewidth=1)
            plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=colors[g], alpha=0.3)
            plt.legend(title='Mutation prob')
            # plt.set_title('Population '+str(i)+', Average Fitness', fontsize=10)
            # plt.set_xlabel('generation', fontsize=8)
            # plt.set_ylabel('avg fitness', fontsize=8)
            plt.title('Mutation vs Best Fitness', fontsize=12)
            plt.xlabel('generation', fontsize=10)
            plt.ylabel('best fitness', fontsize=10)
            g+=1
        plt.savefig(self.path + 'Mutation vs Best Fitness.png')
        plt.show()

    def crossover_avg_fitness(self):

        pop = [0, 0.1, 0.3, 0.5]
        colors = ['k', 'g', 'b', 'm']
        colors_dash = ['b-', 'g-', 'r-', 'y-']
        # fig, axes = plt.subplots(4, figsize=(10,14))

        we = self.df[(self.df['N']==50) & (self.df['P_m']==0.01) & (self.df['Tournament Size']==2)]
        g = 0
        x = np.arange(self.gen_num)
        for i in pop:
            er = we[we['P_c'] == i]
            mean_1 = []
            std_1 = []
            for k in range(self.gen_num):
                op = er[er['Generation'] == k]
                mean_1.append((op['Average Fitness'].mean()))
                std_1.append((op['Average Fitness'].std()))
            mean_1 = np.array(mean_1)
            std_1 = np.array(std_1)
            plt.plot(x, mean_1, colors[g], label=str(i), linewidth=1)
            plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=colors[g], alpha=0.3)
            plt.legend(title='Crossover prob')
            # plt.set_title('Population '+str(i)+', Average Fitness', fontsize=10)
            # plt.set_xlabel('generation', fontsize=8)
            # plt.set_ylabel('avg fitness', fontsize=8)
            plt.title('Crossover vs Average Fitness', fontsize=12)
            plt.xlabel('generation', fontsize=10)
            plt.ylabel('avg fitness', fontsize=10)
            g+=1
        plt.savefig(self.path + 'Crossover vs Average Fitness.png')
        plt.show()

    def crossover_best_fitness(self):

        pop = [0, 0.1, 0.3, 0.5]
        colors = ['k', 'g', 'b', 'm']
        colors_dash = ['b-', 'g-', 'r-', 'y-']
        # fig, axes = plt.subplots(4, figsize=(10,14))

        we = self.df[(self.df['N']==50) & (self.df['P_m']==0.01) & (self.df['Tournament Size']==2)]
        g = 0
        x = np.arange(self.gen_num)
        for i in pop:
            er = we[we['P_c'] == i]
            mean_1 = []
            std_1 = []
            for k in range(self.gen_num):
                op = er[er['Generation'] == k]
                mean_1.append((op['Best Fitness'].mean()))
                std_1.append((op['Best Fitness'].std()))
            mean_1 = np.array(mean_1)
            std_1 = np.array(std_1)
            plt.plot(x, mean_1, colors[g], label=str(i), linewidth=1)
            plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=colors[g], alpha=0.3)
            plt.legend(title='Crossover prob')
            # plt.set_title('Population '+str(i)+', Average Fitness', fontsize=10)
            # plt.set_xlabel('generation', fontsize=8)
            # plt.set_ylabel('avg fitness', fontsize=8)
            plt.title('Crossover vs Best Fitness', fontsize=12)
            plt.xlabel('generation', fontsize=10)
            plt.ylabel('best fitness', fontsize=10)
            g+=1
        plt.savefig(self.path + 'Crossover vs Best Fitness.png')
        plt.show()

    def tournament_avg_fitness(self):

        pop = [2, 3, 4, 5]
        colors = ['k', 'g', 'b', 'm']
        colors_dash = ['b-', 'g-', 'r-', 'y-']
        # fig, axes = plt.subplots(4, figsize=(10,14))

        we = self.df[(self.df['N']==50) & (self.df['P_m']==0.01) & (self.df['P_c']==0.3)]
        g = 0
        x = np.arange(self.gen_num)
        for i in pop:
            er = we[we['Tournament Size'] == i]
            mean_1 = []
            std_1 = []
            for k in range(self.gen_num):
                op = er[er['Generation'] == k]
                mean_1.append((op['Average Fitness'].mean()))
                std_1.append((op['Average Fitness'].std()))
            mean_1 = np.array(mean_1)
            std_1 = np.array(std_1)
            plt.plot(x, mean_1, colors[g], label=str(i), linewidth=1)
            plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=colors[g], alpha=0.3)
            plt.legend(title='Tournament size')
            # plt.set_title('Population '+str(i)+', Average Fitness', fontsize=10)
            # plt.set_xlabel('generation', fontsize=8)
            # plt.set_ylabel('avg fitness', fontsize=8)
            plt.title('Tournament vs Average Fitness', fontsize=12)
            plt.xlabel('generation', fontsize=10)
            plt.ylabel('avg fitness', fontsize=10)
            g+=1
        plt.savefig(self.path + 'Tournament size vs Average Fitness.png')
        plt.show()

    def tournament_best_fitness(self):

        pop = [2, 3, 4, 5]
        colors = ['k', 'g', 'b', 'm']
        colors_dash = ['b-', 'g-', 'r-', 'y-']
        # fig, axes = plt.subplots(4, figsize=(10,14))

        we = self.df[(self.df['N']==50) & (self.df['P_m']==0.01) & (self.df['P_c']==0.3)]
        g = 0
        x = np.arange(self.gen_num)
        for i in pop:
            er = we[we['Tournament Size'] == i]
            mean_1 = []
            std_1 = []
            for k in range(self.gen_num):
                op = er[er['Generation'] == k]
                mean_1.append((op['Best Fitness'].mean()))
                std_1.append((op['Best Fitness'].std()))
            mean_1 = np.array(mean_1)
            std_1 = np.array(std_1)
            plt.plot(x, mean_1, colors[g], label=str(i), linewidth=1)
            plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=colors[g], alpha=0.3)
            plt.legend(title='Tournament size')
            # plt.set_title('Population '+str(i)+', Average Fitness', fontsize=10)
            # plt.set_xlabel('generation', fontsize=8)
            # plt.set_ylabel('avg fitness', fontsize=8)
            plt.title('Tournament vs Best Fitness', fontsize=12)
            plt.xlabel('generation', fontsize=10)
            plt.ylabel('best fitness', fontsize=10)
            g+=1
        plt.savefig(self.path + 'Tournament size vs Best Fitness.png')
        plt.show()

    def population_diversity(self):

        pop = [25, 50, 75, 100]
        colors = ['k', 'g', 'b', 'm']
        colors_dash = ['b-', 'g-', 'r-', 'y-']
        # fig, axes = plt.subplots(4, figsize=(10,14))

        we = self.df[(self.df['P_m']==0.01) & (self.df['P_c']==0.3) & (self.df['Tournament Size']==2)]
        g = 0
        x = np.arange(self.gen_num)
        for i in pop:
            er = we[we['N'] == i]
            mean_1 = []
            std_1 = []
            for k in range(self.gen_num):
                op = er[er['Generation'] == k]
                mean_1.append((op['Diversity Metric'].mean()))
                std_1.append((op['Diversity Metric'].std()))
            mean_1 = np.array(mean_1)
            std_1 = np.array(std_1)
            plt.plot(x, mean_1, colors[g], label=str(i), linewidth=1)
            plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=colors[g], alpha=0.3)
            plt.rcParams['legend.loc'] = 4
            plt.legend(title='Population size')
            # plt.set_title('Population '+str(i)+', Average Fitness', fontsize=10)
            # plt.set_xlabel('generation', fontsize=8)
            # plt.set_ylabel('avg fitness', fontsize=8)
            plt.title('Population vs Diversity', fontsize=12)
            plt.xlabel('generation', fontsize=10)
            plt.ylabel('diversity', fontsize=10)
            g+=1
        plt.savefig(self.path + 'Population vs Diversity.png')
        plt.show()


    def mutation_diveristy(self):

        pop = [0, 0.01, 0.03, 0.05]
        colors = ['k', 'g', 'b', 'm']
        colors_dash = ['b-', 'g-', 'r-', 'y-']
        # fig, axes = plt.subplots(4, figsize=(10,14))

        we = self.df[(self.df['N']==50) & (self.df['P_c']==0.3) & (self.df['Tournament Size']==2)]
        g = 0
        x = np.arange(self.gen_num)
        for i in pop:
            er = we[we['P_m'] == i]
            mean_1 = []
            std_1 = []
            for k in range(self.gen_num):
                op = er[er['Generation'] == k]
                mean_1.append((op['Diversity Metric'].mean()))
                std_1.append((op['Diversity Metric'].std()))
            mean_1 = np.array(mean_1)
            std_1 = np.array(std_1)
            plt.plot(x, mean_1, colors[g], label=str(i), linewidth=1)
            plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=colors[g], alpha=0.3)
            plt.legend(title='Mutation prob')
            # plt.set_title('Population '+str(i)+', Average Fitness', fontsize=10)
            # plt.set_xlabel('generation', fontsize=8)
            # plt.set_ylabel('avg fitness', fontsize=8)
            plt.title('Mutation vs Diversity', fontsize=12)
            plt.xlabel('generation', fontsize=10)
            plt.ylabel('diversity', fontsize=10)
            g+=1
        plt.savefig(self.path + 'Mutation vs Diversity.png')
        plt.show()


    def crossover_diversity(self):

        pop = [0, 0.1, 0.3, 0.5]
        colors = ['k', 'g', 'b', 'm']
        colors_dash = ['b-', 'g-', 'r-', 'y-']
        # fig, axes = plt.subplots(4, figsize=(10,14))

        we = self.df[(self.df['N']==50) & (self.df['P_m']==0.01) & (self.df['Tournament Size']==2)]
        g = 0
        x = np.arange(self.gen_num)
        for i in pop:
            er = we[we['P_c'] == i]
            mean_1 = []
            std_1 = []
            for k in range(self.gen_num):
                op = er[er['Generation'] == k]
                mean_1.append((op['Diversity Metric'].mean()))
                std_1.append((op['Diversity Metric'].std()))
            mean_1 = np.array(mean_1)
            std_1 = np.array(std_1)
            plt.plot(x, mean_1, colors[g], label=str(i), linewidth=1)
            plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=colors[g], alpha=0.3)
            plt.legend(title='Crossover prob')
            # plt.set_title('Population '+str(i)+', Average Fitness', fontsize=10)
            # plt.set_xlabel('generation', fontsize=8)
            # plt.set_ylabel('avg fitness', fontsize=8)
            plt.title('Crossover vs Diversity', fontsize=12)
            plt.xlabel('generation', fontsize=10)
            plt.ylabel('diversity', fontsize=10)
            g+=1
        plt.savefig(self.path + 'Crossover vs Diversity.png')
        plt.show()

    def tournament_diversity(self):

        pop = [2, 3, 4, 5]
        colors = ['k', 'g', 'b', 'm']
        colors_dash = ['b-', 'g-', 'r-', 'y-']
        # fig, axes = plt.subplots(4, figsize=(10,14))

        we = self.df[(self.df['N']==50) & (self.df['P_m']==0.01) & (self.df['P_c']==0.3)]
        g = 0
        x = np.arange(self.gen_num)
        for i in pop:
            er = we[we['Tournament Size'] == i]
            mean_1 = []
            std_1 = []
            for k in range(self.gen_num):
                op = er[er['Generation'] == k]
                mean_1.append((op['Diversity Metric'].mean()))
                std_1.append((op['Diversity Metric'].std()))
            mean_1 = np.array(mean_1)
            std_1 = np.array(std_1)
            plt.plot(x, mean_1, colors[g], label=str(i), linewidth=1)
            plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=colors[g], alpha=0.3)
            plt.legend(title='Tournament size')
            # plt.set_title('Population '+str(i)+', Average Fitness', fontsize=10)
            # plt.set_xlabel('generation', fontsize=8)
            # plt.set_ylabel('avg fitness', fontsize=8)
            plt.title('Tournament vs Diversity', fontsize=12)
            plt.xlabel('generation', fontsize=10)
            plt.ylabel('diveristy', fontsize=10)
            g+=1
        plt.savefig(self.path + 'Tournament size vs Diversity.png')
        plt.show()

    def selection_pressure_solutions(self):
        bar_colors = ['red', 'green', 'blue', 'orange']
        we = self.df[self.df['Total Number of Solutions Found'] > 0.0]
        x = np.arange(4)
        pop = [2, 3, 4, 5]
        num_sol = []
        for i in pop:
            num_sol.append(we[we['Tournament Size']==i]['Total Number of Solutions Found'].sum())
            
        plt.bar(x, height=num_sol, color=bar_colors)
        plt.xticks(x, pop)
        plt.title('Selection pressure vs Number of solutions', fontsize=12)
        plt.xlabel('Selection pressure', fontsize=10)
        plt.ylabel('Number of solutions', fontsize=10)
        for i, height in enumerate(num_sol):
            plt.text(x[i], height+5, str(int(height)), ha='center', va='center')

        plt.savefig(self.path + 'Selection pressure vs Number of solutions')
        plt.show()

    def mutation_prob_solutions(self):
        bar_colors = ['red', 'green', 'blue', 'orange']
        we = self.df[self.df['Total Number of Solutions Found'] > 0.0]
        x = np.arange(4)
        pop = [0, 0.01, 0.03, 0.05]
        num_sol = []
        for i in pop:
            num_sol.append(we[we['P_m']==i]['Total Number of Solutions Found'].sum())
            
        plt.bar(x, height=num_sol, color=bar_colors)
        plt.xticks(x, pop)
        plt.title('Mutation probability vs Number of solutions', fontsize=12)
        plt.xlabel('Mutation probs', fontsize=10)
        plt.ylabel('Number of solutions', fontsize=10)
        for i, height in enumerate(num_sol):
            plt.text(x[i], height+5, str(int(height)), ha='center', va='center')

        plt.savefig(self.path + 'Mutation probability vs Number of solutions')
        plt.show()

    def population_solutions(self):
        bar_colors = ['red', 'green', 'blue', 'orange']
        we = self.df[self.df['Total Number of Solutions Found'] > 0.0]
        x = np.arange(4)
        pop = [25, 50, 75, 100]
        num_sol = []
        for i in pop:
            num_sol.append(we[we['N']==i]['Total Number of Solutions Found'].sum())
            
        plt.bar(x, height=num_sol, color=bar_colors)
        plt.xticks(x, pop)
        plt.title('Population vs Number of solutions', fontsize=12)
        plt.xlabel('Populations', fontsize=10)
        plt.ylabel('Number of solutions', fontsize=10)
        for i, height in enumerate(num_sol):
            plt.text(x[i], height+5, str(int(height)), ha='center', va='center')

        plt.savefig(self.path + 'Population vs Number of solutions')
        plt.show()

    def crossover_solutions(self):
        bar_colors = ['red', 'green', 'blue', 'orange']
        we = self.df[self.df['Total Number of Solutions Found'] > 0.0]
        x = np.arange(4)
        pop = [0, 0.1, 0.3, 0.5]
        num_sol = []
        for i in pop:
            num_sol.append(we[we['P_c']==i]['Total Number of Solutions Found'].sum())
            
        plt.bar(x, height=num_sol, color=bar_colors)
        plt.xticks(x, pop)
        plt.title('Crossover probability vs Number of solutions', fontsize=12)
        plt.xlabel('Crossover probs', fontsize=10)
        plt.ylabel('Number of solutions', fontsize=10)
        for i, height in enumerate(num_sol):
            plt.text(x[i], height+5, str(int(height)), ha='center', va='center')

        plt.savefig(self.path + 'Crossover probability vs Number of solutions')
        plt.show()


    def crossover_mutation_solutions(self):
        bar_colors = ['red', 'green', 'blue', 'orange']
        we = self.df[self.df['Total Number of Solutions Found'] > 0.0]
        x = np.arange(4)
        pop1 = [0.01, 0.03, 0.05]
        pop2 = [0.1, 0.3, 0.5]
        pop = ['All solutions', 'Any mutation', 'Any crossover', 'No mutation or no crossover']
        num_sol = []
        num_sol.append(we['Total Number of Solutions Found'].sum())
        num_sol.append(we[we['P_m'] > 0]['Total Number of Solutions Found'].sum())
        num_sol.append(we[we['P_c']> 0]['Total Number of Solutions Found'].sum())
        num_sol.append(we[(we['P_m']==0) | (we['P_c']==0)]['Total Number of Solutions Found'].sum())
        # for i in pop:
        #     num_sol.append(we[we['P_c']==i]['Total Number of Solutions Found'].sum())
            
        plt.bar(x, height=num_sol, color=bar_colors)
        plt.xticks(x, pop, fontsize=7)
        plt.title('Crossover_mutation vs Number of solutions', fontsize=12)
        plt.xlabel('Crossover_mutation probs', fontsize=10)
        plt.ylabel('Number of solutions', fontsize=10)
        for i, height in enumerate(num_sol):
            plt.text(x[i], height+5, str(int(height)), ha='center', va='center')

        plt.savefig(self.path + 'Crossover_mutation probability vs Number of solutions')
        plt.show()

    def population_no_mutation_avg_fitness(self):

        pop = [25, 50, 75, 100]
        colors = ['k', 'g', 'b', 'm']
        colors_dash = ['b-', 'g-', 'r-', 'y-']
        # fig, axes = plt.subplots(4, figsize=(10,14))

        we = self.df[(self.df['P_m']==0.05) & (self.df['P_c']==0.3) & (self.df['Tournament Size']==2)]
        g = 0
        x = np.arange(self.gen_num)
        for i in pop:
            er = we[we['N'] == i]
            mean_1 = []
            std_1 = []
            for k in range(self.gen_num):
                op = er[er['Generation'] == k]
                mean_1.append((op['Average Fitness'].mean()))
                std_1.append((op['Average Fitness'].std()))
            mean_1 = np.array(mean_1)
            std_1 = np.array(std_1)
            plt.plot(x, mean_1, colors[g], label=str(i), linewidth=1)
            plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=colors[g], alpha=0.3)
            plt.rcParams['legend.loc'] = 4
            plt.legend(title='Population size')
            # plt.set_title('Population '+str(i)+', Average Fitness', fontsize=10)
            # plt.set_xlabel('generation', fontsize=8)
            # plt.set_ylabel('avg fitness', fontsize=8)
            plt.title('Population_mutation(0.05) vs Average Fitness', fontsize=12)
            plt.xlabel('generation', fontsize=10)
            plt.ylabel('avg fitness', fontsize=10)
            g+=1
        plt.savefig(self.path + 'Population_mutation(0.05) vs Average Fitness.png')
        plt.show()

    def crossover_no_mutation_solutions(self):

        bar_colors = ['red', 'green', 'blue', 'orange']
        we = self.df[(self.df['Total Number of Solutions Found'] > 0.0) & (self.df['P_m'] == 0.0)]
        x = np.arange(4)
        pop = [0, 0.1, 0.3, 0.5]
        num_sol = []
        for i in pop:
            num_sol.append(we[we['P_c']==i]['Total Number of Solutions Found'].sum())
            
        plt.bar(x, height=num_sol, color=bar_colors)
        plt.xticks(x, pop)
        plt.title('Crossover with no mutation vs Number of solutions', fontsize=12)
        plt.xlabel('Crossover probs', fontsize=10)
        plt.ylabel('Number of solutions', fontsize=10)
        for i, height in enumerate(num_sol):
            plt.text(x[i], height+5, str(int(height)), ha='center', va='center')

        plt.savefig(self.path + 'Crossover probability_no_probs vs Number of solutions')
        plt.show()

    def mutation_no_crossover_solutions(self):

        bar_colors = ['red', 'green', 'blue', 'orange']
        we = self.df[(self.df['Total Number of Solutions Found'] > 0.0) & (self.df['P_c'] == 0.0)]
        x = np.arange(4)
        pop = [0, 0.01, 0.03, 0.05]
        num_sol = []
        for i in pop:
            num_sol.append(we[we['P_m']==i]['Total Number of Solutions Found'].sum())
            
        plt.bar(x, height=num_sol, color=bar_colors)
        plt.xticks(x, pop)
        plt.title('Mutation with no crossover vs Number of solutions', fontsize=12)
        plt.xlabel('Mutation probs', fontsize=10)
        plt.ylabel('Number of solutions', fontsize=10)
        for i, height in enumerate(num_sol):
            plt.text(x[i], height+5, str(int(height)), ha='center', va='center')

        plt.savefig(self.path + 'Mutation probability_no_crossover vs Number of solutions')
        plt.show()


    def box_plots_solutions(self):
        bar_colors = ['red', 'green', 'blue', 'orange']
        we = self.df[self.df['Total Number of Solutions Found'] > 0.0]
        x = np.arange(4)
        pop1 = [0.01, 0.03, 0.05]
        pop2 = [0.1, 0.3, 0.5]
        pop = ['All solutions', 'Any mutation', 'Any crossover', 'No mutation or no crossover']
        num_sol = []
        num_sol.append(we['Total Number of Solutions Found'].sum())
        num_sol.append(we[we['P_m'] > 0]['Total Number of Solutions Found'].sum())
        num_sol.append(we[we['P_c']> 0]['Total Number of Solutions Found'].sum())
        num_sol.append(we[(we['P_m']==0) | (we['P_c']==0)]['Total Number of Solutions Found'].sum())
        # for i in pop:
        #     num_sol.append(we[we['P_c']==i]['Total Number of Solutions Found'].sum())
            
        plt.bar(x, height=num_sol, color=bar_colors)
        plt.xticks(x, pop, fontsize=7)
        plt.title('Crossover_mutation vs Number of solutions', fontsize=12)
        plt.xlabel('Crossover_mutation probs', fontsize=10)
        plt.ylabel('Number of solutions', fontsize=10)
        for i, height in enumerate(num_sol):
            plt.text(x[i], height+5, str(int(height)), ha='center', va='center')

        plt.savefig(self.path + 'Crossover_mutation probability vs Number of solutions')
        plt.show()


graph = plot()
graph.plot_all()