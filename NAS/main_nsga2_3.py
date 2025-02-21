import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import config

args = config.get_args()


def check_validity(coding):
    count_vector = coding.ravel()
    con_mat = np.zeros((4, 4))
    position = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    flag2 = 1
    flag3 = 1

    for num, (k0, k1) in enumerate(position):
        con_mat[k0, k1] = count_vector[num]

    neigh2_cnts = con_mat @ con_mat
    neigh3_cnts = neigh2_cnts @ con_mat
    neigh4_cnts = neigh3_cnts @ con_mat
    connection_graph = con_mat + neigh2_cnts + neigh3_cnts + neigh4_cnts

    for node in range(3):
        if connection_graph[node, 3] == 0:  # if any node doesnt send information to the last layer, remove it
            flag2 = 0
        if flag2 == 0: return -1

    for node in range(3):
        if connection_graph[0, node + 1] == 0:  # if any node doesnt get information from the input layer, remove it
            flag3 = 0
        if flag3 == 0: return -1

    # 保证node1到node4一定有连接
    if con_mat[0, 3] == 0: return -1  # ensure direct connection between input=>output for fast information propagation

    return con_mat

class individual():
    def __init__(self, args):
        self.fitness_arxiv = []

    def evaluate(self, args, EMO_epochs):
        self.fitness = solution_evaluation()
        self.fitness_arxiv.append([self.used_epoch, self.fitness[0]])
        
        self.used_epoch = EMO_epochs
        

class NSGA2_MFENAS():
    def __init__(self, args):
        self.args = args
        self.Gen = 0
        self.popsize = self.args.population_size
        self.Max_Gen = self.args.generation_number
        
        self.Population = []
        self.Pop_fitness = []
        self.fitness_best = 0
        
        self.offspring = []
        self.off_fitness = []
        
        self.eliminate = []
        self.eliminate_fitness = np.empty((0, 2))
        self.eliminate_size = 2*self.popsize
        
        self.tour_index = []
        self.FrontValue = []
        self.CrowdDistace = []
        self.selec_index = []
        
    def initialization(self):
        initialized_population = []
        while(len(initialized_population) < self.popsize):
            individual = np.random.randint(0, 5, size=6)
            flag = check_validity(individual)
            if type(flag) == int:
                continue
            else:
                initialized_population.append(individual)

        self.Population.append(initialized_population)
        self.Pop_fitness = self.evaluation(self.Population)
        
        self.fitness_best = np.min(self.Pop_fitness[:,0])
        
    def evaluation(self, population):
        fitness = np.zeros((len(population), 2))
        for i, solution in enumerate(population):
            solution.evaluate(self.args, EMO_epochs = self.epoch_now)
            fitness[i] = torch.tensor(solution.fitness, dtype=torch.double)
            
        return fitness[:, :2]
    
    def Binary_Environment_tour_selection(self):
        # 二元锦标赛杂交池选择
        self.MatingPool, self.tour_index = F_mating.F_mating(self.Population.copy(), self.FrontValue, self.CrowdDistance)
        
    def genetic_operation(self):
        # 生成子代个体
        offspring_dec = 
        
    def Main_loop(self):
        self.initialization()
        # 非支配排序
        self.FrontValue = 
        # 拥挤距离计算
        self.CrowdDistace = 
        
        while self.Gen < self.Max_Gen:
            # 二元锦标赛选择
            self.Binary_Environment_tour_selection()
            # 基因操作
            self.genetic_operation()
            # 环境选择
            self.Environment_selection()
            
            self.Gen += 1
        
        
    
        
        
    
    

if __name__ == '__main__':
    MFENAS = NSGA2_MFENAS(args)
    MFENAS.Main_loop()