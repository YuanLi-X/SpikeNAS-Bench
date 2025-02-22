#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File       :main_nsga2_3.py
@Description:基于多保真评估的SNN-NAS算法
@Date       :2025/02/22 00:42:09
@Author     :li-yuan
@Email      :yuanli080800@gmail.com
'''


import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import config2
import utils2
from get_train_log2 import get_train_log
from EMO_public import P_generator, NDsort, F_distance, F_mating,F_EnvironmentSelect

def anyToDecimal(num, n):
    baseStr = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
               "a": 10, "b": 11, "c": 12, "d": 13, "e": 14, "f": 15, "g": 16, "h": 17, "i": 18, "j": 19}
    new_num = 0
    nNum = len(num) - 1
    for i in num:
        new_num = new_num + baseStr[i] * pow(n, nNum)
        nNum = nNum - 1
    return new_num


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


def solution_evaluation(args, id_index, used_epoch, allocated_epochs):
    _, _, model_param, _, _, val_acc, _, _, _, seconds = get_train_log(id_index, 'E:/master_degree/SpikeNAS-Bench/data/CIFAR10')
    val_acc_new = 100 - val_acc[used_epoch+allocated_epochs-1]
    
    return val_acc_new, model_param, 0, 0


class individual():
    def __init__(self, individual_single, args):
        self.individual_single = individual_single
        self.id_index = 0
        # 初始化适应度
        self.fitness = np.array([0.1, 0.1, 0.2, 0.2])
        # 初始化存活和淘汰统计
        self.survive_count = 0
        self.eliminate_count = 0
        
        self.encoding_to_id()
        
        self.fitness_arxiv = []
        self.used_epoch = 0
        
    def encoding_to_id(self):
        encoding_re = self.individual_single[::-1]
        id = anyToDecimal(''.join('%s' %a for a in encoding_re), 5)
        
        self.id_index = id
        

    def evaluate(self, args, EMO_epochs):
        allocated_epochs = EMO_epochs -self.used_epoch
        self.fitness = solution_evaluation(args, self.id_index, self.used_epoch, allocated_epochs)
        self.fitness_arxiv.append([self.used_epoch, self.fitness[0]])
        
        self.used_epoch = EMO_epochs
        

class NSGA2_MFENAS():
    def __init__(self, args):
        self.args = args
        self.popsize = self.args.population_size
        self.Max_Gen = self.args.generation_number
        self.Gen = 0
        
        self.epoch_now = 0
        self.Maxi_Epoch = args.search_epochs
        
        self.coding = 'Real'
        
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
        initialized_population_list = []
        while(len(initialized_population_list) < self.popsize):
            individual_single = np.random.randint(0, 5, size=6)
            flag = check_validity(individual_single)
            if type(flag) == int:
                continue
            else:
                initialized_population_list.append(individual_single)
            self.Population.append(individual(individual_single, self.args))
        
        self.variable_length = 6
        Up_boundary = np.full((self.variable_length, ), 4.5)
        Low_boundary = np.full((self.variable_length, ), -0.5)
        self.Boundary = np.vstack((Up_boundary, Low_boundary))
        
        self.Pop_fitness = self.evaluation(self.Population)
        self.fitness_best = np.min(self.Pop_fitness[:,0])
        
    def evaluation(self, population):
        fitness = np.zeros((len(population), 4))
        for i, solution in enumerate(population):
            # evaluate(self, args, EMO_epochs):
            solution.evaluate(self.args, EMO_epochs = self.epoch_now)
            fitness[i] = torch.tensor(solution.fitness, dtype=torch.double)
            
        return fitness[:, :2]
    
    def Binary_Environment_tour_selection(self):
        # 二元锦标赛杂交池选择
        print("Population:", self.Population)
        self.MatingPool, self.tour_index = F_mating.F_mating(self.Population.copy(), self.FrontValue, self.CrowdDistance)
        
    def genetic_operation(self):
        '''
        生成子代个体
        '''
        offspring_dec = P_generator.P_generator(self.MatingPool, self.Boundary, self.coding, self.popsize)
        offspring_dec = self.deduplication(offspring_dec) # 去掉重复个体
        self.offspring = [individual(i, self.args) for i in offspring_dec]
        self.off_fitness = self.evaluation(self.offspring)
    

    def deduplication(self, offspring_dec):
        '''
        去掉重复个体之后返回子代
        '''
        pop_index = [i.id_index for i in self.Population]
        dedup_offspring_index = []
        for i in offspring_dec:
            if i not in dedup_offspring_index and i not in pop_index:
                dedup_offspring_index.append(i)
        
        return dedup_offspring_index
    
    def first_selection(self):
        '''
        首次选择
        '''
        Population = []
        Population.extend(self.Population)
        Population.extend(self.offspring)
        
        Population_temp = []
        for i, solution in enumerate(Population):
            if solution.fitness[0]<self.fitness_best + self.threshold():
                Population_temp.append(solution)
        
        FunctionValue = np.zeros((len(Population_temp), 2))
        for i, solution in enumerate(Population_temp):
            FunctionValue[i] = solution.fitness[:2]
        
        return Population, FunctionValue
    
    # 对当前种群和淘汰个体进行状态更新，记录每个个体生存和淘汰的计数
    def process_population_eliminate(self):
        self.finess_best = np.min(self.Pop_fitness[:,0])

        for i, solution in enumerate(self.Population):
            if solution.survive_count==0:
                solution.survive_count =1
        for i, solution in enumerate(self.eliminate):
            if solution.eliminate_count == 0:
                solution.eliminate_count = 1
    
    # 进行个体淘汰选择（从淘汰的个体中筛选出来一些个体，最终保留一定数量的个体进入后续操作）
    def eliminate_selection(self):
        Temp_eliminate = self.eliminate.copy()
        self.eliminate = []
        survive_eliminate_count = []
        for i, solution in enumerate(Temp_eliminate):
            # if solution.fitness[0]<self.finess_best + self.threshold():
            if solution.fitness[1] > 0.035 and solution.fitness[1] < 0.11:
                self.eliminate.append(solution)
                survive_eliminate_count.append([solution,solution.survive_count, solution.eliminate_count,solution.fitness[1]])
                
        if len(self.eliminate)>self.eliminate_size:
            survive_eliminate_count.sort(key=lambda x: (-x[1], x[2], -x[3]))
            del Temp_eliminate
            Temp_eliminate = []
            i = 0
            while i<self.eliminate_size:
                j = i
                if i == self.eliminate_size-10:
                    survive_eliminate_count.sort(key=lambda x: (x[2], x[1], -x[3]))
                    t_i = i
                if i >=self.eliminate_size-10:
                    j = i-t_i
                Temp_eliminate.append(survive_eliminate_count[j][0])
                i +=1
            self.eliminate = Temp_eliminate.copy()
        del Temp_eliminate
        
    
    def Environment_selection(self):
        Population, FunctionValue = self.first_selection()
        total_list = list(range(len(Population)))
        Temp_Pop = Population.copy()
        Temp_Fitness = FunctionValue.copy()
        
        # 调用外部选择方法来进行环境选择
        Population_, FunctionValue_, FrontValue, CrowdDistance,select_index = F_EnvironmentSelect.F_EnvironmentSelect(Population.copy(), FunctionValue.copy(), self.popsize)
        
        self.FrontValue = FrontValue
        self.CrowdDistance = CrowdDistance
        self.Population = Population_
        self.Pop_fitness = FunctionValue_
        
        # 去除被淘汰的给体，保存被选择的个体的索引，将被淘汰的个体从原始种群中移除，存储到self.eliminate列表中
        self.select_index = select_index
        self.elimitate_list = list(set(total_list) ^ set(select_index))
        self.eliminate.extend([Temp_Pop[i] for i in self.elimitate_list])
        
        # 进一步处理被淘汰的个体，并执行与淘汰个体相关的操作，清理临时数据，释放内存
        self.process_population_eliminate()
        self.eliminate_selection()
        del Temp_Pop, Temp_Fitness
        
        if self.epoch_flag():

            self.environment_change()
    
    # 合并当前种群和淘汰个体，成为一个临时种群，准备重新评估并进行选择。从合并后的种群中选择出来最优秀的个体，更新当前种群。
    def environment_change(self):
        #1.combine
        #2.re-evaluation
        #3.selection()
        #4. survive_eliminate count

        Temp_Pop = []
        Temp_Pop.extend(self.Population)
        Temp_Pop.extend(self.eliminate)
        Temp_Fitness = self.evaluation(Temp_Pop)

        total_list = list(range(len(Temp_Pop)))

        Population_, FunctionValue_, FrontValue, CrowdDistance, select_index = F_EnvironmentSelect.F_EnvironmentSelect(Temp_Pop.copy(), Temp_Fitness.copy(), self.popsize)

        self.FrontValue = FrontValue
        self.CrowdDistance = CrowdDistance

        self.Population = Population_
        self.Pop_fitness = FunctionValue_

        self.select_index = select_index
        self.elimitate_list = list(set(total_list) ^ set(select_index))

        self.eliminate = []
        self.eliminate.extend([Temp_Pop[i] for i in self.elimitate_list])

        for i, solution in enumerate(self.Population):
            solution.survive_count += 1
        for i, solution in enumerate(self.eliminate):
            solution.eliminate_count += 1
        pass
    
    def epoch_flag(self):

        if self.Gen==19:
            self.epoch_now = self.Maxi_Epoch
            return True
        elif self.Gen>19:
            self.epoch_now = self.Maxi_Epoch
            return False
        elif (self.Gen+1)%3==0:
            self.epoch_now += 2
            return True
        else:
            return False
        
    def threshold(self):
        if self.Gen<5:
            return 0.12
        elif self.Gen<10:
            return 0.1
        elif self.Gen<15:
            return 0.08
        else:
            return 0.06
        
        
        
    def Main_loop(self):
        self.initialization()
        # 非支配排序
        self.FrontValue = NDsort.NDSort(self.Pop_fitness, self.popsize)[0]
        # 拥挤距离计算
        self.CrowdDistance = F_distance.F_distance(self.Pop_fitness, self.FrontValue)
        
        while self.Gen < self.Max_Gen:
            # 二元锦标赛选择
            self.Binary_Environment_tour_selection()
            # 基因操作
            self.genetic_operation()
            # 环境选择
            self.Environment_selection()
            
            self.Gen += 1
        
        
    
         

if __name__ == '__main__':
    args = config2.get_args()
    MFENAS = NSGA2_MFENAS(args)
    MFENAS.Main_loop()