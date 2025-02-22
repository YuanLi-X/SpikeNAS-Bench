# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-
# '''
# @File       :main_nsga2_3.py
# @Description:基于多保真评估的SNN-NAS算法
# @Date       :2025/02/22 00:42:09
# @Author     :li-yuan
# @Email      :yuanli080800@gmail.com
# '''


# import torch
# import random
# import torch.backends.cudnn as cudnn
# import numpy as np
# import matplotlib.pyplot as plt
# import config2

# def anyToDecimal(num, n):
#     baseStr = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
#                "a": 10, "b": 11, "c": 12, "d": 13, "e": 14, "f": 15, "g": 16, "h": 17, "i": 18, "j": 19}
#     new_num = 0
#     nNum = len(num) - 1
#     for i in num:
#         new_num = new_num + baseStr[i] * pow(n, nNum)
#         nNum = nNum - 1
#     return new_num


# def check_validity(coding):
#     count_vector = coding.ravel()
#     con_mat = np.zeros((4, 4))
#     position = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

#     flag2 = 1
#     flag3 = 1

#     for num, (k0, k1) in enumerate(position):
#         con_mat[k0, k1] = count_vector[num]

#     neigh2_cnts = con_mat @ con_mat
#     neigh3_cnts = neigh2_cnts @ con_mat
#     neigh4_cnts = neigh3_cnts @ con_mat
#     connection_graph = con_mat + neigh2_cnts + neigh3_cnts + neigh4_cnts

#     for node in range(3):
#         if connection_graph[node, 3] == 0:  # if any node doesnt send information to the last layer, remove it
#             flag2 = 0
#         if flag2 == 0: return -1

#     for node in range(3):
#         if connection_graph[0, node + 1] == 0:  # if any node doesnt get information from the input layer, remove it
#             flag3 = 0
#         if flag3 == 0: return -1

#     # 保证node1到node4一定有连接
#     if con_mat[0, 3] == 0: return -1  # ensure direct connection between input=>output for fast information propagation

#     return con_mat


# # def solution_evaluation(individual):
# #     idv_id = utils.encoding_to_id(individual)
# #     _, _, model_param, _, _, val_acc, _, _, _, seconds = get_train_log(idv_id, 'E:/master_degree/SpikeNAS-Bench/data/CIFAR10')
    
# #     return 1-val_acc[99]/100, model_param, 0, 0


# class individual():
#     def __init__(self, individual_single):
#         self.individual_single = individual_single
#         self.id_index = 0
#         # 初始化适应度
#         self.fitness = np.array([0.1, 0.1, 0.2, 0.2])
#         # 初始化存活和淘汰统计
#         self.survive_count = 0
#         self.eliminate_count = 0
        
#         self.encoding_to_id()
        
#         self.fitness_arxiv = []
#         self.used_epoch = 0
        
#     def encoding_to_id(self):
#         encoding_re = self.individual_single[::-1]
#         id = anyToDecimal(''.join('%s' %a for a in encoding_re), 5)
        
#         self.id_index = id
        

#     def evaluate(self, EMO_epochs):
#         # self.fitness = solution_evaluation()
#         self.fitness_arxiv.append([self.used_epoch, self.fitness[0]])
        
#         self.used_epoch = EMO_epochs
        
# single_ind = individual([1, 0, 0, 0, 0, 0])
# print(single_ind.individual_single)
# print(single_ind.id_index)
# print(single_ind.fitness)
# print(single_ind.survive_count)
# print(single_ind.eliminate_count)
# print(single_ind.fitness_arxiv)
# print(single_ind.used_epoch)
# single_ind.evaluate(1)
# print(single_ind.fitness_arxiv)
# print(single_ind.used_epoch)

import numpy as np

# 假设 self.max_length = 6
max_length = 6  # 变量个数

# 设定所有变量的上界为 4.5，下界为 -0.5
Up_boundary = np.full((max_length,), 4.5)  # 创建一个全 4.5 的上边界数组
Low_boundary = np.full((max_length,), -0.5)  # 创建一个全 -0.5 的下边界数组

# 组合成 2×max_length 的边界矩阵
Boundary = np.vstack((Up_boundary, Low_boundary))

# 打印边界矩阵以检查结果
print(Boundary)
