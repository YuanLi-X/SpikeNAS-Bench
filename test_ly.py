import numpy as np

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


def anyToDecimal(num, n):
    baseStr = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
               "a": 10, "b": 11, "c": 12, "d": 13, "e": 14, "f": 15, "g": 16, "h": 17, "i": 18, "j": 19}
    new_num = 0
    nNum = len(num) - 1
    for i in num:
        new_num = new_num + baseStr[i] * pow(n, nNum)
        nNum = nNum - 1
    return new_num


def encoding_to_id(encoding):
    encoding_re = encoding[::-1]
    id = anyToDecimal(''.join('%s' %a for a in encoding_re), 5)

    return id



def initialize_population(population_size):
    initialized_population = []
    while(len(initialized_population) < population_size):
        individual = np.random.randint(0, 5, size=6)
        flag = check_validity(individual)
        if type(flag) == int:
            continue
        else:
            initialized_population.append(individual)

    return initialized_population


initialize_population = initialize_population(10)
print(initialize_population)
id_index = encoding_to_id(initialize_population[0])
print(id_index)

array_list = [np.array([2, 1, 3, 2, 3, 4])]
id_index = encoding_to_id(array_list[0])
print(id_index)

flag = check_validity(array_list[0])
print("flag: ", flag)
print(encoding_to_id(array_list[0]))