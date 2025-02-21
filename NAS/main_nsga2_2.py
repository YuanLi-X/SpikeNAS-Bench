import time
import os
import logging
import numpy as np
import torch
import random
import sys
from get_train_log import get_train_log
import config
import utils
from evolution_algorithm_obj2 import initialize_population, generate_offspring, enviromental_selection
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import IntegerRandomSampling

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seed_torch(seed=640):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

start = time.time()

args = config.get_args()
seed_torch(args.seed)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)
    
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(
    os.path.join(args.save_dir, 'search_{}_log_{}.txt').format(args.fitness_evaluator, time.strftime("%Y%m%d-%H%M%S")))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.getLogger().setLevel(logging.INFO)

for arg, val in args.__dict__.items():
    logging.info(arg + '.' * (60 - len(arg) - len(str(val))) + str(val))

# ========== 适应度评估函数 ========== #
def evaluate_fitness(individual, generation):
    print(individual)
    idv_id = utils.encoding_to_id(individual)
    print("idv_id: ", idv_id)
    _, _, model_param, _, _, val_acc, _, _, _, seconds = get_train_log(idv_id, 'E:/master_degree/SpikeNAS-Bench/data/CIFAR10')
    print("generation:", generation)
    
    return np.array([100 - val_acc[generation*1-1], model_param]), seconds*(generation*1/100)

# ========== 定义 NSGA-II 适配的问题 ========== #
class NASProblem(Problem):
    def __init__(self):
        super().__init__(n_var=6, n_obj=2, n_constr=0, xl=-0.5, xu=4.5)
        self.args = args
        self.all_individuals = {}  # 改为字典，key: 个体的 tuple，value: (acc, param)
        self.generation = 0

    def _evaluate(self, X, out, *args, **kwargs):
        acc_list, param_list = [], []

        for individual in X:
            individual = tuple(int(round(x)) for x in individual)  # 确保整数并转换为 tuple
            print("individual: ", individual, "generation: ", self.generation)

            if individual in self.all_individuals:
                # 复用之前的评估结果
                prev_acc, prev_param = self.all_individuals[individual]
                acc_list.append(prev_acc)
                param_list.append(prev_param)
            else:
                # 计算新个体的适应度
                acc_param, _ = evaluate_fitness(individual, self.generation)
                acc_list.append(acc_param[0])
                param_list.append(acc_param[1])
                
                # 存储到历史记录，避免重复评估
                self.all_individuals[individual] = (acc_param[0], acc_param[1])

        out["F"] = np.column_stack([acc_list, param_list])  # 确保 shape 正确
        
    def next_generation(self):
        self.generation += 1

    # 去重操作
    def remove_duplicates(self, population):
        int_X = np.array([np.round(ind).astype(int) for ind in population])
        unique_X, indices = np.unique(int_X, axis=0, return_index=True)
        unique_F = population.F[indices]
        return unique_X, unique_F

# ========== 自定义 NSGA-II 采样 ========== #
class CustomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        print("initialize_population:", n_samples)
        return initialize_population(n_samples)

# ========== 运行 NSGA-II ========== #
problem = NASProblem()

algorithm = NSGA2(
    pop_size=args.population_size,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=0.9, eta=2),
    mutation=PM(prob=0.2, eta=20),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", args.generation_number)

def callback(algorithm):
    problem.generation += 1



res = minimize(problem, algorithm, termination, verbose=True, callback=callback)

# 转换为整数， 去重，返回唯一的架构及索引
int_X = np.array([np.round(ind).astype(int) for ind in res.X])
unique_X, indices = np.unique(int_X, axis=0, return_index=True)

# 根据索引找到对应的目标值
unique_F = res.F[indices]

# 输出去重后的架构
print("最优架构（去重后）:")
for i in range(len(unique_X)):
    print(f"架构索引: {utils.encoding_to_id(unique_X[i])}, 目标: {unique_F[i]}")
    best_id = utils.encoding_to_id(unique_X[i])
    _, _, model_param, _, _, val_acc, _, test_acc, _, _ = get_train_log(best_id, 'E:/master_degree/SpikeNAS-Bench/data/CIFAR10')
    logging.info("[architecture_{}] [params = {}] [val_acc = {:.3f}] [test_acc = {:.3f}]".format(best_id, model_param, val_acc[-1], test_acc))
