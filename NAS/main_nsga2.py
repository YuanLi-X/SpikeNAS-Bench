import time
import os
import logging
import numpy as np
import torch
import random
import sys
from get_train_log2 import get_train_log
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

# ==========  适应度评估函数  ========== #
def evaluate_fitness(args, individual, evaluator):
    print(individual)
    idv_id = utils.encoding_to_id(individual)
    print("idv_id: ", idv_id)
    _, _, model_param, _, _, val_acc, _, _, _, seconds = get_train_log(idv_id, 'E:/master_degree/SpikeNAS-Bench/data/CIFAR10')

    if evaluator == 'early_stop_test':
        return np.array([100 - val_acc[-1], model_param]), seconds
    elif evaluator == 'early_stop_10':
        return np.array([100 - val_acc[9], model_param]), seconds * 0.1
    elif evaluator == 'early_stop_40':
        return np.array([100 - val_acc[39], model_param]), seconds * 0.4
    elif evaluator == 'early_stop_60':
        return np.array([100 - val_acc[59], model_param]), seconds * 0.6

# ==========  定义 NSGA-II 适配的问题  ========== #
class NASProblem(Problem):
    def __init__(self, evaluator):
        super().__init__(n_var=6, n_obj=2, n_constr=0, xl=-0.5, xu=4.5)  # 0-4 只是示例
        self.evaluator = evaluator

    def _evaluate(self, X, out, *args, **kwargs):
        acc_list, param_list = [], []
        for individual in X:
            individual = [int(round(x)) for x in individual]
            print("individual: ", individual)
            acc_param, _ = evaluate_fitness(args, individual, self.evaluator)
            print("acc_param: ", acc_param)
            acc_list.append(acc_param[0])
            param_list.append(acc_param[1])
        out["F"] = np.column_stack([acc_list, param_list])


# ==========  自定义 NSGA-II 采样 ========== #
class CustomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        print("initialize_population:", n_samples)
        return initialize_population(n_samples)


# ==========  运行 NSGA-II  ========== #
problem = NASProblem(evaluator='early_stop_10')

algorithm = NSGA2(
    pop_size=args.population_size,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=0.9, eta=2),
    mutation=PM(prob=0.2, eta=20),
    eliminate_duplicates=True #避免种群中出现重复的个体，保持种群的多样性
)

termination = get_termination("n_gen", args.generation_number)

res = minimize(problem, algorithm, termination, verbose=True)


# 转换为整数， 去重，返回唯一的架构及索引
int_X = np.array([np.round(ind).astype(int) for ind in res.X])
unique_X, indices = np.unique(int_X, axis=0, return_index=True)

# 根据索引找到对应的目标值
unique_F = res.F[indices]

# 输出去重后的架构
print("最优架构（去重后）:")
for i in range(len(unique_X)):
    # print(f"架构: {unique_X[i].tolist()}, 目标: {unique_F[i]}")
    print(f"架构索引: {utils.encoding_to_id(unique_X[i])}, 目标: {unique_F[i]}")
    best_id = utils.encoding_to_id(unique_X[i])
    _, _, model_param, _, _, val_acc, _, test_acc, _, _ = get_train_log(best_id, 'E:/master_degree/SpikeNAS-Bench/data/CIFAR10')
    logging.info("[architecture_{}] [params = {}] [val_acc = {:.3f}] [test_acc = {:.3f}]".format(best_id, model_param, val_acc[-1], test_acc))
    # end = time.time()
    # search_time_with_SNAS = end - start
    # search_time_total = search_time_total + end - start
    # hour = search_time_total // 3600
    # minute = (search_time_total - hour * 3600) // 60
    # second = search_time_total - hour * 3600 - minute * 60
    # logging.info("total search time seconds: {}".format(search_time_total))
    # logging.info("total search time: {}h {}m {}s".format(hour, minute, second))
    # logging.info("search time with SNAS: {}s".format(search_time_with_SNAS))