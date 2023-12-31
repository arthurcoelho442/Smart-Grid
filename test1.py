import numpy as np
import random
from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.ref_dirs import get_reference_directions

# Definir o problema de otimização
n_var = 1

objs = [
    lambda x: np.sum((x - 2) ** 2),
    lambda x: np.sum((x + 2) ** 2),
]

problem = FunctionalProblem(n_var, objs, xl=np.full(n_var, -10), xu=np.full(n_var, 10))

# Configurar o algoritmo NSGA-III
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=random.randint(1, 300))
algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)

# Executar a otimização
result = minimize(problem, algorithm, seed=1, termination=('n_gen', 300), verbose=True)

# Acessar os resultados
final_population = result.X
final_objectives = result.F

# Imprimir os resultados
lista = []
for individual, objectives in zip(final_population, final_objectives):
    print("Individual:", individual)
    print("Objectives:", objectives)
    print("------------------------")
    
    lista.append((individual, objectives, np.linalg.norm([0, 0] - objectives)))

print(min(lista, key=lambda x: x[2]))

Scatter().add(result.F).show()