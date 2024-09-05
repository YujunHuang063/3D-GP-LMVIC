import math
import random
import numpy as np
import networkx as nx
from functools import lru_cache

# 计算路径长度
def calculate_total_distance(order, distance_matrix):
    total_distance = 0
    for i in range(len(order) - 1):
        total_distance += distance_matrix[order[i], order[i+1]]
    return total_distance

# 生成初始种群
def generate_initial_population(pop_size, num_nodes):
    population = []
    for _ in range(pop_size):
        individual = list(np.random.permutation(num_nodes))
        population.append(individual)
    return population

# 适应度函数
def fitness(individual, distance_matrix):
    return 1 / calculate_total_distance(individual, distance_matrix)

# 选择操作（轮盘赌选择）
def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    probs = [f / total_fitness for f in fitnesses]
    selected = np.random.choice(len(population), size=len(population), p=probs)
    return [population[i] for i in selected]

# 交叉操作（次序交叉 OX）
def crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child1 = [None] * size
    child1[a:b] = parent1[a:b]
    child2 = [None] * size
    child2[a:b] = parent2[a:b]
    
    fill_pos = lambda p, c: [item for item in p if item not in c[a:b]]
    
    child1[:a] = fill_pos(parent2, child1)[:a]
    child1[b:] = fill_pos(parent2, child1)[a:]
    child2[:a] = fill_pos(parent1, child2)[:a]
    child2[b:] = fill_pos(parent1, child2)[a:]
    
    return child1, child2

# 变异操作（交换变异）
def mutate(individual):
    a, b = random.sample(range(len(individual)), 2)
    individual[a], individual[b] = individual[b], individual[a]

# 遗传算法
def genetic_algorithm(distance_matrix, pop_size=100, generations=1000, mutation_rate=0.1):
    num_nodes = distance_matrix.shape[0]
    population = generate_initial_population(pop_size, num_nodes)
    
    for generation in range(generations):
        fitnesses = [fitness(ind, distance_matrix) for ind in population]
        
        population = select(population, fitnesses)
        
        next_population = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = population[i], population[i+1]
            child1, child2 = crossover(parent1, parent2)
            next_population.extend([child1, child2])
        
        for individual in next_population:
            if random.random() < mutation_rate:
                mutate(individual)
        
        population = next_population
        
        if generation % 100 == 0:
            best_individual = min(population, key=lambda ind: calculate_total_distance(ind, distance_matrix))
            print(f'Generation {generation}: Best Distance = {calculate_total_distance(best_individual, distance_matrix)}')
    
    best_individual = min(population, key=lambda ind: calculate_total_distance(ind, distance_matrix))
    return best_individual

def tsp_path(dist_matrix):
    n = len(dist_matrix)

    # 使用lru_cache进行记忆化
    @lru_cache(None)
    def dp(pos, visited):
        if visited == (1 << n) - 1:
            return 0
        
        min_cost = float('inf')
        for i in range(n):
            if not visited & (1 << i):
                min_cost = min(min_cost, dist_matrix[pos][i] + dp(i, visited | (1 << i)))
        return min_cost

    # 寻找路径
    def find_path():
        path = []
        visited = 0
        pos = 0
        for _ in range(n):
            min_cost = float('inf')
            next_pos = -1
            for i in range(n):
                if not visited & (1 << i):
                    cost = dist_matrix[pos][i] + dp(i, visited | (1 << i))
                    if cost < min_cost:
                        min_cost = cost
                        next_pos = i
            path.append(next_pos)
            pos = next_pos
            visited |= (1 << next_pos)
        return path

    # 初始化dp并找到最小路径
    min_cost = float('inf')
    start_pos = -1
    for i in range(n):
        cost = dp(i, 1 << i)
        if cost < min_cost:
            min_cost = cost
            start_pos = i

    # 从最优起点开始寻找路径
    optimal_path = [start_pos]
    pos = start_pos
    visited = 1 << start_pos
    for _ in range(n - 1):
        min_cost = float('inf')
        next_pos = -1
        for i in range(n):
            if not visited & (1 << i):
                cost = dist_matrix[pos][i] + dp(i, visited | (1 << i))
                if cost < min_cost:
                    min_cost = cost
                    next_pos = i
        optimal_path.append(next_pos)
        pos = next_pos
        visited |= (1 << next_pos)
    
    return optimal_path, min_cost

def nearest_neighbor_tsp(similarity_matrix):
    num_elements = similarity_matrix.shape[0]
    visited = [False] * num_elements
    order = []

    current_index = 0
    order.append(current_index)
    visited[current_index] = True

    while len(order) < num_elements:
        similarity = -np.inf
        next_index = -1
        for i in range(num_elements):
            if not visited[i] and similarity_matrix[current_index, i] > similarity:
                similarity = similarity_matrix[current_index, i]
                next_index = i
        
        order.append(next_index)
        visited[next_index] = True
        current_index = next_index

    return order

def nearest_neighbor_tsp_distance(distance_matrix):
    num_elements = distance_matrix.shape[0]
    visited = [False] * num_elements
    order = []

    current_index = 0
    order.append(current_index)
    visited[current_index] = True

    while len(order) < num_elements:
        min_distance = np.inf
        next_index = -1
        for i in range(num_elements):
            if not visited[i] and distance_matrix[current_index, i] < min_distance:
                min_distance = distance_matrix[current_index, i]
                next_index = i
        
        order.append(next_index)
        visited[next_index] = True
        current_index = next_index

    return order

def predict_next_rotation_matrices(rotations):
    if len(rotations) < 2:
        # 如果点少于2个，直接返回最近的旋转矩阵
        return rotations[-1]

    R1 = rotations[-2]
    R2 = rotations[-1]
    
    # 计算 R1 的逆矩阵
    R1_inv = np.linalg.inv(R1)
    
    # 计算 delta_R
    delta_R = np.dot(R2, R1_inv)
    
    # 预测下一个旋转矩阵 next_R
    next_R = np.dot(delta_R, R2)
    
    return next_R

def decompose_transform(transform):
    rotation_matrix = transform[:3, :3]
    translation_vector = transform[:3, 3]
    return rotation_matrix, translation_vector

def compose_transform(rotation_matrix, translation_vector):
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation_vector
    return transform

def predict_next_translation(translations):
    if len(translations) < 2:
        # 如果点少于2个，直接返回最近的平移向量
        return translations[-1]

    if len(translations) >= 2:
        # 如果点等于2个，使用线性预测
        delta_t = translations[-1] - translations[-2]
        next_t = translations[-1] + delta_t
        return next_t

    # 只使用最近的三个数据点进行二次函数拟合
    recent_translations = translations[-3:]
    t = np.array([0, 1, 2])
    recent_translations = np.array(recent_translations)
    
    next_t = np.zeros(3)
    for i in range(3):  # 对 x, y, z 轴分别进行拟合
        coeffs = np.polyfit(t, recent_translations[:, i], 2)  # 拟合二次多项式
        next_t[i] = np.polyval(coeffs, 3)  # 预测下一个时间点的值（对应 t=3）
    
    return next_t

def predict_next_transform(view_world_transforms):
    rotations = []
    translations = []
    for transform in view_world_transforms:
        rotation_matrix, translation = decompose_transform(transform)
        rotations.append(rotation_matrix)
        translations.append(translation)

    next_rotation = predict_next_rotation_matrices(rotations)
    next_translation = predict_next_translation(translations)
    next_transform = compose_transform(next_rotation, next_translation)
    return next_transform

def frobenius_norm_difference(predicted_transform, target_transform):
    target_inv = np.linalg.inv(target_transform)
    AB_inv = np.dot(predicted_transform, target_inv)
    identity_matrix = np.eye(4)
    difference_AB_inv = AB_inv - identity_matrix
    frobenius_norm = np.linalg.norm(difference_AB_inv, 'fro')
    return frobenius_norm

def sort_by_distance_using_mst(distance_matrix):
    # Ensure the diagonal elements are zero
    np.fill_diagonal(distance_matrix, 0)
    
    # Step 2: Build a graph from the distance matrix
    graph = nx.from_numpy_matrix(distance_matrix)
    
    # Step 3: Compute the Minimum Spanning Tree (MST)
    mst = nx.minimum_spanning_tree(graph)
    
    # Step 4: Perform DFS on the MST to get the order
    start_node = 0
    order = list(nx.dfs_preorder_nodes(mst, source=start_node))
    
    return order

def simulated_annealing(matrix, initial_order=None, initial_temp=1000, cooling_rate=0.99, max_iter=10000):
    if initial_order is None:
        initial_order = list(range(len(matrix)))
    
    current_order = initial_order
    current_distance = calculate_total_distance(matrix, current_order)
    best_order = current_order
    best_distance = current_distance
    temperature = initial_temp

    for iteration in range(max_iter):
        new_order = current_order[:]
        i, j = random.sample(range(len(new_order)), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_distance = calculate_total_distance(matrix, new_order)

        if new_distance < current_distance:
            current_order = new_order
            current_distance = new_distance
            if new_distance < best_distance:
                best_order = new_order
                best_distance = new_distance
        else:
            delta = new_distance - current_distance
            acceptance_probability = math.exp(-delta / temperature)
            if random.random() < acceptance_probability:
                current_order = new_order
                current_distance = new_distance

        temperature *= cooling_rate

    return best_order

def calculate_total_distance(distance_matrix, order):
    total_distance = 0
    for i in range(len(order) - 1):
        total_distance += distance_matrix[order[i], order[i + 1]]
    return total_distance
