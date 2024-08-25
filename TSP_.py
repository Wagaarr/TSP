import numpy as np
import random
import matplotlib.pyplot as plt

def calculate_distance(path, distance_matrix):
    return sum(distance_matrix[path[i-1], path[i]] for i in range(len(path)))

def initialize_population(num_cities, population_size):
    population = []
    for _ in range(population_size):
        individual = np.random.permutation(num_cities)
        population.append(individual)
    return np.array(population)

def evaluate_population(population, distance_matrix):
    return np.array([calculate_distance(ind, distance_matrix) for ind in population])

def select_parents(population, fitness):
    fitness = 1 / fitness 
    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness
    selected = np.random.choice(len(population), size=2, p=probabilities)
    return population[selected[0]], population[selected[1]]

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(np.random.choice(range(size), 2, replace=False))
    child = -np.ones(size, dtype=int)
    child[start:end + 1] = parent1[start:end + 1]
    pointer = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[pointer] in child:
                pointer += 1
            child[i] = parent2[pointer]
    return child

def mutate(individual):
    return reciprocal_exchange_mutation(individual)

def reciprocal_exchange_mutation(cam):
    N = len(cam)
    ncam = np.zeros(N, dtype=np.int16)
    
    i = np.random.randint(N)
    j = i
    while j == i:
        j = np.random.randint(N)
    
    if i > j:
        ini, fim = j, i
    else:
        ini, fim = i, j
    
    for k in range(N):
        if ini <= k <= fim:
            ncam[k] = cam[fim - k + ini]
        else:
            ncam[k] = cam[k]
    
    return ncam

def plot_paths(city_positions, initial_path, optimized_path):
    plt.figure(figsize=(10, 6))

    
    plt.plot(city_positions[initial_path, 0], city_positions[initial_path, 1], 
             marker='o', color='red', linestyle='-', alpha=0.3, label='Caminho Inicial')

    # Caminho otimizado em verde
    plt.plot(city_positions[optimized_path, 0], city_positions[optimized_path, 1], 
             marker='o', color='green', linestyle='-', label='Caminho Otimizado')

    
    plt.scatter(city_positions[:, 0], city_positions[:, 1], c='blue', marker='x', s=100, label='Cidades')

    plt.title('Caminho Inicial e Otimizado')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def genetic_algorithm(num_cities, city_positions, population_size=20, generations=1000, mutation_rate=0.1):
    # Matriz de distâncias para coordenadas normalizadas
    distance_matrix = np.sqrt(np.sum((city_positions[:, np.newaxis] - city_positions[np.newaxis, :]) ** 2, axis=2))
    population = initialize_population(num_cities, population_size)
    best_distances = []
    best_individual = None
    best_distance = float('inf')

    initial_path = population[0].copy()

    for generation in range(generations):
        fitness = evaluate_population(population, distance_matrix)
        min_distance = np.min(fitness)
        best_distances.append(min_distance)

        if min_distance < best_distance:
            best_distance = min_distance
            best_individual = population[np.argmin(fitness)]

        new_population = []
        new_population.append(best_individual)  # Clonar o melhor indivíduo

        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitness)
            child = crossover(parent1, parent2)
            if np.random.rand() < mutation_rate:
                child = mutate(child)
            new_population.append(child)
        
        population = np.array(new_population)

    # Plot dos caminhos inicial e otimizado
    plot_paths(city_positions, initial_path, best_individual)

    return best_individual, best_distance, best_distances


num_cities = 140
city_positions = np.random.rand(num_cities, 2)  # Gera posições aleatórias entre 0 e 1 para as cidades

best_path, best_distance, best_distances = genetic_algorithm(num_cities, city_positions)

# Plot da evolução da melhor distância
plt.plot(best_distances)
plt.title('Evolução da Menor Distância Percorrida')
plt.xlabel('Geração')
plt.ylabel('Distância')
plt.grid(True)
plt.show()

print(f"Melhor caminho: {best_path}")
print(f"Menor distancia: {best_distance}")
