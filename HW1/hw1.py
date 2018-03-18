import random

import matplotlib.pyplot as plt
import numpy as np


# create matrix [0,1] => multiply to [0,2] => shift to [-1,1]
def MatrixCreate(a, b):
    return (np.random.rand(a, b) * 2) - 1


# average of 'v' => sum/length
def Fitness(v):
    temp = v.reshape(-1, 1)
    return (sum(temp) / len(temp))


# population fitness
def ComputeFitness(pop):
    return np.array([Fitness(i) for i in pop]).flatten().transpose()


# hill climber mutation - just replacing elements
def MatrixPerturb(v, u):
    def mutate_or_copy(element):
        if (np.random.rand() <= u):
            return (np.random.rand() * 2) - 1
        else:
            return element

    return np.array([mutate_or_copy(e) for e in v])


# genetic algorithm mutation - adding values from normal distribution
def Mutate(v, u, m):
    def mutate_or_copy(element):
        if (np.random.rand() <= u):
            value = element + m * np.random.randn()

            # too large
            if value > 1:
                return 1

            # too small
            if value < -1:
                return -1

            # ok
            return value
        else:
            return element

    return np.array([mutate_or_copy(e) for e in v])


# Tournament selection with probability t of selecting the better one
def TourSel(pop, fitness, t):
    # random choice of contestants
    first_index = random.randrange(0, len(pop))
    second_index = random.randrange(0, len(pop))

    # first_index points to the better one
    if (fitness[first_index] < fitness[second_index]):
        temp = second_index
        first_index = second_index
        second_index = temp

    # probability for returning the bigger one
    if (np.random.rand() <= t):
        return pop[first_index]
    else:
        return pop[second_index]


# Crossover operator for Genetic Algorithm
def Crossover(i2, c):
    first = np.array(i2[0])
    second = np.array(i2[1])

    if (np.random.rand() <= c):
        index = random.randrange(0, len(first))
        first_result = np.append(first[:index], second[index:])
        second_result = np.append(second[:index], first[index:])
    else:
        first_result = first.copy()
        second_result = second.copy()

    return [first_result, second_result]


# gradient optimizer;
# MaxGen is the number of generation
# u is the probability if mutation
# L is the length of a vector
def HillClimber(MaxGen=5000, u=0.05, L=50):
    parent = MatrixCreate(1, L).flatten()
    parentFitness = Fitness(parent)

    result_genes = [parent]
    result_fitness = [parentFitness]

    for currentGeneration in range(0, MaxGen):

        child = MatrixPerturb(parent, u)
        childFitness = Fitness(child)

        if (childFitness > parentFitness):
            parent = child
            parentFitness = childFitness

        result_genes.append(parent)
        result_fitness.append(parentFitness)

    result_fitness = np.array(result_fitness)
    result_genes = np.array(result_genes).transpose()

    return (result_fitness, result_genes)


# Genetic algorithm
def GenAlg(MaxGen=333, PopSize=15, L=50, t=0.8, u=0.05, m=0.9, c=0.7):
    pop = MatrixCreate(PopSize, L)
    pop_fitness = ComputeFitness(pop)

    best_element = max(pop, key=lambda row: Fitness(row))
    best_fitness = Fitness(best_element)

    result_fitness = [best_fitness]
    result_genes = [best_element]

    for generation in range(0, MaxGen):
        next_generation = [best_element]

        for i in range(0, (PopSize // 2) + 1):
            first_parent = TourSel(pop, pop_fitness, t)
            second_parent = TourSel(pop, pop_fitness, t)

            [first_child, second_child] = Crossover([first_parent, second_parent], c)
            first_child = Mutate(first_child, u, m)
            second_child = Mutate(second_child, u, m)

            next_generation.append(first_child)
            next_generation.append(second_child)

        # next generation population
        pop = np.array(next_generation[0:PopSize])
        pop_fitness = ComputeFitness(pop)

        # store best values from the population
        best_element = max(pop, key=lambda row: Fitness(row))
        best_fitness = Fitness(best_element)

        result_genes.append(best_element)
        result_fitness.append(best_fitness)

    # transform results & return
    result_fitness = np.array(result_fitness)
    result_genes = np.array(result_genes).transpose()

    return (result_fitness, result_genes)


# Testing runs for Genetic Algorithm
def GenAlgRuns():
    # generate results
    results = [GenAlg() for _ in range(0, 5)]

    # Fitness plot
    for run in results:
        plt.plot(run[0])

    plt.title("Genetic Algorithm Runs")
    plt.xlabel("Fitness")
    plt.ylabel("Generation")

    plt.show()

    # Genes plot
    plt.title("Genetic Algorithm Genes")
    plt.imshow(results[0][1], cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
    plt.show()


# Testing runs for basic Hill Climber
def HillClimberRuns():
    results = [HillClimber() for _ in range(0, 5)]

    # add to plot
    for run in results:
        plt.plot(run[0])

    # labels
    plt.title("Hill Climber Runs")
    plt.xlabel("Fitness")
    plt.ylabel("Generation")

    plt.show()

    # Genes plot
    plt.title("Hill Climber Genes")
    plt.imshow(results[0][1], cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
    plt.show()


# MAIN
if __name__ == "__main__":
    HillClimberRuns()
    GenAlgRuns()
