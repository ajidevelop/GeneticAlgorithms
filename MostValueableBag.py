__author__ = 'LobaAjisafe'

import random
from collections import OrderedDict
import matplotlib.pyplot as plt


def fitness(value, max_value):
    """
    :param value: value of current bag (int)
    :param max_value: max value of bag (int)
    :return: score (float)
    """
    score = abs((max_value - value[0]) / 100)
    score += len(value[1]) ** .5
    return score


def generate_bag(max_value, prices):
    """
    :param max_value: max value of bag (int)
    :param prices: list of prices (list)
    :return: bag (int)
    """
    bag = 0
    bag_items = []
    while bag < max_value:
        p = random.choice(prices)
        bag += p
        bag_items.append(p)
    return bag, bag_items


def generate_first_population(population_size, max_value, prices):
    """
    :param population_size: size of population (int)
    :param max_value: max value of bag (int)
    :return: population (list b
    """
    population = []
    for i in range(population_size):
        population.append(generate_bag(max_value, prices))
    return population


def compute_perfect_population(population, max_value):
    """
    :param population:
    :param max_value:
    :return:
    """
    population_perfect = {}
    for i in range(len(population)):
        population_perfect[i] = {
            'score': fitness(population[i], max_value),
            'value': population[i][0],
            'items': population[i][1]
        }
    return OrderedDict(sorted(population_perfect.items(), key=lambda x: x[1]['score']))


def select_from_population(population_sorted, best_sample, lucky_few):
    """
    :param population_sorted:
    :param best_sample:
    :param lucky_few:
    :return:
    """

    next_generation = []
    for i in range(best_sample):
        next_generation.append(list(population_sorted.values())[i])
    for i in range(lucky_few):
        next_generation.append(random.choice(population_sorted))
    random.shuffle(next_generation)
    return next_generation


def create_child(bag1, bag2):
    """
    :param bag1:
    :param bag2:
    :return:
    """
    child = 0
    items = []
    while child < 60:
        if int(100 * random.random()) < 50:
            x = random.choice(bag1['items'])
            child += x
            items.append(x)
        else:
            x = random.choice(bag2['items'])
            child += x
            items.append(x)
    return child, items


def create_children(breeders, number_of_child):
    """
    :param breeders:
    :param number_of_child:
    :return:
    """
    next_population = []
    for i in range(len(breeders)//2):
        for j in range(number_of_child):
            next_population.append(create_child(breeders[0], breeders[len(breeders) - 1 - i]))
    return next_population


def mutateBag(bag, prices):
    """
    :param bag:
    :return:
    """
    x = random.choice(prices)
    i = random.randint(0, len(bag[1])-1)
    bag[1][i] = x
    new_bag = (sum(bag[1]), bag[1])
    return new_bag


def mutatePopulation(population, chance_of_mutation, prices):
    """
    :param population:
    :param chance_of_mutation:
    :return:
    """
    for i in range(len(population)):
        if random.random() * 100 < chance_of_mutation:
            population[i] = mutateBag(population[i], prices)
    return population


def next_generations(first_genertion, max_value, best_sample, lucky_few, number_of_child, chance_of_mutation, prices):
    """
    :param first_genertion:
    :param max_value:
    :param best_sample:
    :param lucky_few:
    :param number_of_child:
    :param chance_of_mutation:
    :return:
    """
    population_sorted = compute_perfect_population(first_genertion, max_value)
    next_breeders = select_from_population(population_sorted, best_sample, lucky_few)
    next_population = create_children(next_breeders, number_of_child)
    nextGeneration = mutatePopulation(next_population, chance_of_mutation, prices)
    return nextGeneration


def multiple_generations(number_of_generation, max_value, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation, prices):
    historic = [generate_first_population(size_population, max_value, prices)]
    for i in range(number_of_generation):
        historic.append(next_generations(historic[i], max_value, best_sample, lucky_few, number_of_child, chance_of_mutation, prices))
    return historic


def get_best_inidividual_from_population(population, max_value):
    return compute_perfect_population(population, max_value)[0]


def get_list_best_individual_from_historique(historic, max_value):
    best_individuals = []
    for population in historic:
        best_individuals.append(get_best_inidividual_from_population(population, max_value))
    return best_individuals


def print_simple_result(historic, max_value, number_of_generations):
    result = get_list_best_individual_from_historique(historic, max_value)[number_of_generations - 1]
    print(f'solution: / {result} / de fitness: {str(result)}')


def evolutionBestFitness(historic, max_value):
    plt.axis([0, len(historic), 0, max_value])
    plt.title(f'Max Value: {max_value}')

    evolutionFitness = []
    for population in historic:
        evolutionFitness.append(get_best_inidividual_from_population(population, max_value)[1])
    plt.plot(evolutionFitness)
    plt.ylabel('fitness best individual')
    plt.xlabel('generation')
    plt.show()


def evolutionAverageFitness(historic, max_value, size_population):
    plt.axis([0, len(historic), 0, max_value])
    plt.title(max_value)

    evolutionFitness = []
    for population in historic:
        populationPerf = compute_perfect_population(population, max_value)
        averageFitness = 0
        for individual in populationPerf:
            averageFitness += individual[1]
        evolutionFitness.append(averageFitness/size_population)
    plt.plot(evolutionFitness)
    plt.ylabel('Average fitness')
    plt.xlabel('generation')
    plt.show()


bag_max_value = 60
prices = [10, 12, 15, 2, 5]
population_size = 25
bag_best_sample = 5
bag_lucky_few = 5
number_of_children = 5
number_of_generation = 20
chance_of_mutation = 1

if (bag_best_sample + bag_lucky_few) / 2 * number_of_children != population_size:
    print('population ize not stable')
else:
    historic = multiple_generations(number_of_generation, bag_max_value, population_size, bag_best_sample, bag_lucky_few, number_of_children,
                                    chance_of_mutation, prices)

    print_simple_result(historic, bag_max_value, number_of_generation)

