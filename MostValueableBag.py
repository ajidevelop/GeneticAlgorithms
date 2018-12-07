__author__ = 'LobaAjisafe'

import random
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np


def fitness(value, max_value):
    """
    :param value: value of current bag (int)
    :param max_value: max value of bag (int)
    :return: score (float)
    """
    score = abs((max_value - value[0]) / 100)
    score += len(value[1]) ** (1/4)
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
        try:
            next_generation.append(list(population_sorted.values())[i])
        except IndexError:
            pass
    for i in range(lucky_few):
        next_generation.append(random.choice(population_sorted))
    random.shuffle(next_generation)
    return next_generation


def create_child(bag1, bag2, max_value):
    """
    :param bag1:
    :param bag2:
    :return:
    """
    child = 0
    items = []
    while child < max_value:
        if int(100 * random.random()) < 50:
            x = random.choice(bag1['items'])
            child += x
            items.append(x)
        else:
            x = random.choice(bag2['items'])
            child += x
            items.append(x)
    return child, items


def create_children(breeders, number_of_child, max_value):
    """
    :param breeders:
    :param number_of_child:
    :return:
    """
    next_population = []
    for i in range(len(breeders)//2):
        for j in range(number_of_child):
            next_population.append(create_child(breeders[0], breeders[len(breeders) - 1 - i], max_value))
    return next_population


def mutateBag(bag, prices):
    """
    :param bag:
    :return:
    """
    x = random.choice(prices)
    i = random.randint(0, len(bag[1])-1)
    if random.random() * 100 < 50:
        bag[1][i] = x   # change a value of the bag
    else:
        bag[1].append(x)  # add an extra value into the bag
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
    next_population = create_children(next_breeders, number_of_child, max_value)
    nextGeneration = mutatePopulation(next_population, chance_of_mutation, prices)
    return nextGeneration


def multiple_generations(number_of_generation, max_value, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation, prices):
    evolutionFitnessS = []
    evolutionFitnessL = []
    x_vec = np.linspace(0, number_of_generation, number_of_generation)
    y_vec = np.linspace(0, 0, number_of_generation)
    y2_vec = np.linspace(0, 0, number_of_generation)
    line1, line2 = [], []
    historic = [generate_first_population(size_population, max_value, prices)]
    for i in range(number_of_generation):
        x = next_generations(historic[i], max_value, best_sample, lucky_few, number_of_child, chance_of_mutation, prices)
        historic.append(x)
        x = get_best_inidividual_from_population(x, max_value)
        # evolutionFitnessS.append(x['score'])
        # evolutionFitnessL.append(len(x['items']))
        # y_vec[-1] = evolutionFitnessL[-1]
        # y2_vec[-1] = evolutionFitnessS[-1]
        # line1, line2 = lp(x_vec, y_vec, line1, line2=line2, y2_data=y2_vec)
        # y_vec = np.append(y_vec[1:], 0.0)
        # y2_vec = np.append(y2_vec[1:], 0.0)
    return historic


def get_best_inidividual_from_population(population, max_value):
    c = compute_perfect_population(population, max_value)
    return list(c.values())[0]


def get_list_best_individual_from_historique(historic, max_value):
    best_individuals = []
    for population in historic:
        best_individuals.append(get_best_inidividual_from_population(population, max_value))
    return best_individuals


def print_simple_result(historic, max_value, number_of_generations, i=False):
    result = get_list_best_individual_from_historique(historic, max_value)[number_of_generations - 1]
    if i is not False:
        print(f'Score: / {result["score"]} / Value: {str(result["value"])} - {i}')
    else:
        print(f'Score: / {result["score"]} / Value: {str(result["value"])}')
    return result


def evolutionBestFitness(historic, max_value):
    evolutionFitnessS = []
    evolutionFitnessL = []
    for population in historic:
        x = get_best_inidividual_from_population(population, max_value)
        evolutionFitnessS.append(x['score'])
        evolutionFitnessL.append(len(x['items']))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ln1 = ax.plot(range(len(historic)), evolutionFitnessL, '-', label='Length', color='g')
    ln2 = ax2.plot(range(len(historic)), evolutionFitnessS, '-', label='Score', color='b')

    lns = ln1 + ln2
    lnl = [l.get_label() for l in lns]
    plt.legend(lns, lnl, loc=0)
    ax.set_xlabel('generation')
    ax.set_ylabel('Length')
    ax2.set_ylabel('Score')
    plt.show()


def evolutionAverageFitness(historic, max_value, size_population):
    evolutionFitnessS = []
    evolutionFitnessL = []
    for population in historic:
        populationPerf = compute_perfect_population(population, max_value)
        averageFitnessS = 0
        averageFitnessL = 0
        for individual in populationPerf:
            averageFitnessS += populationPerf[individual]['score']
            averageFitnessL += len(populationPerf[individual]['items'])
        evolutionFitnessS.append(averageFitnessS/size_population)
        evolutionFitnessL.append(averageFitnessL/size_population)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ln1 = ax.plot(range(len(historic)), evolutionFitnessL, '-', label='Length', color='g')
    ln2 = ax2.plot(range(len(historic)), evolutionFitnessS, '-', label='Score', color='b')

    lns = ln1 + ln2
    lnl = [l.get_label() for l in lns]
    plt.legend(lns, lnl, loc=0)
    ax.set_xlabel('generation')
    ax.set_ylabel('Length')
    ax2.set_ylabel('Score')
    plt.show()


bag_max_value = 8255383
prices = [116273, 71, 8, 2, 300, 20, 135, 50]
population_size = 250
bag_best_sample = 25
bag_lucky_few = 25
number_of_children = 10
number_of_generation = 400
chance_of_mutation = 20


def run(bag_max_value, prices, population_size, bag_best_sample, bag_lucky_few, number_of_children, number_of_generation, chance_of_mutation,
        i=False):
    if (bag_best_sample + bag_lucky_few) / 2 * number_of_children != population_size:
        print(f'population ize not stable - {i}')
        return None
    else:
        historic = multiple_generations(number_of_generation, bag_max_value, population_size, bag_best_sample, bag_lucky_few, number_of_children,
                                        chance_of_mutation, prices)

        p = print_simple_result(historic, bag_max_value, number_of_generation, i)

        # evolutionBestFitness(historic, bag_max_value)
        # evolutionAverageFitness(historic, bag_max_value, population_size)
        return p


def optimize_run(pop_size, sampl, luck_few, num_chil, num_gen, mut, trials):
    runs = []
    stats = OrderedDict()
    v = 0
    for a in range(1, pop_size):
        for b in range(1, sampl):
            for c in range(1, luck_few):
                for d in range(1, num_chil):
                    for e in range(1, num_gen):
                        for f in range(1, mut):
                            z = []
                            for i in range(trials):
                                x = run(bag_max_value, prices, a, b, c, d, e, f, v)
                                z.append(x)
                                v += 1
                            best_trial = z[0]
                            for i in z:
                                try:
                                    if best_trial['score'] < i['score']:
                                        best_trial = i
                                except TypeError:
                                    pass
                            stats[a+b+c+d+e+f-6] = {
                                'population_size': pop_size,
                                'best_sample': sampl,
                                'lucky_few': luck_few,
                                'number_of_children': num_chil,
                                'number_of_generation': num_gen,
                            }

    best = runs[0]
    x = 0
    a = 0
    for i in runs:
        if best['score'] > i['score']:
            best = i
            x = a
        a += 1
    print(best)
    print(stats[x])


optimize_run(population_size, bag_best_sample, bag_lucky_few, number_of_children, number_of_generation, chance_of_mutation, 5)
# run(bag_max_value, prices, population_size, bag_best_sample, bag_lucky_few, number_of_children, number_of_generation, chance_of_mutation)

