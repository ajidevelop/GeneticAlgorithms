__author__ = 'LobaAjisafe'

import random
import operator
import matplotlib.pyplot as plt

def fitness(password, test_word):
    """
    Determines which word has the best score
    :param password: actual password (str)
    :param test_word: test password (str)
    :return: score
    """
    score = 0
    for i in range(len(password)):
        if password[i] == test_word[i]:
            score += 1
    return score * 100 / len(password)


# Creating Population


def generateAWord(length):
    """
    :param length: length of word (int)
    :return: word (str)
    """
    result = ''
    for i in range(length):
        result += chr(97 + int(25 * random.random()))
    return result


def generateFirstPopulation(sizePopulation, password):
    """
    creates the first population
    :param sizePopulation: population size (int)
    :param password: actual password (str)
    :return: set of words (list)
    """
    population = []
    for i in range(sizePopulation):
        population.append(generateAWord(len(password)))
    return population


def computePerfectPopulation(population, password):
    """
    takes the best of each population
    :param population: population (list)
    :param password: actual password (str)
    :return:
    """

    population_perf = {}
    for individual in population:
        population_perf[individual] = fitness(password, individual)
    return sorted(population_perf.items(), key=operator.itemgetter(1), reverse=True) # set reverse to False


def selectFromPopulation(population_sorted, best_sample, lucky_few):
    """
    makes new population
    :param population_sorted: sorted dictionary of population (dict)
    :param best_sample:
    :param lucky_few:
    :return:
    """

    next_generation = []
    for i in range(best_sample):
        next_generation.append(population_sorted[i][0])
    for i in range(lucky_few):
        next_generation.append(random.choice(population_sorted)[0])
    random.shuffle(next_generation)
    return next_generation


# breeding


def createChild(individual1, individual2):
    """
    :param individual1:
    :param individual2:
    :return:
    """
    child = ''
    for i in range(len(individual1)):
        if int(100 * random.random()) < 50:
            print(individual1)
            child += individual1[i]
        else:
            print(individual2)
            child += individual2[i]
    return child


def createChildren(breeders, number_of_child):
    """
    :param breeders:
    :param number_of_child:
    :return:
    """
    next_population = []
    for i in range(len(breeders)//2):
        for j in range(number_of_child):
            next_population.append(createChild(breeders[i], breeders[len(breeders) - 1 - i]))
    return next_population


def mutateWord(word):
    """
    :param word:
    :return:
    """
    index_modification = int(random.random() * len(word))
    if index_modification == 0:
        word = chr(97 + int(25 + random.random())) + word[1:]
    else:
        word = word[:index_modification] + chr(97 + int(25 + random.random())) + word[index_modification+1:]
    return word


def mutatePopulation(population, chance_of_mutation):
    """
    :param population:
    :param chance_of_mutation:
    :return:
    """
    for i in range(len(population)):
        if random.random() * 100 < chance_of_mutation:
            population[i] = mutateWord(population[i])
    return population


def nextGeneration(firstGeneration, password, best_sample, lucky_few, number_of_child, chance_of_mutation):
    """
    :param firstGeneration:
    :param password:
    :param best_sample:
    :param lucky_few:
    :param number_of_child:
    :param chance_of_mutation:
    :return:
    """
    population_sorted = computePerfectPopulation(firstGeneration, password)
    next_breeders = selectFromPopulation(population_sorted, best_sample, lucky_few)
    next_population = createChildren(next_breeders, number_of_child)
    next_generation = mutatePopulation(next_population, chance_of_mutation)
    return next_generation


def multipleGenerations(number_of_generation, password, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation):
    historic = [generateFirstPopulation(size_population, password)]
    for i in range(number_of_generation):
        historic.append(nextGeneration(historic[i], password, best_sample, lucky_few, number_of_child, chance_of_mutation))
    return historic


def getBestIndividualFromPopulation(population, password):
    return computePerfectPopulation(population, password)[0]


def getListBestInidividualFromHistorique(historic, password):
    bestIndividuals = []
    for population in historic:
        bestIndividuals.append(getBestIndividualFromPopulation(population, password))
    return bestIndividuals


def printSimpleResult(historic, password, number_of_generation):
    result = getListBestInidividualFromHistorique(historic, password)[number_of_generation - 1]
    print(f'solution: / {result[0]} / de fitness: {str(result[1])}')


def evolutionBestFitness(historic, password):
    plt.axis([0,len(historic),0,105])
    plt.title(password)

    evolutionFitness = []
    for population in historic:
        evolutionFitness.append(getBestIndividualFromPopulation(population, password)[1])
    plt.plot(evolutionFitness)
    plt.ylabel('fitness best individual')
    plt.xlabel('generation')
    plt.show()


def evolutionAverageFitness(historic, password, size_population):
    plt.axis([0,len(historic),0,105])
    plt.title(password)

    evolutionFitness = []
    for population in historic:
        populationPerf = computePerfectPopulation(population, password)
        averageFitness = 0
        for individual in populationPerf:
            averageFitness += individual[1]
        evolutionFitness.append(averageFitness/size_population)
    plt.plot(evolutionFitness)
    plt.ylabel('Average fitness')
    plt.xlabel('generation')
    plt.show()


# variables

password = input('Enter Word: ')
size_population = 100
best_sample = 20
lucky_few = 20
number_of_child = 5
number_of_generation = 200
chance_of_mutation = 5

# program

if (best_sample + lucky_few) / 2 * number_of_child != size_population:
    print('population size not stable')
else:
    historic = multipleGenerations(number_of_generation, password, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation)

    printSimpleResult(historic, password, number_of_generation)

    evolutionBestFitness(historic, password)
    evolutionAverageFitness(historic, password, size_population)
