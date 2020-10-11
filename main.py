# -*- coding: utf-8 -*-
from util import *

np.random.seed(16516)


def main():
    fitness = mlrose.OneMax()
    optimization_alg(fitness, 'OneMax')

    fitness = mlrose.FourPeaks()
    optimization_alg(fitness, 'FourPeaks')

    fitness = mlrose.FlipFlop()
    optimization_alg(fitness, 'FlipFlop')

    fitness = mlrose.FlipFlop()
    simulated_annealing(fitness, 'FlipFlop')

    fitness = mlrose.FlipFlop()
    genetic_algorithm(fitness, 'FlipFlop')
    fitness = mlrose.OneMax()
    genetic_algorithm(fitness, 'OneMax')
    fitness = mlrose.FourPeaks()
    genetic_algorithm(fitness, 'FourPeaks')

    fitness = mlrose.OneMax()
    mimic(fitness, 'OneMax')

    neural_network(hidden_nodes=[128])
    neural_network(hidden_nodes=[32])


if __name__ == '__main__':
    main()
