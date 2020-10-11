import matplotlib.pyplot as plt
import pandas as pd
import sys
import six
sys.modules['sklearn.externals.six'] = six
import numpy as np
import mlrose
import time
import os
from sklearn.model_selection import train_test_split, learning_curve
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_curve, auc


def optimization_alg(fitness, name, queens=False):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 4))
    fig.tight_layout(pad=3.0)
    for alg in ['Randomized Hill Climb', 'Simulated Annealing', 'Genetic Algorithm', 'MIMIC']:
        bf, size, times = [], [], []

        for ind, i in enumerate([5, 50, 500]):
            start = time.time()
            problem = mlrose.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True,
                                         max_val=10)
            if not queens:
                init_state = np.random.choice([0, 1], size=i, replace=True)
            else:
                init_state = np.random.choice(np.arange(i - 1), size=i, replace=True)

            if alg == 'Randomized Hill Climb':
                best_state, best_fitness, curve = mlrose.random_hill_climb(problem, max_attempts=25, max_iters=1000,
                                                                           restarts=0,
                                                                           init_state=init_state, curve=True,
                                                                           random_state=0)
            elif alg == 'Simulated Annealing':
                best_state, best_fitness, curve = mlrose.simulated_annealing(problem, max_attempts=25, max_iters=1000,
                                                                             init_state=init_state, curve=True,
                                                                             random_state=0)
            elif alg == 'Genetic Algorithm':
                best_state, best_fitness, curve = mlrose.genetic_alg(problem, pop_size=i, mutation_prob=0.1,
                                                                     max_attempts=25, max_iters=1000,
                                                                     curve=True, random_state=0)
            elif alg == 'MIMIC':
                best_state, best_fitness, curve = mlrose.mimic(problem, pop_size=200, max_attempts=25, max_iters=1000,
                                                               curve=True, fast_mimic=True, random_state=0)

            bf.append(best_fitness)
            size.append(i)
            times.append(time.time() - start)
            axes[ind].plot(np.arange(len(curve)), curve, label=alg)
            axes[ind].legend()
            axes[ind].set_title(f'Problem Size = {i}')
            axes[ind].set_xlabel('Iteration')
            axes[ind].set_ylabel('Fitness State')

        axes[3].plot(size, times, label=alg)
        axes[3].set_title('Algorithm Time per Problem Size')
        axes[3].set_xlabel('Problem Size')
        axes[3].set_ylabel('Time (seconds)')
        axes[3].legend()
    fig.suptitle(f'{name} Optimization Problem', fontsize=16, y=1.0)
    plt.savefig(f'{fitness}-{name}.png')


def simulated_annealing(fitness, name, queens=False):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    fig.tight_layout(pad=3.0)
    bf, size, times = [], [], []

    ind = 0
    for i, alg in zip([mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()],
                      ['Geometric Decay', 'Arithmetric Decay', 'Exponential Decay']):
        start = time.time()
        problem = mlrose.DiscreteOpt(length=500, fitness_fn=fitness, maximize=True,
                                     max_val=10)
        init_state = np.random.choice([0, 1], size=500, replace=True)
        best_state, best_fitness, curve = mlrose.simulated_annealing(problem, schedule=i, max_attempts=25,
                                                                     max_iters=5000,
                                                                     init_state=init_state, curve=True, random_state=0)

        bf.append(best_fitness)
        size.append(i)
        times.append(time.time() - start)
        axes.plot(np.arange(len(curve)), curve, label=alg)
        axes.legend()
        # axes.set_title(f'{alg}')
        axes.set_xlabel('Iteration')
        axes.set_ylabel('Fitness State')
        ind += 1
    fig.suptitle(f'{name} Simulated Annealing Schedules', fontsize=16, y=1.0)
    plt.savefig(f'{fitness}-{name}.png')


def genetic_algorithm(fitness, name, queens=False):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    fig.tight_layout(pad=3.0)
    bf, size, times = [], [], []

    ind = 0
    for i, alg in zip([0.25, 0.5, 0.75],
                      ['Mutation Prob. 0.25', 'Mutation Prob. 0.50', 'Mutation Prob. 0.75']):
        start = time.time()
        problem = mlrose.DiscreteOpt(length=500, fitness_fn=fitness, maximize=True,
                                     max_val=10)
        init_state = np.random.choice([0, 1], size=500, replace=True)
        best_state, best_fitness, curve = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=i, max_attempts=25,
                                                             max_iters=1000, curve=True, random_state=0)

        bf.append(best_fitness)
        size.append(i)
        times.append(time.time() - start)
        axes.plot(np.arange(len(curve)), curve, label=alg)
        axes.legend()
        # axes.set_title(f'{alg}')
        axes.set_xlabel('Iteration')
        axes.set_ylabel('Fitness State')
        ind += 1
    fig.suptitle(f'{name} Genetic Algorithms Mutation Rate', fontsize=16, y=1.0)
    plt.savefig(f'{fitness}-{name}.png')


def mimic(fitness, name, queens=False):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    fig.tight_layout(pad=3.0)
    bf, size, times = [], [], []

    ind = 0
    for i, alg in zip([0.25, 0.5, 0.75],
                      ['Keep Percent 0.25', 'Keep Percent 0.50', 'Keep Percent 0.75']):
        start = time.time()
        problem = mlrose.DiscreteOpt(length=500, fitness_fn=fitness, maximize=True,
                                     max_val=10)
        init_state = np.random.choice([0, 1], size=500, replace=True)
        best_state, best_fitness, curve = mlrose.mimic(problem, pop_size=200, keep_pct=i, max_attempts=25,
                                                       max_iters=1000, curve=True, random_state=0)

        bf.append(best_fitness)
        size.append(i)
        times.append(time.time() - start)
        axes[0].plot(np.arange(len(curve)), curve, label=alg)
        axes[0].legend()
        # axes.set_title(f'{alg}')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Fitness State')

        axes[1].plot(size, times, label=alg)
        axes[1].set_title('Algorithm Time per Problem Size')
        axes[1].set_xlabel('Problem Size')
        axes[1].set_ylabel('Time (seconds)')
        axes[1].legend()
    fig.suptitle(f'{name} MIMIC Keep Percentage', fontsize=16, y=1.0)
    plt.savefig(f'{fitness}-{name}.png')


def neural_network(hidden_nodes=[128, 2], filename='data/heart_failure_clinical_records_dataset.csv'):
    # Preproccessing
    df = pd.read_csv(filename)
    x = df.copy()
    x.drop(columns='DEATH_EVENT', inplace=True)
    y = df['DEATH_EVENT']
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75,
                                                        random_state=0)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    scaler = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler.transform(X_test)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    for alg in ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent']:
        # for alg in ['gradient_descent']:
        print(alg)
        start = time.time()
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes=hidden_nodes, activation='relu', \
                                         algorithm=alg, max_iters=5000, \
                                         bias=True, is_classifier=True, learning_rate=0.0001, \
                                         random_state=0, curve=True)

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(nn_model1, X_train, y_train, cv=5,
                           train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy',
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Accuracy")
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-',
                     label=f"{alg} Training score")
        # axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
        # label=f"{alg} Cross-validation score")
        axes[0].legend(loc="best")

        nn_model1.fit(X_train, y_train)
        # Predict labels for train set and assess accuracy
        y_train_pred = nn_model1.predict(X_train)
        y_train_accuracy = accuracy_score(y_train, y_train_pred)

        # Predict labels for test set and assess accuracy
        y_test_pred = nn_model1.predict(X_test)
        y_test_accuracy = accuracy_score(y_test, y_test_pred)

        axes[1].plot(nn_model1.fitted_weights, label=f'{alg} final loss: {np.round(nn_model1.loss, 3)}')
        axes[1].legend()
        axes[1].set_ylim(-5, 5)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Fitted Weights')
        axes[1].set_title('Fitted Weights on Training Data')

        axes[2].plot((nn_model1.fitness_curve * -1), label=f'{alg} fit time: {np.round(time.time() - start, 3)}s')
        axes[2].legend()
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Fitness State')
        axes[2].set_title('Fitness Curve')

        fpr, tpr, thresholds = roc_curve(y_train, y_train_pred)
        axes2[0].plot(fpr, tpr, label=f'{alg} AUC: {np.round(auc(fpr, tpr), 3)}')
        axes2[0].legend()
        axes2[0].set_xlabel('False Positive Rate')
        axes2[0].set_ylabel('True Positive Rate')
        axes2[0].set_title('AUROC of Heart Failure Training Data')

        fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
        axes2[1].plot(fpr, tpr, label=f'{alg} AUC: {np.round(auc(fpr, tpr), 3)}')
        axes2[1].legend()
        axes2[1].set_xlabel('False Positive Rate')
        axes2[1].set_ylabel('True Positive Rate')
        axes2[1].set_title('AUROC of Heart Failure Testing Data')
    fig.suptitle(f'Neural Network Training Summary', fontsize=16, y=1.0)
    plt.savefig(f'{fitness}-{name}.png')

