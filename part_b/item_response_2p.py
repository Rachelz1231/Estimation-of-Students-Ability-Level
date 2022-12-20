import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
from utils import *

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))

def neg_log_likelihood(data, theta, alpha, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """

    sum = 0.
    for i in range(len(data["user_id"])):
        user = data["user_id"][i]
        question = data["question_id"][i]
        correct = data["is_correct"][i]
        b = beta[question]
        a = alpha[question]
        t = theta[user]
        log_beta_theta = np.logaddexp(0, a * (b - t))
        log_theta_beta = np.logaddexp(0, a * (t - b))
        sum += -correct * log_beta_theta + (correct-1) * log_theta_beta
    log_lklihood = sum

    return -log_lklihood

def update_theta_alpha_beta(data, lr, theta, alpha, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    de_theta = np.zeros(theta.shape[0])
    de_alpha = np.zeros(alpha.shape[0])
    de_beta = np.zeros(beta.shape[0])

    for i in range(len(data["user_id"])):
        user = data["user_id"][i]
        question = data["question_id"][i]
        correct = data["is_correct"][i]
        
        t = theta[user]
        a = alpha[question]
        b = beta[question]

        de_theta[user] += (correct * a - a * sigmoid(a * (t - b)))
        de_alpha[question] += correct * (t - b) - (t - b) * sigmoid(a * (t - b))
        de_beta[question] += (a * sigmoid(a * (t - b)) - correct * a)
    theta += lr * de_theta
    alpha += lr * de_alpha
    beta += lr * de_beta
        
    return theta, beta, alpha

def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.ones(542)
    beta = np.ones(1774)
    alpha = np.ones(1774)
    
    neg_lld_trains = []
    neg_lld_vals = []
    accuracies = []

    val_score = 0

    for _ in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, alpha=alpha, beta=beta)
        neg_lld_trains.append(-neg_lld)
        neg_lld = neg_log_likelihood(val_data, theta=theta, alpha=alpha, beta=beta)
        neg_lld_vals.append(-neg_lld)

        val_score = evaluate(data=val_data, theta=theta, alpha=alpha, beta=beta)
        accuracies.append(val_score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, val_score))
        theta, beta, alpha = update_theta_alpha_beta(data, lr, theta, alpha, beta)

    return val_score, neg_lld_trains, neg_lld_vals, theta, beta, alpha

def evaluate(data, theta, alpha, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    corrects = []
    for i in range(len(data["user_id"])):
        user = data["user_id"][i]
        question = data["question_id"][i]
        thre = sigmoid((alpha[question] * (theta[user] - beta[question])).sum())
        if thre >= 0.5:
            corrects.append(1)
        else:
            corrects.append(0)
    return np.sum((data["is_correct"] == np.array(corrects))) / len(data["is_correct"])


def enhancement_after_split(train_data, val_data, test_data):

    print("Train Accuracy: ", irt(train_data, train_data, 0.01, 70))
    print("Validation Accuracy: ", irt(train_data, val_data, 0.01, 70))
    print("Test Accuracy: ", irt(train_data, test_data, 0.01, 70))



def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Set the hyper parameters
    # learning_rates = [0.01, 0.03, 0.05, 0.07, 0.1]
    # num_iterations = [10, 30, 50, 70, 100]
    learning_rates = [0.03]
    num_iterations = [100]
    hypers = []
    results = []
    for num in num_iterations:
        for lr in learning_rates:
            val_score = irt(train_data, val_data, lr=lr, iterations=num)[0]
            results.append(val_score)
            hypers.append([lr, num])
            print("lr: {} \t iterations: {} \t Score: {}".format(lr, num, val_score))
    best_score_index = results.index(max(results))
    best_hyper = hypers[best_score_index]
    print("Best learning rate: ", best_hyper)

    print("Train Accuracy: ", irt(train_data, train_data, best_hyper[0], best_hyper[1])[0])
    print("Validation Accuracy: ", irt(train_data, val_data, best_hyper[0], best_hyper[1])[0])
    print("Test Accuracy: ", irt(train_data, test_data, best_hyper[0], best_hyper[1])[0])


def plot():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    lr = 0.01
    iterations = 70
    val_score, neg_lld_trains, neg_lld_vals, theta, beta, alpha= \
        irt(train_data, val_data, lr, iterations)
    plt.figure()
    plt.plot(range(len(neg_lld_trains)), neg_lld_trains)
    plt.plot(range(len(neg_lld_trains)), neg_lld_vals)
    plt.xlabel("iteration")
    plt.ylabel("log-likelihood")
    plt.title("Training Curve: log-likelihoods as function of iteration")
    plt.legend(["Training Data", "Validation Data"])

    plt.figure()
    theta_sorted = np.sort(theta)
    for j in range(3):
        prob = []
        beta_i = beta[j]
        for i in theta_sorted:
            prob.append(sigmoid(alpha[j]*(i - beta_i)))
        plt.plot(theta_sorted, prob)
    plt.xlabel("Theta")
    plt.ylabel("Probability of the correct response")
    plt.title("Probability of the correct response on j1, j2, j3 vs. Theta")
    plt.legend(["Question 1(j1)", "Question 2(j2)", "Question 3(j3)"])
    plt.show()

if __name__ == "__main__":
    # main()
    plot()