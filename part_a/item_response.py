from utils import *
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    sum = 0.
    for i in range(len(data["user_id"])):
        user = data["user_id"][i]
        question = data["question_id"][i]
        correct = data["is_correct"][i]
        b = beta[question]
        t = theta[user]
        log_beta_theta = np.logaddexp(0, b - t)
        log_theta_beta = np.logaddexp(0, t - b)
        sum += -correct * log_beta_theta + (correct-1) * log_theta_beta
    log_lklihood = sum
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
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
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    de_theta = np.zeros((theta.shape[0]))
    de_beta = np.zeros((beta.shape[0]))
    for i in range(len(data["user_id"])):
        user = data["user_id"][i]
        question = data["question_id"][i]
        correct = data["is_correct"][i]
        b = beta[question]
        t = theta[user]
        fraction = sigmoid(t - b)
        de_theta[user] += correct - fraction
        de_beta[question] += fraction - correct
    theta += lr * de_theta
    beta += lr * de_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


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
    # TODO: Initialize theta and beta.
    user_count = len(set(data["user_id"]))
    question_count = len(set(data["question_id"]))
    theta = np.ones((user_count))
    beta = np.ones((question_count))
    val_acc_lst = []
    neg_lld_trains = []
    neg_lld_vals = []
    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        neg_lld_vals.append(-neg_lld_val)
        neg_lld_trains.append(-neg_lld_train)
        if i == iterations-1:
            print("NLLK: {} \t Score: {}".format(neg_lld_train, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst[len(val_acc_lst)-1], neg_lld_trains, neg_lld_vals


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # learning_rates = [0.01, 0.03, 0.05, 0.07, 0.1]
    # num_iterations = [10, 30, 50, 70, 100]
    # for lr in learning_rates:
    #     for iterations in num_iterations:
    #         print("Learning Rate: " + str(lr) + " Iterate " +
    #               str(iterations) + " times")
    #         irt(train_data, val_data, lr, iterations)
    lr = 0.01
    iterations = 30
    print("Learning Rate: " + str(lr) + " Iterate " + str(iterations) +
          " times")
    theta, beta, score_val, neg_lld_trains, neg_lld_vals = \
        irt(train_data, val_data, lr, iterations)
    _, _, score_test, _, _ = \
        irt(train_data, test_data, lr, iterations)
    print("Validation Accuracy: ", score_val)
    print("Test Accuracy: ", score_test)
    plt.figure()
    plt.plot(range(len(neg_lld_trains)), neg_lld_trains)
    plt.plot(range(len(neg_lld_trains)), neg_lld_vals)
    plt.xlabel("iteration")
    plt.ylabel("log-likelihood")
    plt.title("Training Curve: log-likelihoods as function of iteration")
    plt.legend(["Training Data", "Validation Data"])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    # Choose question 1, 2, 3, corresponding to question index 0, 1, 2 in data
    plt.figure()
    theta_sorted = np.sort(theta)
    for j in range(3):
        prob = []
        beta_i = beta[j]
        for i in theta_sorted:
            prob.append(sigmoid(i - beta_i))
            print(i, sigmoid(i - beta_i))
        plt.plot(theta_sorted, prob)
    plt.xlabel("Theta")
    plt.ylabel("Probability of the correct response")
    plt.title("Probability of the correct response on j1, j2, j3 vs. Theta")
    plt.legend(["Question 1(j1)", "Question 2(j2)", "Question 3(j3)"])
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
