import numpy as np 
from matplotlib import pyplot as plt 
import item_response_2p as p
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
from utils import *
sys.path.append(os.path.abspath(os.path.join('..','part_a')))
import item_response as b

def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    lr = 0.05
    iterations = 50
    ba = b_irt(train_data, val_data, lr, iterations)[-1]
    pa = p_irt(train_data, val_data, lr, iterations)[-1]
    plt.figure()
    plt.plot(list(range(3, 51)), pa[2:])
    plt.plot(list(range(3, 51)), ba[2:])
    plt.xlabel("iterations")
    plt.ylabel("validation accuracy")
    plt.title("Comparison between accuarcy of models with lr 0.05 over 50 iterations")
    plt.legend(["2P Model", "Base Model"])
    plt.show()

def b_irt(data, val_data, lr, iterations):
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
        neg_lld_train = b.neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = b.neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = b.evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        neg_lld_vals.append(-neg_lld_val)
        neg_lld_trains.append(-neg_lld_train)
        if i == iterations-1:
            print("NLLK: {} \t Score: {}".format(neg_lld_train, score))
        theta, beta = b.update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst[len(val_acc_lst)-1], neg_lld_trains, neg_lld_vals, val_acc_lst

def p_irt(data, val_data, lr, iterations):
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
        neg_lld = p.neg_log_likelihood(data, theta=theta, alpha=alpha, beta=beta)
        neg_lld_trains.append(-neg_lld)
        neg_lld = p.neg_log_likelihood(val_data, theta=theta, alpha=alpha, beta=beta)
        neg_lld_vals.append(-neg_lld)

        val_score = p.evaluate(data=val_data, theta=theta, alpha=alpha, beta=beta)
        accuracies.append(val_score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, val_score))
        theta, beta, alpha = p.update_theta_alpha_beta(data, lr, theta, alpha, beta)

    return val_score, neg_lld_trains, neg_lld_vals, theta, beta, alpha, accuracies

if __name__ == '__main__':
    main()