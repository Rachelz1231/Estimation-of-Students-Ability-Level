###
#  using cosine similarity instead of euclidean distance 
#  to find the kth nearest neighbours
###
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
from utils import *
import math
import matplotlib.pyplot as plt

def cosine_similarity(c1, c2):
    #from https://www.linkedin.com/pulse/cosine-similarity-classification-michael-lin/
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    mat = matrix.copy()
    
    #from sparse matrix evaluate
    #incomplete 
    acc = sparse_matrix_evaluate(valid_data, mat)
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    K = [1, 6, 11, 16, 21, 26]
    acc_user = []
    print("Using user-based collaborative filtering on validation set: ")
    for k in K:
        print("With k=" + str(k))
        acc_user.append(knn_impute_by_user(sparse_matrix, val_data, k))

    k_user = 11
    print("Using user-based collaborative filtering, k=11 on test set: ")
    knn_impute_by_user(sparse_matrix, test_data, k_user)

    acc_item = []
    print("Using item-based collaborative filtering on validation set: ")
    for k in K:
        print("With k=" + str(k))
        acc_item.append(knn_impute_by_item(sparse_matrix, val_data, k))

    k_item = 21
    print("Using item-based collaborative filtering, k=21 on test set: ")
    knn_impute_by_item(sparse_matrix, test_data, k_item)

    plt.plot(K, acc_user)
    plt.plot(K, acc_item)
    plt.xlabel("value of k for k-Nearest Neighbor")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of collaborative filtering by kNN")
    plt.legend(["User-based", "Item-based"])
    plt.show()
