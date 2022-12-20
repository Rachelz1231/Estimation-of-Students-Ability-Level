# TODO: complete this file.
from utils import *
from torch.autograd import Variable
import numpy as np
import torch
import neural_network


def generate_sample():
    # Set up
    zero_train_matrix, train_matrix, valid_data, test_data = neural_network.load_data()
    num_student = train_matrix.shape[0]
    num_question = train_matrix.shape[1]
    sample = np.random.randint(0, num_student, num_student)
    sample_zero_train_matrix = np.zeros(zero_train_matrix.shape)
    sample_train_matrix = np.zeros(train_matrix.shape)

    for i in range(len(sample)):
        sample_zero_train_matrix[i] = zero_train_matrix[sample[i]]
        sample_train_matrix[i] = train_matrix[sample[i]]

    new_zero_train_matrix = torch.FloatTensor(sample_zero_train_matrix)
    new_train_matrix = torch.FloatTensor(sample_train_matrix)

    # Use the k, learning rate, lambda, epoch from question 3
    k = 50
    lr = 0.05
    epoch = 10
    lamb = 0.001

    # Create model
    model = neural_network.AutoEncoder(num_question, k)

    neural_network.train(model, lr, lamb, new_train_matrix, new_zero_train_matrix, valid_data, epoch)

    return model, new_zero_train_matrix


def bagging(model, train_data1, train_data2, train_data3, valid_data):
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        input1 = Variable(train_data1[u]).unsqueeze(0)
        output1 = model(input1)
        input2 = Variable(train_data2[u]).unsqueeze(0)
        output2 = model(input2)
        input3 = Variable(train_data3[u]).unsqueeze(0)
        output3 = model(input3)

        guess = ((output1[0][valid_data["question_id"][i]].item()+
                 output2[0][valid_data["question_id"][i]].item()+
                 output3[0][valid_data["question_id"][i]].item()))*1/3 >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


if __name__ == "__main__":
    zero_train_matrix, train_matrix, valid_data, test_data = neural_network.load_data()
    # Generate three model
    model, train_data1= generate_sample()
    redundant_model1, train_data2 = generate_sample()
    redundant_model2, train_data3 = generate_sample()

    valid_accuracy = bagging(model, train_data1, train_data2, train_data3, valid_data)
    test_accuracy = bagging(model, train_data1, train_data2, train_data3, test_data)

    print("Valid accuracy:", valid_accuracy)
    print("Test accuracy:", test_accuracy)

