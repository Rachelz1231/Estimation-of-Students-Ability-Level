from posixpath import split
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
from utils import *
import pandas as pd
import numpy as np
from item_response_2p import *


def load_csv():
    students = {"user_id": [], "gender": [], "age": []}
    file = os.path.join("../data", "student_meta.csv")
    with open(file, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if (row[0].isdigit() and row[2][:4] != ''):
                students["user_id"].append(int(row[0]))
                students["age"].append(2021-int(row[2][:4]))
    return students

def by_age_quartiles(data):
    """
    separate age groups by quartiles, d1, d2, d3, d4
    """
    students = load_csv()
    ids = students["user_id"]
    ages = students["age"]
    d1, d2, d3, d4 = [], [], [], []
    oids = data["user_id"]
    oqs = data["question_id"]
    corrects = data["is_correct"]
    splitted_data = [{"user_id": [], "question_id": [], "is_correct": []},
                    {"user_id": [], "question_id": [], "is_correct": []},
                    {"user_id": [], "question_id": [], "is_correct": []},
                    {"user_id": [], "question_id": [], "is_correct": []}]
    # seperate by quartiles
    for i in range(len(ids)):
        if np.quantile(ages, .25) <= ages[i] < np.quantile(ages, .5):
            d2.append(ids[i])
        elif np.quantile(ages, .5) <= ages[i] < np.quantile(ages, .75):
            d3.append(ids[i])
        elif np.quantile(ages, .75) <= ages[i]:
            d4.append(ids[i])
        else:
            d1.append(ids[i])

    for i in range(len(oids)):
        if oids[i] in d1:
            splitted_data[0]["user_id"].append(oids[i])
            splitted_data[0]["question_id"].append(oqs[i])
            splitted_data[0]["is_correct"].append(corrects[i])
        elif oids[i] in d2:
            splitted_data[1]["user_id"].append(oids[i])
            splitted_data[1]["question_id"].append(oqs[i])
            splitted_data[1]["is_correct"].append(corrects[i])
        elif oids[i] in d3:
            splitted_data[2]["user_id"].append(oids[i])
            splitted_data[2]["question_id"].append(oqs[i])
            splitted_data[2]["is_correct"].append(corrects[i])
        else:
            splitted_data[3]["user_id"].append(oids[i])
            splitted_data[3]["question_id"].append(oqs[i])
            splitted_data[3]["is_correct"].append(corrects[i])
    return splitted_data

def by_age_teens(data):
    """
    separate age groups by teens(age over 13 vs under 13)
    """
    students = load_csv()
    ids = students["user_id"]
    ages = students["age"]
    d1, d2 = [], []
    oids = data["user_id"]
    oqs = data["question_id"]
    corrects = data["is_correct"]
    splitted_data = [{"user_id": [], "question_id": [], "is_correct": []},
                    {"user_id": [], "question_id": [], "is_correct": []}]
    # seperate by teens
    for i in range(len(ids)):
        if 13 <= ages[i]:
            d1.append(ids[i])
        else:
            d2.append(ids[i])

    for i in range(len(oids)):
        if oids[i] in d1:
            splitted_data[0]["user_id"].append(oids[i])
            splitted_data[0]["question_id"].append(oqs[i])
            splitted_data[0]["is_correct"].append(corrects[i])
        else:
            splitted_data[1]["user_id"].append(oids[i])
            splitted_data[1]["question_id"].append(oqs[i])
            splitted_data[1]["is_correct"].append(corrects[i])
    return splitted_data

if __name__ == '__main__':
    train_data_dic = load_train_csv("../data")
    valid_data_dic = load_valid_csv("../data")
    test_data_dic = load_public_test_csv("../data")
    splitted_train_data = by_age_quartiles(train_data_dic)
    splitted_valid_data = by_age_quartiles(valid_data_dic)
    splitted_test_data = by_age_quartiles(test_data_dic)
    for i in range(4):
        enhancement_after_split(splitted_train_data[i], splitted_valid_data[i], splitted_test_data[i])

    splitted_train_data = by_age_teens(train_data_dic)
    splitted_valid_data = by_age_teens(valid_data_dic)
    splitted_test_data = by_age_teens(test_data_dic)
    for i in range(2):
        enhancement_after_split(splitted_train_data[i], splitted_valid_data[i], splitted_test_data[i])
