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
            if (row[0].isdigit()):
                students["user_id"].append(int(row[0]))
                students["gender"].append(int(row[1]))
    return students

def by_gender(data):
    """
    separate gender groups by 1 and 2
    """
    students = load_csv()
    ids = students["user_id"]
    genders = students["gender"]
    d1, d2 = [], []
    oids = data["user_id"]
    oqs = data["question_id"]
    corrects = data["is_correct"]
    splitted_data = [{"user_id": [], "question_id": [], "is_correct": []},
                    {"user_id": [], "question_id": [], "is_correct": []}]
    # seperate by 1 and 2
    for i in range(len(ids)):
        if genders[i] == 1:
            d1.append(ids[i])
        elif genders[i] == 2:
            d2.append(ids[i])

    for i in range(len(oids)):
        if oids[i] in d1:
            splitted_data[0]["user_id"].append(oids[i])
            splitted_data[0]["question_id"].append(oqs[i])
            splitted_data[0]["is_correct"].append(corrects[i])
        elif oids[i] in d2:
            splitted_data[1]["user_id"].append(oids[i])
            splitted_data[1]["question_id"].append(oqs[i])
            splitted_data[1]["is_correct"].append(corrects[i])
    return splitted_data

if __name__ == '__main__':
    train_data_dic = load_train_csv("../data")
    valid_data_dic = load_valid_csv("../data")
    test_data_dic = load_public_test_csv("../data")
    splitted_train_data = by_gender(train_data_dic)
    splitted_valid_data = by_gender(valid_data_dic)
    splitted_test_data = by_gender(test_data_dic)
    for i in range(2):
        enhancement_after_split(splitted_train_data[i], splitted_valid_data[i], splitted_test_data[i])
