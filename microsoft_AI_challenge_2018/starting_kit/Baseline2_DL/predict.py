from __future__ import print_function
import os
import sys
import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_recall_fscore_support

import cntk as C
from cntk.io import (FULL_DATA_SWEEP, INFINITELY_REPEAT, CTFDeserializer,
                     MinibatchSource, StreamDef, StreamDefs)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def GetPredictionOnEvalSet(model, testfile, submissionfile):
    print('Doing predictions on eval set.')
    global q_max_words, p_max_words, emb_dim

    f = open(testfile, 'r', encoding="utf-8")
    all_scores = {}  # Dictionary with key = query_id and value = array of scores for respective passages
    for line in f:
        tokens = line.strip().split("|")
        # tokens[0] will be empty token since the line is starting with |
        x1 = tokens[1].replace("qfeatures", "").strip()  # Query Features
        x2 = tokens[2].replace("pfeatures", "").strip()  # Passage Features
        query_id = tokens[3].replace("qid", "").strip()  # Query_id
        x1 = [float(v) for v in x1.split()]
        x2 = [float(v) for v in x2.split()]
        queryVec = np.array(x1, dtype="float32").reshape(1, q_max_words, emb_dim)
        passageVec = np.array(x2, dtype="float32").reshape(1, p_max_words, emb_dim)
        score = model(queryVec, passageVec)[0][1]  # do forward-prop on model to get score
        if(query_id in all_scores):
            all_scores[query_id].append(score)
        else:
            all_scores[query_id] = [score]
    fw = open(submissionfile, "w", encoding="utf-8")
    for query_id in all_scores:
        scores = all_scores[query_id]
        scores_str = [str(sc) for sc in scores]  # convert all scores to string values
        scores_str = "\t".join(scores_str)  # join all scores in list to make it one string with  tab delimiter.
        fw.write(query_id + "\t" + scores_str + "\n")
    fw.close()


if __name__ == "__main__":
    epoch = 150
    model = C.load_model(f'data/models/CNN_{epoch}.dnn')

    # *****Hyper-Parameters******
    q_max_words = 12
    p_max_words = 50
    emb_dim = 50
    num_classes = 2
    minibatch_size = 250
    epoch_size = 100000  # No.of samples in training set
    total_epochs = 200  # Total number of epochs to run
    query_total_dim = q_max_words * emb_dim
    label_total_dim = num_classes
    passage_total_dim = p_max_words * emb_dim

    testSetFileName = "data/EvaluationData.ctf"
    submissionFileName = "data/answer.tsv"

    GetPredictionOnEvalSet(model, testSetFileName, submissionFileName)  # Get Predictions on Evaluation Set
