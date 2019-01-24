from __future__ import print_function
import numpy as np
import sys
import os
import cntk as C
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from sklearn.metrics import precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



#Initialize Global variables
validation_query_vectors = []
validation_passage_vectors = []
validation_labels = []
q_max_words=12
p_max_words=50
emb_dim=50

## The following LoadValidationSet method reads ctf format validation file and creates query, passage feature vectors and also copies labels for each pair.
## the created vectors will be useful to find metrics on validation set after training each epoch which will be useful to decide the best model
def LoadValidationSet(validationfile):
    f = open(validationfile,'r',encoding="utf-8")
    for line in f:
        tokens = line.strip().split("|")
        #tokens[0] will be empty token since the line is starting with |
        x1 = tokens[1].replace("qfeatures","").strip() #Query Features
        x2 = tokens[2].replace("pfeatures","").strip() # Passage Features
        y = tokens[3].replace("labels","").strip() # labels
        x1 = [float(v) for v in x1.split()]
        x2 = [float(v) for v in x2.split()]
        y = [int(w) for w in y.split()]
        y = y[1] # label will be at index 1, i.e. if y = "1 0" then label=0 else if y="0 1" then label=1

        validation_query_vectors.append(x1)
        validation_passage_vectors.append(x2)
        validation_labels.append(y)

        #print("1")

    print("Validation Vectors are created")

#The following method defines a CNN network which has series of convolution and max pooling steps on query features and passage features and then a merge step and it follows a fully connected layer
def cnn_network(queryfeatures, passagefeatures, num_classes):
    with C.layers.default_options(activation=C.ops.relu, pad=False):
        convA1 = C.layers.Convolution2D((3,10),4,pad=False,activation=C.tanh,name='convA1')(queryfeatures) #input : 12*50 #output : 4*10*41
        poolA1 = C.layers.MaxPooling((2,3),(2,3),name='poolA1')(convA1)  #output : 4*5*13
        convA2 = C.layers.Convolution2D((2,4),2,pad=False,activation=C.tanh,name='convA2')(poolA1) #output : 2*4*10
        poolA2 = C.layers.MaxPooling((2,2),(2,2),name='poolA2')(convA2)  #output : 2*2*5
        denseA = C.layers.Dense(num_classes*num_classes,activation=C.tanh,name='denseA')(poolA2)  # output : 4

        convB1 = C.layers.Convolution2D((5,10),4,pad=False,activation=C.tanh,name='convB1')(passagefeatures) #input : 50*50  #output : 4*46*41
        poolB1 = C.layers.MaxPooling((5,5),(5,5),name='poolB1')(convB1) #output : 4*9*8
        convB2 = C.layers.Convolution2D((3,3),2,pad=False,activation=C.tanh,name='convB2')(poolB1) #output : 2*7*6
        poolB2 = C.layers.MaxPooling((2,2),(2,2),name='poolB2')(convB2) #output : 2*3*3
        denseB = C.layers.Dense(num_classes*num_classes,activation=C.tanh,name='denseB')(poolB2)  # output : 4


        mergeQP     = C.element_times(denseA,denseB) # output : 4

        model   = C.layers.Dense(num_classes, activation=C.softmax,name="overall")(mergeQP) #outupt : 2


    return model

def create_reader(path, is_training, query_total_dim, passage_total_dim, label_total_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs( queryfeatures = StreamDef(field='qfeatures', shape=query_total_dim,is_sparse=False),
                                                            passagefeatures = StreamDef(field='pfeatures', shape=passage_total_dim,is_sparse=False),
                                                            labels   = StreamDef(field='labels', shape=label_total_dim,is_sparse=False)
                                                            )),
                           randomize=is_training, max_sweeps = INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)

def TrainAndValidate(trainfile):

    #*****Hyper-Parameters******
    q_max_words= 12
    p_max_words = 50
    emb_dim = 50
    num_classes = 2
    minibatch_size = 250
    epoch_size = 100000 #No.of samples in training set
    total_epochs = 200 #Total number of epochs to run
    query_total_dim = q_max_words*emb_dim
    label_total_dim = num_classes
    passage_total_dim = p_max_words*emb_dim


    #****** Create placeholders for reading Training Data  ***********
    query_input_var =  C.ops.input_variable((1,q_max_words,emb_dim),np.float32,is_sparse=False)
    passage_input_var =  C.ops.input_variable((1,p_max_words,emb_dim),np.float32,is_sparse=False)
    output_var = C.input_variable(num_classes,np.float32,is_sparse = False)
    train_reader = create_reader(trainfile, True, query_total_dim, passage_total_dim, label_total_dim)
    input_map = { query_input_var : train_reader.streams.queryfeatures, passage_input_var:train_reader.streams.passagefeatures, output_var : train_reader.streams.labels}

    # ********* Model configuration *******
    model_output = cnn_network(query_input_var, passage_input_var, num_classes)
    loss = C.binary_cross_entropy(model_output, output_var)
    pe = C.classification_error(model_output, output_var)
    lr_per_minibatch = C.learning_rate_schedule(0.03, C.UnitType.minibatch)
    learner = C.adagrad(model_output.parameters, lr=lr_per_minibatch)
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=total_epochs)

    #************Create Trainer with model_output object, learner and loss parameters*************
    trainer = C.Trainer(model_output, (loss, pe), learner, progress_printer)
    C.logging.log_number_of_parameters(model_output) ; print()

    # **** Train the model in batchwise mode *****
    for epoch in range(total_epochs):       # loop over epochs
        print("Epoch : ",epoch)
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = train_reader.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)        # training step
            sample_count += data[output_var].num_samples   # count samples processed so far

        trainer.summarize_training_progress()

        model_output.save("CNN_{}.dnn".format(epoch)) # Save the model for every epoch

        #*** Find metrics on validation set after every epoch ******#  (Note : you can skip doing this for every epoch instead to optimize the time, do it after every k epochs)
        predicted_labels=[]
        for i in range(len(validation_query_vectors)):
            queryVec   = np.array(validation_query_vectors[i],dtype="float32").reshape(1,q_max_words,emb_dim)
            passageVec = np.array(validation_passage_vectors[i],dtype="float32").reshape(1,p_max_words,emb_dim)
            scores = model_output(queryVec,passageVec)[0]   # do forward-prop on model to get score
            predictLabel = 1 if scores[1]>=scores[0] else 0
            predicted_labels.append(predictLabel)
        metrics = precision_recall_fscore_support(np.array(validation_labels), np.array(predicted_labels), average='binary')
        #print("precision : "+str(metrics[0])+" recall : "+str(metrics[1])+" f1 : "+str(metrics[2])+"\n")



    return model_output

## The following GetPredictionOnEvalSet method reads all query passage pair vectors from CTF file and does forward prop with trained model to get similarity score
## after getting scores for all the pairs, the output will be written into submission file.
def GetPredictionOnEvalSet(model,testfile,submissionfile):
    global q_max_words,p_max_words,emb_dim

    f = open(testfile,'r',encoding="utf-8")
    all_scores={} # Dictionary with key = query_id and value = array of scores for respective passages
    for line in f:
        tokens = line.strip().split("|")
        #tokens[0] will be empty token since the line is starting with |
        x1 = tokens[1].replace("qfeatures","").strip() #Query Features
        x2 = tokens[2].replace("pfeatures","").strip() # Passage Features
        query_id = tokens[3].replace("qid","").strip() # Query_id
        x1 = [float(v) for v in x1.split()]
        x2 = [float(v) for v in x2.split()]
        queryVec   = np.array(x1,dtype="float32").reshape(1,q_max_words,emb_dim)
        passageVec = np.array(x2,dtype="float32").reshape(1,p_max_words,emb_dim)
        score = model(queryVec,passageVec)[0][1] # do forward-prop on model to get score
        if(query_id in all_scores):
            all_scores[query_id].append(score)
        else:
            all_scores[query_id] = [score]
    fw = open(submissionfile,"w",encoding="utf-8")
    for query_id in all_scores:
        scores = all_scores[query_id]
        scores_str = [str(sc) for sc in scores] # convert all scores to string values
        scores_str = "\t".join(scores_str) # join all scores in list to make it one string with  tab delimiter.
        fw.write(query_id+"\t"+scores_str+"\n")
    fw.close()


if __name__ == "__main__":

    trainSetFileName = "TrainData.ctf"
    validationSetFileName = "ValidationData.ctf"
    testSetFileName = "EvaluationData.ctf"
    submissionFileName = "answer.tsv"

    LoadValidationSet(validationSetFileName)    #Load Validation Query, Passage Vectors from Validation CTF File
    model = TrainAndValidate(trainSetFileName) # Training and validation methods
    GetPredictionOnEvalSet(model,testSetFileName,submissionFileName) # Get Predictions on Evaluation Set
