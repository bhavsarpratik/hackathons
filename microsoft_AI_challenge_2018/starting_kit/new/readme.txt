
Pre-requisites  : 
	1) Install Anaconda from here : https://www.anaconda.com/download/ 
		 
	2) Install CNTK from here:  https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine 
		a) pip install cntk (recommended)
			OR
		b) any other way mentioned in the above page
					       
	3) Install scikit learn package :
		a) pip install -U scikit-learn   or b) conda install scikit-learn

Instructions to Run baseline code.

Step 1 : Download "Data.tsv" from codalab 

Step 2 : Split the data into two parts as train data and validation data. you can split with any ratio that you feel good. (90%-10% is recommended). Name the files "traindata.tsv", "validationdata.tsv" respectively

Step 3 : Download Evaluation data  ("eval1_unlabelled.tsv") from codalab 

Step 4 : a) Download Glove Embeddings from here http://nlp.stanford.edu/data/glove.6B.zip.
		 b) Extract zip file and copy "glove.6B.50d.txt" file to the current directory

Step 5 : run "text2ctf.py" file. Make sure you have "traindata.tsv", "validationdata.tsv" and "eval1_unlabelled.tsv" in the current directory
				python text2ctf.py 
		 you should see three files "TrainData.ctf","ValidationData.ctf","EvaluationData.ctf" generated in the current directory

Step 6 : run "PassageRanking.py" file for running CNN on train data (Make sure you set hyper-parameters such as : epoch_size,minibatch_size etc. as per the size of your traindata.tsv)
			python PassageRanking.py
			Note: This training of deep learnt model might take a lot of time. You can try reducing the size of training data or number of epochs. 
Step 7 : you should see "answer.tsv" file generated with query-passage similarity scores for Evaluation data. The format of the file will be queryid followed by 10 scores.

Step 8 : Compress(zip) the submission file(answer.tsv) and upload in codalab
