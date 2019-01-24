
Pre-requisites  : 
	1) Install Anaconda from here : https://www.anaconda.com/download/ 

Instructions to Run baseline code :

Step 1 : Download Training  data ("Data.tsv") from codalab and keep it in current directory 

Step 2 : Download Evaluation data  ("eval1_unlabelled.tsv") from codalab  keep it in current directory

Step 3 : run "BaselineBM25.py" file for running BM25 technique
			python BaselineBM25.py

Step 4 : you should see "answer.tsv" file generated with query-passage similarity scores for Evaluation data. The format of the file will be queryid followed by 10 similarity scores.

Step 5 : Compress(zip) the submission file(answer.tsv) and upload in codalab
