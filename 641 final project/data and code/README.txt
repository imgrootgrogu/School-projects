1. Datasets
1.1 training data
(1) Use shared_task_posts.csv and crowd_train.csv as training set. Merge those two sets on user_id and continue with the data processing.
(2) We saved this dataset as data_for_prediction.csv to efficiently work on downstream tasks for each team member. Although in finalproj_fns.py file, saving dataset code is commented out because it is not necessary in the complete code setting.

1.2 Test dataset
Shared_task_posts_text.csv and crowd_test.csv are used for held out testing dataset. We did same process for test dataset and saved it as data_for_test.csv for final testing task.

1.3 Prediction dataset
logistic_regression_predictions.csv and svm_predictions.csv are the output from empath_logistic_regression.py and empath_optimized.py. Each of us ran different models during the training process, then we collected the output in order to do final aggregation based on user_id.

2. Code file
2.1 Main code
finalproj_fns.py is our main code block, which contains data processing, tokenizing, vectorizing, training, testing and evaluation on both post level and user level. 
2.2 SVM model
We used empath_randomsearch_SVM.py to get the best parameter combination for SVM model then use the optimized parameter in training a simpler SVM model to get output since it takes shorter time and less computational power to run. The code is stored in empath_optimized_SVM.py. The output is then used in main code block to perform aggregation and evaluation.
2.3 Logistic regression
Another model used in our project is logistic regression, it is stored in empath_logistic_regression.py file. The output is then used in main code block to perform aggregation and evaluation.
2.4 Exploratory models
The notebook file exploratory_models.ipynb includes different models that were used for experimenting purpose such as BERT and LSTM etc.