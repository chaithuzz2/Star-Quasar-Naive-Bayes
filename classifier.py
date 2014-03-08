import numpy as np
from sklearn import naive_bayes
from sklearn import metrics
from Naive_Bayes import Naive_Bayes	

""" In this program we will try to know the accuracy of Naive_Bayes classifier we are using and later compare it 
	with scikit-learn's naive bayes classifier"""

""" The total training data has been split into two parts training_data which contains 500000 samples of photometric
	measurements and testing_data which contains 200000 samples. The reason we are doing this is to check the accuracy
	of our classifier with already classified data as testing data. Our task here is to classify the sources into
	stars and quasars . The data and the program were part of a tutorial by astroML on scikit-learn
	check this tutorial http://www.astroml.org/sklearn_tutorial/classification.html     """  


train_data = np.load('data/sdssdr6_colors_class_train.npy')
test_data = np.load('data/sdssdr6_colors_class.200000.npy')


"""	getting the data into the training and testing sets for use in classifier """
X_train = np.vstack([train_data['u-g'], train_data['g-r'], train_data['r-i'], train_data['i-z']]).T
y_train = (train_data['redshift'] > 0).astype(int)
X_test = np.vstack([test_data['u-g'], test_data['g-r'], test_data['r-i'], test_data['i-z']]).T
y_test = (test_data['label'] == 0).astype(int)


""" Apply naive bayes """

gnb = Naive_Bayes()
print "It starts . Hang on for around 20 seconds buddy :)" 
gnb.train(X_train, y_train)
print "training is over"
y_pred = gnb.predict_label(X_test)
print "predicting is over"

gnb1 = naive_bayes.GaussianNB()
gnb1.fit(X_train, y_train)
y_pred1 = gnb1.predict(X_test) 


""" checking the accuracy of our classifier and printing it	"""
accuracy_of_naivebayes_we_used = float(np.sum(y_test == y_pred)) / len(y_test)
accuracy_of_sklearns_naive_bayes = float(np.sum(y_test == y_pred1)) / len(y_test)

print "accuracy of our naive bayes classifier is : " + str(accuracy_of_naivebayes_we_used)
print "accuracy of scikit-learn's naive bayes classifier is : " + str(accuracy_of_sklearns_naive_bayes)
 









