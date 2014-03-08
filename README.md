Star-Quasar-Naive-Bayes
========================

Star - quasar classification using naive bayes method

I wrote a naive bayes classifier to classify stars and quasars using photometric color measurements provided by Sloan Digital Sky Suvey.

The repository contains two python source files and a data folder in which we have the training data for our method.

classifier.py - checks the accuracy of naive bayes classifier to scikit-learn's naive bayes classifier.

naive_bayes.py - contains the naive bayes class to classify the testing data.


Requirements
========================

The program is written in Python and also makes use of numpy and scikit-learn modules.

To install numpy:

    sudo apt-get install python-numpy
    
To install scikit-learn instructions can be found here http://scikit-learn.org/stable/install.html


Using the Naive Bayes
========================
    import naive_bayes
    # Initialize the classifier
    classifier = naive_bayes()
    # Train the classifier
    classifier.train(xtrain, ytrain)
    #predict the labels of the testing data
    ypredict = classifier.predict_label(xtest)


Output of our program
=========================

The classifier.py program outputs the accuracy of our naive bayes classifier and in the next line the accuracy of scikit-learn's naive bayes classifier.


Remarks
=========================

It works accurately but performance needs to be improved. Any suggestions to do so are extremely welcome :)
