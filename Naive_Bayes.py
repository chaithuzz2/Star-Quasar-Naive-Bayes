#------------------------------------------------------------------------------------------------------------------------------------------------------------
#Author: Krishna Chaitanya chavati
#Email: chaithukrishnazz2@gmail.com
#Date: 7th March'14
#Title: Naive_bayes classifier
#execution time	: 20 seconds for 500000 training samples and 200000 testing samples with 4 attributes and 2 labels
#-------------------------------------------------------------------------------------------------------------------------------------------------------------

import sys
import numpy as np
import math

class Naive_Bayes(object):

	def __init__(self):
		self.Accumulator = None
		self.Predictor = None
		self.number_of_attributes = None
		self.number_of_samples = None
		self.number_of_labels = None
		self.unique_labels = None
		self.intermediate_array = None
		self.normalization_constant = None

	"""	Method to fit the training data in to the classifier. Builds a model by accumulating the mean and standard deviation
		for each attribute for each label and stores it for the purpose of predicting """
	def train(self, xtrain, ytrain):
		self.number_of_attributes = len(xtrain[0])
		self.number_of_samples =  len(xtrain)
		self.unique_labels = list(set(ytrain))
		self.number_of_labels = len(self.unique_labels)
		self.Accumulator = np.zeros((self.number_of_attributes, self.number_of_labels, 2), float)
		for i in range(0, self.number_of_attributes):
			for j in range(0, self.number_of_labels):
				temp_array = [xtrain[n][i] for n in range(0, self.number_of_samples) if ytrain[n] == self.unique_labels[j]]
				temporary_mean = np.mean(temp_array)
				self.Accumulator[i][j][0] = temporary_mean
				self.Accumulator[i][j][1] = np.std(temp_array)
				
	""" This method predicts the labels of the testing data by obtaining the probability distribution functions of each attribute
	for each label and stores it an three dimensional array. Later it gets an estimate of the probablities and predicts the
	label for which the value is the highest	"""	
	
	def predict_label(self, xtest):
		number_of_testing_samples = len(xtest)
		ytest = np.zeros((number_of_testing_samples))
		PDF = np.zeros((number_of_testing_samples, self.number_of_attributes, self.number_of_labels), float)
		self.intermediate_array = np.zeros((number_of_testing_samples, self.number_of_labels), float)
		for k in range(0, number_of_testing_samples):		
			for i in range(0, self.number_of_attributes):
				for j in range(0, self.number_of_labels):
					PDF[k][i][j] = self.calculatePDF(xtest[k][i], self.Accumulator[i][j][0], self.Accumulator[i][j][1])
		for l in range(0, number_of_testing_samples):
			for m in range(0, self.number_of_labels):
				self.intermediate_array[l][m] = self.product([PDF[l][s][m] for s in range(0, self.number_of_attributes)])
		for n in range(0, number_of_testing_samples):
			maximum_index = 0
			maximum_value = self.intermediate_array[n][0]	
			for p in range(0, self.number_of_labels):
				if(self.intermediate_array[n][p] > maximum_value):
					maximum_value = self.intermediate_array[n][p]
					maximum_index = p
			ytest[n] = int(self.unique_labels[maximum_index])
		return ytest

	""" calculates the probability distribution function for a given distribution at x with Mean and Standard_Deviation known
	It is necessary to estimate the probabilities of a sample belonging to a label 	""" 
	
	def calculatePDF(self, x, Mean, Standard_Deviation):
		Variance = float(Standard_Deviation)**2 
		Pi = 3.14159
		Denominator = (2*Pi*Variance)**.5
		Numerator = 2.71828**(-(float(x)-float(Mean))**2/(2*Variance))
		return Numerator/Denominator

	""" calculates the mean given an array	"""
	
	def mean(self, array):
		length = len(array)
		mean_sum = 0
		for i in range(0, length):
			mean_sum+=array[i]		
		return mean_sum/length
	
	""" calculates the standardDeviation given an array and mean . I later moved on to numpy's std method for performance  """

	def standardDeviation(self, array, temp_mean):
		length = len(array)
		square_array = []
		for i in range(0, length):
			square_array.append(array[i]**2)  
		return ((self.mean(square_array) - ((temp_mean)**2)**0.5))

	""" calculates the product of all the elements in the array """
	def product(self, array):
		result = 1.0
		for i in range(0, len(array)):
			result = result * float(array[i])
		return result
