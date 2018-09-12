# Planet: Understanding the Amazon from Space

In this project, we are challenged to perform a deep learning analysis with satellite image
classification, in order to label image chips with atmospheric conditions and various classes of
land cover/land use. Resulting algorithms will help the global community to better understand
where, how, and why deforestation happens all over the world - and ultimately how to respond.
The technical target of the project is to train a model that would recognize the two most
important types objects in satellite data in order to track the human footprint in the Amazon
rainforest.

Repository contains code to solve Kaggle problem Planet: Understanding the Amazon from
Space and a concise report.

For this purpose, we will use deep learning and neural network techniques through Tensorflow.

The dataset can be downloaded from here: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data

In order for the code to run, the following libraries are needed, which can be installed via pip:
- numpy
- matplotlib
- tensorflow
- keras

# Requirements:
Python 3.0, Keras 2.2.0, Tensorflow 1.5

# How to run:
Please execute the script cnn.py above using the training jpeg dataset from kaggle and the corresponding csv with the labels.

# Notes:
We reduced the number of different labels, 17 in total, to the 2 of the most frequent and
important ones for our purpose to analyze, agriculture and road.

Directory report contains a short report for our work on the project,
while presentation directory contains the file of our final presentation.

Authors: Eirini Aloukou (Statistician) – Orianna Lymperi (Environmental Engineer)

This project was conducted as part of the "Big Data Content Analytics" class of the M.Sc. in Business Analytics, Athens University of Economics & Business.
