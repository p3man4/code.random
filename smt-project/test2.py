# coding: utf-8
## from https://jmetzen.github.io/2015-11-27/vae.html

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys

sys.path.append("/home/junwon/tensorflow_code/tensorflow/tensorflow/examples/tutorials/mnist/")
import input_data
np.random.seed(0)
tf.set_random_seed(0)
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
n_samples = mnist.train.num_examples
           
x_sample = mnist.test.next_batch(100)[0]
print x_sample[0].shape

