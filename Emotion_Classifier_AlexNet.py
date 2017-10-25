import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import pandas as pd
import csv
import cv2
import numpy as np

base_path = "/home/ubuntu/EmotionDetection"

train_label_file = base_path + "/train.txt"
valid_label_file = base_path + "/val.txt"

train_label_dict = {}
valid_label_dict = {}

with open(train_label_file, mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        lines = str(rows[0]).split(' ')
        k = lines[0]
        v = lines[1]
        train_label_dict[k] = v

with open(valid_label_file, mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        lines = str(rows[0]).split(' ')
        k = str(lines[0])
        v = str(lines[1])
        valid_label_dict[k] = v
        
X_train = []
y_train = []
X_valid = []
y_val = []
n_train = len(train_label_dict)
n_val = len(valid_label_dict)

def load_image(im_dict, X, y):   
    for key in im_dict.keys():
        im_file = base_path + "/" + str(key)
        img = cv2.imread(im_file)
        X.append(img)
        y.append(im_dict[key])


load_image(train_label_dict, X_train, y_train)
load_image(valid_label_dict, X_valid, y_val)

X_train = np.array(X_train)
y_train = np.array(y_train)

print("X_train shape = " + str(X_train.shape))
print("y_train shape = " + str(y_train.shape))

X_valid = np.array(X_valid)
y_val = np.array(y_val)

print("X_valid shape = " + str(X_valid.shape))
print("y_val shape = " + str(y_val.shape))

# Load traffic signs data.
nb_classes = 7

# : Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 48, 48, 3))
resized = tf.image.resize_images(x, (227, 227))

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

fc8_W  = tf.Variable(tf.truncated_normal(shape=shape, mean = 0, stddev = 0.1))
fc8_b  = tf.Variable(tf.zeros(nb_classes))

logits = tf.matmul(fc7, fc8_W) + fc8_b

probs = tf.nn.softmax(logits)

# Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# Train and evaluate the feature extraction model.

y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, nb_classes)

rate = 0.0005
EPOCHS = 50
BATCH_SIZE = 128

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    validation_accuracy_list = []
    training_accuracy_list = []
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_val)
        train_accuracy = evaluate(X_train, y_train)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        validation_accuracy_list.append(validation_accuracy)
        training_accuracy_list.append(train_accuracy)
        print()
