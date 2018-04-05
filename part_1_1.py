import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime

layer_num = 1
def build_layer(X, neuron_num = 1000):
    """
    part 1.1.1
    :param X: output from previous layer; shape = (# of samples, length of data vector)
    :param neuron_num: number of neuron in this layer
    :return: output of size (# of samples, # of neurons in this layer);
             output[i][j] = weighted sum produced by j'th neuron over of i'th sample data
             for output layer: i'th row is i'th image's probabilities to be each of 10 classes
    """
    global layer_num
    with tf.variable_scope("W"):
        W = tf.get_variable(name="W" + str(layer_num), shape=[X.shape[1], neuron_num], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros([1, neuron_num]), name="bias")
    Sum = tf.matmul(X, W) + b
    layer_num += 1 # give each W a unique name
    return Sum


def compute_2_layer_error(data, one_hot_labels, W1, W2):
    X1 = tf.nn.relu(tf.matmul(data, W1))
    preds = tf.nn.softmax(tf.matmul(X1, W2))
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(one_hot_labels, 1))
    error = 1 - tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    return error


def build_2_layer_NN(hidden_units_num = 1000, reg =3e-4, learning_rate = 0.001, dropout = False):
    """
    :param hidden_units_num: number of neurons in each hidden layer
    :param reg: regularization strength
    :param learning_rate: learning rate
    :return: a 2 layer fully connected neural network
    """
    X0 = tf.placeholder(tf.float32, [None, 28 * 28])
    Y = tf.placeholder(tf.int32, [None])
    Y_onehot = tf.one_hot(Y, 10)

    # layer 1
    S1 = build_layer(X0, neuron_num=hidden_units_num)
    X1 = tf.nn.relu(S1)
    if dropout:
        print("applying dropout")
        X1 = tf.nn.dropout(X1, 0.5)

    # layer 2
    S2 = build_layer(X1, neuron_num=10)
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=S2)

    with tf.variable_scope("W", reuse=True):
        W1 = tf.get_variable("W" + str(layer_num - 2))
        W2 = tf.get_variable("W" + str(layer_num - 1))
    loss = 0.5 * tf.reduce_mean(entropy) + reg * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return X0, Y, S2, loss, optimizer


def tune_learning_rate():
    train_data, train_target, valid_data, valid_target, test_data, test_target = load_data()

    batch_size = 500
    num_iterations = 1800
    num_train = train_data.shape[0]
    num_batches = num_train // batch_size
    num_epochs = num_iterations // num_batches
    num_iterations_leftover = num_iterations % num_batches
    print("batch size:", batch_size, "; number of batches", num_batches)
    print("number of epochs:", num_epochs, "; iteration left-overs:", num_iterations_leftover)
    print("number of iterations:", num_iterations)
    learning_rates = [0.005, 0.001, 0.0001]

    for lr in learning_rates:
        print("learning rate:", lr)
        X0, Y, S2, loss, optimizer = build_2_layer_NN(learning_rate=lr)
        init = tf.global_variables_initializer()
        train_loss_list = []

        with tf.Session() as sess:
            sess.run(init)

            # sample epoch 0
            train_loss = sess.run(loss, feed_dict={
                X0: train_data[0:batch_size],
                Y: train_target[0:batch_size]
            })
            print("initial training loss:", train_loss)
            train_loss_list.append(train_loss)

            shuffled_inds = np.arange(num_train)

            for epoch in range(num_epochs):

                np.random.shuffle(shuffled_inds)
                temp_train_data = train_data[shuffled_inds]
                temp_train_targets = train_target[shuffled_inds]

                for j in range(num_batches):
                    batch_X0 = temp_train_data[j * batch_size: (j + 1) * batch_size]
                    batch_Y = temp_train_targets[j * batch_size: (j + 1) * batch_size]
                    _, train_loss = sess.run([optimizer, loss], feed_dict={
                        X0: batch_X0,
                        Y: batch_Y
                    })
                train_loss_list.append(train_loss)
                if epoch % 10 == 0:
                    print("training loss:", train_loss)

        plt.plot(train_loss_list)
    plt.legend(['0.005', '0.001', '0.01'])
    plt.title("SGD training - error vs epoch #")
    plt.xlabel('epoch number')
    plt.ylabel('mean entropy')
    plt.show()


def compute_errors(sess, X0, Y, S2, loss, batch_X0, batch_Y, valid_data, valid_target_onehot, test_data, test_target_onehot):
    train_loss, Output = sess.run([loss, S2], feed_dict={
        X0: batch_X0,
        Y: batch_Y
    })

    with tf.variable_scope("W", reuse=True):
        W1 = tf.get_variable("W1")
        W2 = tf.get_variable("W2")

    # compute classficiation errors
    train_preds = tf.nn.softmax(Output)
    batch_Y_onehot = tf.one_hot(batch_Y, 10)
    correct_train_preds = tf.equal(tf.argmax(train_preds, 1), tf.argmax(batch_Y_onehot, 1))
    train_error = 1 - tf.reduce_mean(tf.cast(correct_train_preds, tf.float32))

    valid_error = compute_2_layer_error(valid_data, valid_target_onehot, W1, W2)
    test_error = compute_2_layer_error(test_data, test_target_onehot, W1, W2)

    return train_loss, train_error.eval(), valid_error.eval(), test_error.eval()



def train_no_early_stopping():
    train_data, train_target, valid_data, valid_target, test_data, test_target = load_data()
    valid_target_onehot = tf.one_hot(valid_target, 10)
    test_target_onehot = tf.one_hot(test_target, 10)
    valid_data = tf.cast(valid_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    batch_size = 500
    num_iterations = 9000
    num_train = train_data.shape[0]
    num_batches = num_train // batch_size
    num_epochs = num_iterations // num_batches
    num_iterations_leftover = num_iterations % num_batches
    print("batch size:", batch_size, "; number of batches", num_batches)
    print("number of epochs:", num_epochs, "; iteration left-overs:", num_iterations_leftover)
    print("number of iterations:", num_iterations)

    X0, Y, S2, loss, optimizer = build_2_layer_NN()
    init = tf.global_variables_initializer()
    train_loss_list = []
    train_error_list = []
    valid_error_list = []
    test_error_list = []

    with tf.Session() as sess:
        sess.run(init)

        # sample epoch 0
        batch_X0 = train_data[0:batch_size]
        batch_Y = train_target[0:batch_size]
        train_loss, train_error, valid_error, test_error = compute_errors(sess, X0, Y, S2, loss, batch_X0, batch_Y,
                                                                          valid_data, valid_target_onehot,
                                                                          test_data, test_target_onehot)
        print("initial loss:", train_loss)
        train_loss_list.append(train_loss)
        train_error_list.append(train_error)
        valid_error_list.append(valid_error)
        test_error_list.append(test_error)

        shuffled_inds = np.arange(num_train)

        prev_valid_error = 100000000
        prev_test_error = 100000000
        valid_inc = 0
        test_inc = 0
        for epoch in range(num_epochs):

            np.random.shuffle(shuffled_inds)
            temp_train_data = train_data[shuffled_inds]
            temp_train_targets = train_target[shuffled_inds]

            for j in range(num_batches):
                batch_X0 = temp_train_data[j * batch_size: (j + 1) * batch_size]
                batch_Y = temp_train_targets[j * batch_size: (j + 1) * batch_size]
                sess.run(optimizer, feed_dict={
                    X0: batch_X0,
                    Y: batch_Y
                })

            train_loss, train_error, valid_error, test_error = compute_errors(sess, X0, Y, S2, loss, batch_X0, batch_Y,
                                                                              valid_data, valid_target_onehot,
                                                                              test_data, test_target_onehot)
            train_loss_list.append(train_loss)
            train_error_list.append(train_error)
            valid_error_list.append(valid_error)
            test_error_list.append(test_error)

            # check for early stopping points
            if valid_inc > -1:
                if valid_error >= prev_valid_error:
                    valid_inc += 1
                else:
                    valid_inc = 0
                if valid_inc == 5:
                    valid_inc = -1 if epoch > 100 else 0
                    with open("part_1_1_3.txt", "a") as file:
                        file.write("\n" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n")
                        file.write("valid early stop pt at epoch: " + str(epoch))

            if test_inc > -1:
                if test_error >= prev_test_error:
                    test_inc += 1
                else:
                    test_inc = 0
                if test_inc == 5:
                    test_inc = -1 if epoch > 100 else 0
                    with open("part_1_1_3.txt", "a") as file:
                        file.write("\n" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n")
                        file.write("test early stop pt at epoch: " + str(epoch))

            prev_valid_error = valid_error
            prev_test_error = test_error

            if epoch % 10 == 0:
                print("training loss:", train_loss)
                print("valid classficiation error:", valid_error)

        print("final training loss:", train_loss_list[-1])
        print("final training classification error:", train_error_list[-1])
        print("final validation classification error:", valid_error_list[-1])
        print("final test classification error:", test_error_list[-1])
        with open("part_1_1_2.txt", "a") as file:
            file.write("\n" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n")
            file.write("final training loss: " +str(train_loss_list[-1]) + "\n")
            file.write("final training classification error: " + str(train_error_list[-1]) + "\n")
            file.write("final validation classification error: " + str(valid_error_list[-1]) + "\n")
            file.write("final test classification error:" + str(test_error_list[-1]) + "\n")
        plt.subplot(2, 1, 1)
        plt.plot(train_loss_list)
        plt.xlabel("epoch #")
        plt.ylabel("entropy loss")
        plt.title("entropy loss vs epoch #")
        plt.subplot(2, 1, 2)
        plt.plot(train_error_list)
        plt.plot(valid_error_list)
        plt.plot(test_error_list)
        plt.xlabel("epoch #")
        plt.ylabel("classification error")
        plt.title("classification error vs epoch #")
        plt.legend(["training", "validation", "test"])
        plt.tight_layout()
        plt.savefig("part_1_1_2", dpi=400)
        plt.show()
        plt.gcf().clear()



def load_data():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]

    print("trainData:", trainData.shape, "trainTarget:", trainTarget.shape)
    print("validData:", validData.shape, "validTarget:", validTarget.shape)
    print("testData:", testData.shape, "testTarget:", testTarget.shape)

    train_data = np.reshape(trainData, [-1, 28 * 28])
    valid_data = np.reshape(validData, [-1, 28 * 28])
    test_data = np.reshape(testData, [-1, 28 * 28])

    return train_data, trainTarget, valid_data, validTarget, test_data, testTarget

if __name__ == "__main__":
    # tune_learning_rate()
    train_no_early_stopping()
