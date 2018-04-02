import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

layer_num = 1
W1 = None
W2 = None
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
    W = tf.get_variable(name="W" + str(layer_num), shape=[X.shape[1], neuron_num], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros([1, neuron_num]), name="bias")
    Sum = tf.matmul(X, W) + b
    if layer_num == 1:
        global W1
        W1 = W
    elif layer_num == 2:
        global W2
        W2 = W
    layer_num += 1
    return Sum


def build_graph(reg = 0.1, learning_rate = 0.005):
    """
    :param reg: regularization strength
    :param learning_rate: learning rate
    :return: a 2 layer fully connected neural network
    """
    X0 = tf.placeholder(tf.float32, [None, 28 * 28])
    Y = tf.placeholder(tf.int32, [None])
    Y_onehot = tf.one_hot(Y, 10)

    # layer 1
    S1 = build_layer(X0)
    X1 = tf.nn.relu(S1)

    # layer 2
    S2 = build_layer(X1, neuron_num=10)
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=S2)
    loss = 0.5 * tf.reduce_mean(entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return X0, Y, S2, loss, optimizer


def tune_learning_rate():
    train_data, train_target, valid_data, valid_target, test_data, test_target = load_data()

    batch_size = 500
    num_iterations = 900
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
        X0, Y, S2, loss, optimizer = build_graph(learning_rate=lr)
        init = tf.global_variables_initializer()
        train_loss_list = []

        with tf.Session() as sess:
            sess.run(init)
            shuffled_inds = np.arange(num_train)

            for epoch in range(num_epochs):

                np.random.shuffle(shuffled_inds)
                temp_train_data = train_data[shuffled_inds]
                temp_train_targets = train_target[shuffled_inds]

                for j in range(num_batches):
                    batch_X0 = temp_train_data[j * batch_size: (j + 1) * batch_size]
                    batch_Y = temp_train_targets[j * batch_size: (j + 1) * batch_size]
                    _, train_error, Output = sess.run([optimizer, loss, S2], feed_dict={
                        X0: batch_X0,
                        Y: batch_Y
                    })
                print(train_error)
                train_loss_list.append(train_error)

        plt.plot(np.arange(num_epochs), train_loss_list)
    plt.legend(['0.005', '0.001', '0.01'])
    plt.title("SGD training - error vs epoch #")
    plt.xlabel('epoch number')
    plt.ylabel('mean entropy')
    plt.show()


def train_no_early_stopping():
    train_data, train_target, valid_data, valid_target, test_data, test_target = load_data()
    valid_target_onehot = tf.one_hot(valid_target, 10)
    test_target_onehot = tf.one_hot(test_target, 10)
    valid_data = tf.cast(valid_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    batch_size = 500
    num_iterations = 900
    num_train = train_data.shape[0]
    num_batches = num_train // batch_size
    num_epochs = num_iterations // num_batches
    num_iterations_leftover = num_iterations % num_batches
    print("batch size:", batch_size, "; number of batches", num_batches)
    print("number of epochs:", num_epochs, "; iteration left-overs:", num_iterations_leftover)
    print("number of iterations:", num_iterations)

    X0, Y, S2, loss, optimizer = build_graph()
    init = tf.global_variables_initializer()
    train_loss_list = []
    train_error_list = []
    valid_error_list = []
    test_error_list = []

    with tf.Session() as sess:
        sess.run(init)
        shuffled_inds = np.arange(num_train)

        for epoch in range(num_epochs):

            np.random.shuffle(shuffled_inds)
            temp_train_data = train_data[shuffled_inds]
            temp_train_targets = train_target[shuffled_inds]

            for j in range(num_batches):
                batch_X0 = temp_train_data[j * batch_size: (j + 1) * batch_size]
                batch_Y = temp_train_targets[j * batch_size: (j + 1) * batch_size]
                _, train_loss, Output = sess.run([optimizer, loss, S2], feed_dict={
                    X0: batch_X0,
                    Y: batch_Y
                })
            # print("training loss:", train_loss)
            train_loss_list.append(train_loss)

            # W1 = tf.get_default_graph().get_tensor_by_name("W1")
            # W2 = tf.get_default_graph().get_tensor_by_name("W2")
            # W1 = tf.global_variables()["W1"]
            # W2 = tf.global_variables()["W2"]
            # tf.global_variables()
            # W1 = tf.Graph.get_tensor_by_name(name="W1")
            # W2 = tf.Graph.get_tensor_by_name(name="W2")

            # compute classficiation errors
            train_preds = tf.nn.softmax(Output)
            batch_Y_onehot = tf.one_hot(batch_Y, 10)
            correct_train_preds = tf.equal(tf.argmax(train_preds, 1), tf.argmax(batch_Y_onehot, 1))
            train_error = 1 - tf.reduce_mean(tf.cast(correct_train_preds, tf.float32))

            X1 = tf.nn.relu(tf.matmul(valid_data, W1))
            valid_preds = tf.nn.softmax(tf.matmul(X1, W2))
            correct_valid_preds = tf.equal(tf.argmax(valid_preds, 1), tf.argmax(valid_target_onehot, 1))
            valid_error = 1 - tf.reduce_mean(tf.cast(correct_valid_preds, tf.float32))

            X1 = tf.nn.relu(tf.matmul(test_data, W1))
            test_preds = tf.nn.softmax(tf.matmul(X1, W2))
            correct_test_preds = tf.equal(tf.argmax(test_preds, 1), tf.argmax(test_target_onehot, 1))
            test_error = 1 - tf.reduce_mean(tf.cast(correct_test_preds, tf.float32))

            train_error_list.append(train_error.eval())
            valid_error_list.append(valid_error.eval())
            test_error_list.append(test_error.eval())
            # print("training classficiation error:", train_error.eval())
            # print("valid classficiation error:", valid_error.eval())
            # print("test classficiation error:", test_error.eval())
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(num_epochs), train_loss_list)
        plt.xlabel("epoch #")
        plt.ylabel("entropy loss")
        plt.title("entropy loss vs epoch #")
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(num_epochs), train_error_list)
        plt.plot(np.arange(num_epochs), valid_error_list)
        plt.plot(np.arange(num_epochs), test_error_list)
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