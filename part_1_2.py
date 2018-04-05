import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import part_1_1 as p11
from time import gmtime, strftime


def tune_num_of_hidden_units():
    train_data, train_target, valid_data, valid_target, test_data, test_target = p11.load_data()
    valid_data = tf.cast(valid_data, tf.float32)
    valid_target_onehot = tf.one_hot(valid_target, 10)

    batch_size = 500
    num_iterations = 4500
    num_train = train_data.shape[0]
    num_batches = num_train // batch_size
    num_epochs = num_iterations // num_batches
    num_iterations_leftover = num_iterations % num_batches
    print("batch size:", batch_size, "; number of batches", num_batches)
    print("number of epochs:", num_epochs, "; iteration left-overs:", num_iterations_leftover)
    print("number of iterations:", num_iterations)

    hidden_units_num = [100, 500, 1000]
    for hidden_num in hidden_units_num:
        print("number of hidden units:", hidden_num)
        X0, Y, S2, loss, optimizer = p11.build_2_layer_NN(hidden_units_num=hidden_num)
        init = tf.global_variables_initializer()

        valid_error_list = []
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
                    _, train_loss, Output = sess.run([optimizer, loss, S2], feed_dict={
                        X0: batch_X0,
                        Y: batch_Y
                    })

                with tf.variable_scope("W", reuse=True):
                    W1 = tf.get_variable("W" + str(p11.layer_num - 2))
                    W2 = tf.get_variable("W" + str(p11.layer_num - 1))

                valid_error = p11.compute_2_layer_accuracy(valid_data, valid_target_onehot, W1, W2)

                train_loss_list.append(train_loss)
                valid_error_list.append(valid_error.eval())

                if epoch % 10 == 0:
                    print("training loss:", train_loss)
                    print("validation error:", valid_error.eval())

            print("final training loss:", train_loss_list[-1])
            print("final validation classification error:", valid_error_list[-1])

            with open("part_1_2_1.txt", "a") as file:
                file.write("\n" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n")
                file.write(str(hidden_num) + "final training loss: " + str(train_loss_list[-1]) + "\n")
                file.write(str(hidden_num) + "final validation classification error: " + str(valid_error_list[-1]) + "\n")

            plt.subplot(2, 1, 1)
            plt.plot(np.arange(num_epochs), train_loss_list)
            plt.xlabel("epoch #")
            plt.ylabel("entropy loss")
            plt.title("entropy loss vs epoch #")
            plt.subplot(2, 1, 2)
            plt.plot(np.arange(num_epochs), valid_error_list)
            plt.xlabel("epoch #")
            plt.ylabel("validation classification error")
            plt.title("validation classification error vs epoch #")
            # plt.tight_layout()
            plt.savefig("part_1_2_1_" + str(hidden_num), dpi=400)
            plt.gcf().clear()
            train_loss_list.clear()
            valid_error_list.clear()

            # compute test classification error
            # test_error = p11.compute_accuracy(test_data, tf.one_hot(test_target, 10), W1, W2)


def build_3_layer_NN(hidden_units_num = 500, reg =3e-4, learning_rate = 0.001):
    """
    :param hidden_units_num: # of neurons to be used in each hidden leayer
    :param reg: regularization strength
    :param learning_rate: learning rate
    :return: a 3 layer fully connected neural network
    """
    X0 = tf.placeholder(tf.float32, [None, 28 * 28])
    Y = tf.placeholder(tf.int32, [None])
    Y_onehot = tf.one_hot(Y, 10)

    # layer 1
    S1 = p11.build_layer(X0, neuron_num=hidden_units_num)
    X1 = tf.nn.relu(S1)

    # layer 2
    S2 = p11.build_layer(X1, neuron_num=hidden_units_num)
    X2 = tf.nn.relu(S2)

    # layer 3
    S3 = p11.build_layer(X2, neuron_num=10)
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=S3)

    with tf.variable_scope("W", reuse=True):
        W1 = tf.get_variable("W" + str(p11.layer_num - 3))
        W2 = tf.get_variable("W" + str(p11.layer_num - 2))
        W3 = tf.get_variable("W" + str(p11.layer_num - 1))
    loss = 0.5 * tf.reduce_mean(entropy) + reg * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return X0, Y, S3, loss, optimizer


def compute_3_layer_accuracy(data, one_hot_labels, W1, W2, W3):
    X1 = tf.nn.relu(tf.matmul(data, W1))
    X2 = tf.nn.relu(tf.matmul(X1, W2))
    preds = tf.nn.softmax(tf.matmul(X2, W3))
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(one_hot_labels, 1))
    error = 1 - tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    return error


def tune_num_of_layers():
    train_data, train_target, valid_data, valid_target, test_data, test_target = p11.load_data()
    valid_data = tf.cast(valid_data, tf.float32)
    valid_target_onehot = tf.one_hot(valid_target, 10)

    batch_size = 500
    num_iterations = 4500
    num_train = train_data.shape[0]
    num_batches = num_train // batch_size
    num_epochs = num_iterations // num_batches
    num_iterations_leftover = num_iterations % num_batches
    print("batch size:", batch_size, "; number of batches", num_batches)
    print("number of epochs:", num_epochs, "; iteration left-overs:", num_iterations_leftover)
    print("number of iterations:", num_iterations)

    X0, Y, S3, loss, optimizer = build_3_layer_NN()
    init = tf.global_variables_initializer()

    valid_error_list = []
    train_error_list = []

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
                _, _, Output = sess.run([optimizer, loss, S3], feed_dict={
                    X0: batch_X0,
                    Y: batch_Y
                })

            with tf.variable_scope("W", reuse=True):
                W1 = tf.get_variable("W" + str(p11.layer_num - 3))
                W2 = tf.get_variable("W" + str(p11.layer_num - 2))
                W3 = tf.get_variable("W" + str(p11.layer_num - 1))

            valid_error = compute_3_layer_accuracy(valid_data, valid_target_onehot, W1, W2, W3).eval()

            train_preds = tf.nn.softmax(Output)
            batch_Y_onehot = tf.one_hot(batch_Y, 10)
            correct_train_preds = tf.equal(tf.argmax(train_preds, 1), tf.argmax(batch_Y_onehot, 1))
            train_error = (1 - tf.reduce_mean(tf.cast(correct_train_preds, tf.float32))).eval()

            valid_error_list.append(valid_error)
            train_error_list.append(train_error)
            if epoch % 10 == 0:
                print("training error:", train_error)
                print("validation error:", valid_error)

        with open("part_1_2_2.txt", "a") as file:
            file.write("\n" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n")
            file.write("final training classification error: " + str(train_error_list[-1]) + "\n")
            file.write("final validation classification error: " + str(valid_error_list[-1]) + "\n")

        plt.plot(np.arange(num_epochs), train_error_list)
        plt.plot(np.arange(num_epochs), valid_error_list)
        plt.legend(['training', 'validation'])
        plt.title("classification error vs epoch #")
        plt.xlabel('epoch number')
        plt.ylabel('classification error')
        plt.savefig("part_1_2_2_valid_train", dpi=400)


def compare_with_2_layer():
    train_data, train_target, valid_data, valid_target, test_data, test_target = p11.load_data()
    test_data = tf.cast(test_data, tf.float32)
    test_target_onehot = tf.one_hot(test_target, 10)

    batch_size = 500
    num_iterations = 4500
    num_train = train_data.shape[0]
    num_batches = num_train // batch_size
    num_epochs = num_iterations // num_batches
    num_iterations_leftover = num_iterations % num_batches
    print("batch size:", batch_size, "; number of batches", num_batches)
    print("number of epochs:", num_epochs, "; iteration left-overs:", num_iterations_leftover)
    print("number of iterations:", num_iterations)

    X0_3, Y_3, S3, loss_3, optimizer_3 = build_3_layer_NN()
    X0_2, Y_2, S2, loss_2, optimizer_2 = p11.build_2_layer_NN()
    init = tf.global_variables_initializer()

    test_error_list_3 = []
    test_error_list_2 = []

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
                sess.run([optimizer_3, optimizer_2], feed_dict={
                    X0_3: batch_X0,
                    Y_3: batch_Y,
                    X0_2: batch_X0,
                    Y_2: batch_Y
                })

            with tf.variable_scope("W", reuse=True):
                W1_3 = tf.get_variable("W" + str(p11.layer_num - 5))
                W2_3 = tf.get_variable("W" + str(p11.layer_num - 4))
                W3_3 = tf.get_variable("W" + str(p11.layer_num - 3))
                W1_2 = tf.get_variable("W" + str(p11.layer_num - 2))
                W2_2 = tf.get_variable("W" + str(p11.layer_num - 1))
            test_error_3 = compute_3_layer_accuracy(test_data, test_target_onehot, W1_3, W2_3, W3_3).eval()
            test_error_2 = p11.compute_2_layer_accuracy(test_data, test_target_onehot, W1_2, W2_2).eval()

            if epoch % 10 == 0:
                print("3 layer test error:", test_error_3)
                print("2 layer test error:", test_error_2)

            test_error_list_3.append(test_error_3)
            test_error_list_2.append(test_error_2)

        with open("part_1_2_2.txt", "a") as file:
            file.write("\n" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n")
            file.write("3 layer final test classification error: " + str(test_error_list_3[-1]) + "\n")
            file.write("2 layer final test classification error: " + str(test_error_list_2[-1]) + "\n")

        plt.plot(np.arange(num_epochs), test_error_list_3)
        plt.plot(np.arange(num_epochs), test_error_list_2)
        plt.legend(['3-layer NN', '2-layer NN'])
        plt.title("test classification error vs epoch #")
        plt.xlabel('epoch number')
        plt.ylabel('test classification error')
        plt.savefig("part_1_2_2_test", dpi=400)








if __name__ == "__main__":
    # tune_num_of_hidden_units()
    # tune_num_of_layers()
    compare_with_2_layer()
