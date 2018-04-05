import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
import part_1_1 as p11


def compute_errors(sess, X0, S2, batch_X0, batch_Y, valid_data, valid_target_onehot):
    with tf.variable_scope("W", reuse=True):
        W1 = tf.get_variable("W1")
        W2 = tf.get_variable("W2")

    Output = sess.run(S2, feed_dict={
        X0: batch_X0
    })
    # compute training error
    train_preds = tf.nn.softmax(Output)
    batch_Y_onehot = tf.one_hot(batch_Y, 10)
    correct_train_preds = tf.equal(tf.argmax(train_preds, 1), tf.argmax(batch_Y_onehot, 1))
    train_error = 1 - tf.reduce_mean(tf.cast(correct_train_preds, tf.float32))

    # compute validation error
    valid_error = p11.compute_2_layer_error(valid_data, valid_target_onehot, W1, W2)

    return train_error.eval(), valid_error.eval()

def train_dropout():
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

    X0, Y, S2, loss, optimizer = p11.build_2_layer_NN(dropout=True)

    init = tf.global_variables_initializer()

    valid_error_list = []
    train_error_list = []

    with tf.Session() as sess:
        sess.run(init)
        shuffled_inds = np.arange(num_train)

        batch_X0 = train_data[0:batch_size]
        batch_Y = train_target[0:batch_size]
        train_error, valid_error = compute_errors(sess, X0, S2, batch_X0, batch_Y, valid_data, valid_target_onehot)
        train_error_list.append(train_error)
        valid_error_list.append(valid_error)
        print("initial train classification error:", train_error)
        print("initial valid classification error:", valid_error)

        best_valid_errors = []
        best_valid_error = 100000
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
                
            train_error, valid_error = compute_errors(sess, X0, S2, batch_X0, batch_Y, valid_data, valid_target_onehot)

            train_error_list.append(train_error)
            valid_error_list.append(valid_error)

            if valid_error < best_valid_error:
                best_valid_errors.clear()
                best_valid_errors.append((epoch, valid_error))
                best_valid_error = valid_error
            elif valid_error == best_valid_error:
                best_valid_errors.append((epoch, valid_error))

            if epoch % 10 == 0:
                print("training classification error:", train_error)
                print("valid classification error:", valid_error)

        print("final training classification error:", train_error_list[-1])
        print("final validation classification error:", valid_error_list[-1])

        with open("part_1_3_1.txt", "a") as file:
            file.write("\n" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n")
            file.write("final training classification error: " + str(train_error_list[-1]) + "\n")
            file.write("final validation classification error: " + str(valid_error_list[-1]) + "\n")
            file.write("best validation classification error: " + str(best_valid_errors) + "\n")

        plt.plot(train_error_list)
        plt.plot(valid_error_list)
        plt.xlabel("epoch #")
        plt.ylabel("classification error")
        plt.title("classification error vs epoch #")
        plt.legend(["training", "validation"])
        plt.savefig("part_1_3_1", dpi=400)
        plt.gcf().clear()



if __name__ == "__main__":
    train_dropout()