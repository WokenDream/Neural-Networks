import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import part_1_1 as p11


def tune_num_of_hidden_units():
    train_data, train_target, valid_data, valid_target, test_data, test_target = p11.load_data()
    valid_data = tf.cast(valid_data, tf.float32)
    valid_target_onehot = tf.one_hot(valid_target, 10)

    batch_size = 500
    num_iterations = 9000
    num_train = train_data.shape[0]
    num_batches = num_train // batch_size
    num_epochs = num_iterations // num_batches
    num_iterations_leftover = num_iterations % num_batches
    print("batch size:", batch_size, "; number of batches", num_batches)
    print("number of epochs:", num_epochs, "; iteration left-overs:", num_iterations_leftover)
    print("number of iterations:", num_iterations)
    hidden_units_num = [100, 500, 1000]
    # hidden_units_num = [500]
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
                    W1 = tf.get_variable("W1")
                    W2 = tf.get_variable("W2")

                X1 = tf.nn.relu(tf.matmul(valid_data, W1))
                valid_preds = tf.nn.softmax(tf.matmul(X1, W2))
                correct_valid_preds = tf.equal(tf.argmax(valid_preds, 1), tf.argmax(valid_target_onehot, 1))
                valid_error = 1 - tf.reduce_mean(tf.cast(correct_valid_preds, tf.float32))

                train_loss_list.append(train_loss)
                valid_error_list.append(valid_error.eval())

                if epoch % 10 == 0:
                    print("training loss:", train_loss)
                    print("validation error:", valid_error.eval())

            print("final training loss:", train_loss_list[-1])
            print("final validation classification error:", valid_error_list[-1])

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
            plt.tight_layout()
            plt.savefig("part_1_2_1_" + str(hidden_num), dpi=400)
            plt.gcf().clear()


if __name__ == "__main__":
    tune_num_of_hidden_units()
