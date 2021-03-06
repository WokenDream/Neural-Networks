import math
import random
import numpy as np
import tensorflow as tf

layer = 1
def layer_constructor(X, hidden_units):

    x_dim = X.shape[1].value
    #784 x hidden_units
    global layer
    with tf.variable_scope("W"):
        W = tf.get_variable(name="W" + str(layer), shape=[x_dim, hidden_units],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    b = tf.Variable(tf.zeros(hidden_units), name='bias');
    layer = layer + 1
    return tf.add(tf.matmul(X, W), b)

def neural_network(pixel_shape, num_hidden_units, num_classifications, weight_decay, learning_rate, n_iter, num_batch, batch_size,
                   trainData, trainTarget, validData, validTarget, testData, testTarget):
    X0 = tf.placeholder(tf.float32, shape=[None, pixel_shape])
    X0_valid = tf.placeholder(tf.float32, shape=[None, pixel_shape])
    X0_test = tf.placeholder(tf.float32, shape=[None, pixel_shape])

    Y = tf.placeholder(tf.int32, shape=[None])
    Y_valid = tf.placeholder(tf.int32, shape=[None])
    Y_test = tf.placeholder(tf.int32, shape=[None])

    Y_true = tf.one_hot(Y, 10)
    Y_true_valid = tf.one_hot(Y_valid, 10)
    Y_true_test = tf.one_hot(Y_test, 10)

    #training
    S1 = (layer_constructor(X0, num_hidden_units))
    X1 = tf.nn.relu(S1)
    S2 = (layer_constructor(X1, num_hidden_units))
    X2 = tf.nn.relu(S2)
    S3 = (layer_constructor(X2, num_classifications))

    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_true, logits=S3)

    with tf.variable_scope("W", reuse=True):
        W1 = tf.get_variable("W1")
        W2 = tf.get_variable("W2")
        W3 = tf.get_variable("W3")

    #validation
    X1_valid = tf.nn.relu(tf.matmul(X0_valid, W1))
    X2_valid = tf.nn.relu(tf.matmul(X1_valid, W2))
    valid_pred = tf.nn.softmax(tf.matmul(X2_valid, W3))
    accuracy_valid = 1 - (tf.count_nonzero(tf.equal((tf.argmax(valid_pred, 1)), tf.argmax(Y_true_valid, 1))) / 1000)

    #test
    X1_test = tf.nn.relu(tf.matmul(X0_test, W1))
    X2_test = tf.nn.relu(tf.matmul(X1_test, W2))
    test_pred = tf.nn.softmax(tf.matmul(X2_test, W3))
    accuracy_test = 1 - (tf.count_nonzero(tf.equal((tf.argmax(test_pred, 1)), tf.argmax(Y_true_test, 1))) / 2724)

    #cross enropy
    loss = 0.5 * tf.reduce_mean(entropy) + weight_decay * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))

    #accuracy
    accuracy = 1 - (tf.count_nonzero(tf.equal((tf.argmax(S3, 1)), tf.argmax(Y_true, 1))) / batch_size)

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)

        shuffled_ind = np.arange(trainData.shape[0])
        validData = np.reshape(validData, [1000, 784])
        testData = np.reshape(testData, [2724, 784])
        train_error = []
        train_accuracy = []
        valid_accuracy = []
        test_accuracy = []
        for j in range(n_iter // num_batch):
            np.random.shuffle(shuffled_ind)
            temp_trainData = trainData[shuffled_ind]
            temp_trainTarget = trainTarget[shuffled_ind]
            for i in range(num_batch):
                x_value = temp_trainData[i * batch_size: (i + 1) * batch_size].reshape(batch_size, -1)
                y_value = temp_trainTarget[i * batch_size: (i + 1) * batch_size]

                _, error_value, train_acc, valid_acc, test_acc = session.run([train_op, loss, accuracy, accuracy_valid, accuracy_test],
                                                                   feed_dict={X0: x_value, Y: y_value,
                                                                              X0_valid: validData, Y_valid: validTarget,
                                                                              X0_test: testData, Y_test: testTarget})
                #print(error_value)
                train_error.append(error_value)
                train_accuracy.append(train_acc)
                valid_accuracy.append(valid_acc)
                test_accuracy.append(test_acc)

    return train_error, train_accuracy, valid_accuracy, test_accuracy
def main():
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

    batch_size = 500
    num_batch = 30
    pixel_shape = 784
    num_classifications = 10
    n_iter = 3000

    learning_rate = random.uniform(math.exp(-7.5), math.exp(-4.5))
    num_hidden_units = random.randint(100,500)
    weight_decay = random.uniform(math.exp(-9), math.exp(-6))
    num_layers = random.randint(1,5)
    dropout = random.randint(0,1)

    print("learning rate: ", learning_rate)
    print("num hidden units: ", num_hidden_units)
    print("weight decay: ", weight_decay)
    print("num_layers: ", num_layers)
    print("dropout: ", dropout)

    train_error, train_accuracy, valid_accuracy, test_accuracy = neural_network(pixel_shape, num_hidden_units, num_classifications, weight_decay, learning_rate, n_iter, num_batch, batch_size,
               trainData, trainTarget, validData, validTarget, testData, testTarget)

    print("Final Validation error: ", valid_accuracy[-1])
    print("Final test error:", test_accuracy[-1])



if __name__ == "__main__":
    main()