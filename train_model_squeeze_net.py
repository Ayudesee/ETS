from SqueezeNet import *
import numpy as np
import tensorflow as tf
import time
import random

WIDTH = 432
HEIGHT = 270
LR = 1e-3
EPOCHS = 3
CLASSES = 3
BATCH_SIZE = 32
MODEL_NAME = 'squeeze-net-1.model'.format(LR, 'squeezenet', EPOCHS)

train_data = np.load('D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-100.npy', allow_pickle=True)
tf.compat.v1.disable_v2_behavior()

c = len(train_data)
train = train_data[:-c // 10]
test = train_data[-c // 10:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]

print('Dataset size:', len(X))

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = [i[1] for i in test]

x = tf.compat.v1.placeholder(tf.float32, (None, WIDTH, HEIGHT, 1))
y = tf.compat.v1.placeholder(tf.float32, (None, CLASSES))
keep_prob = tf.compat.v1.placeholder(tf.float32)

_, train_step, accuracy = getSqueezeNetModel(x, y, keep_prob, LR)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    summaryWriter = tf.summary.FileWriter('./SqueezeNet/Tensorboard', sess.graph)
    sess.run(tf.global_variables_initializer())
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    saver = tf.train.Saver()

    print('Started')
    start_time = time.time()
    for i in range(EPOCHS):
        print(i)
        if i % 100 == 0:
            summary, train_accuracy = sess.run([merged, accuracy],
                                               feed_dict={x: test_x, y: test_y, keep_prob: 1},
                                               options=run_options,
                                               run_metadata=run_metadata)
            summaryWriter.add_run_metadata(run_metadata, 'step%03d' % i)
            summaryWriter.add_summary(summary, i)
            print("step %d, training accuracy %g %f" % (i, train_accuracy, time.time() - start_time))
            start_time = time.time()
            saver.save(sess, './SqueezeNet/' + MODEL_NAME)

        batch = np.array(random.sample(list(zip(X, Y)), BATCH_SIZE))
        batchX = list(batch[:, 0])
        batchY = list(batch[:, 1])
        train_step.run(feed_dict={x: batchX, y: batchY, keep_prob: .5})
