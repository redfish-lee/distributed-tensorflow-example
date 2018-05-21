import argparse
import sys
import time
import tensorflow as tf

FLAGS = None

def main(_):
  # ps_hosts = FLAGS.ps_hosts.split(",")
  # worker_hosts = FLAGS.worker_hosts.split(",")

  ps_hosts = ["localhost:2221", "localhost:2222"]
  ps_hosts = ["localhost:2223", "localhost:2224", "localhost:2225"]
  
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": ps_hosts})


  # input flags
  tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
  tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
  FLAGS = tf.app.flags.FLAGS

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  # config
  batch_size = 100
  learning_rate = 0.0005
  training_epochs = 20
  logs_path = "/tmp/mnist/1"

  # load mnist data set
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      loss = ...
      global_step = tf.contrib.framework.get_or_create_global_step()

      # input images
      with tf.name_scope('input'):
        # None -> batch size can be any size, 784 -> flattened mnist image
        x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
        # target 10 output classes
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

      # model parameters will change during training so we use tf.Variable
      tf.set_random_seed(1)
      with tf.name_scope("weights"):
        W1 = tf.Variable(tf.random_normal([784, 100]))
        W2 = tf.Variable(tf.random_normal([100, 10]))

      # bias
      with tf.name_scope("biases"):
        b1 = tf.Variable(tf.zeros([100]))
        b2 = tf.Variable(tf.zeros([10]))

      # implement model
      with tf.name_scope("softmax"):
        # y is our prediction
        z2 = tf.add(tf.matmul(x,W1),b1)
        a2 = tf.nn.sigmoid(z2)
        z3 = tf.add(tf.matmul(a2,W2),b2)
        y  = tf.nn.softmax(z3)

      # specify cost function
      with tf.name_scope('cross_entropy'):
        # this is our cost
        cross_entropy = tf.reduce_mean(
                  -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

      # specify optimizer
      with tf.name_scope('train'):
        # optimizer is an "operation" which we can execute in a session
        grad_op = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = grad_op.minimize(cross_entropy, global_step=global_step)

      with tf.name_scope('Accuracy'):
        # accuracy
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      # create a summary for our cost and accuracy
      tf.summary.scalar("cost", cross_entropy)
      tf.summary.scalar("accuracy", accuracy)

      # merge all summaries into a single "operation" which we can execute in a session 
      summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()
      print("Variables initialized ...")

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # mon_sess.run handles AbortedError in case of preempted PS.
        mon_sess.run(train_op)
