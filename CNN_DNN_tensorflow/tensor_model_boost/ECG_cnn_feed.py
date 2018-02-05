import os.path
import time

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report,accuracy_score #模型准确率,查准率，查全率,f1_score

import input_data
import tensor_ecg_model


BOOST_rate=3

# 命令行参数
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 3000, 'Number of steps to run trainer.')

#根据不同模型改变层数
flags.DEFINE_integer('conv1', 32, 'Number of units in conv layer 1.')
flags.DEFINE_integer('conv2', 64, 'Number of units in conv layer 2.')
flags.DEFINE_integer('full', 256, 'Number of units in conv layer 2.')

flags.DEFINE_integer('batch_size', 256, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', r'D:\aa_learning\pypro\python3-practice\ECG_classification\dnn_cnn\tensor_ecg_model', 'Directory to put the training data.')


#生成占位符函数
def placeholder_inputs(batch_size):
  """
  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         tensor_ecg_model.IMAGE_SIZE_w))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder

#字典输入
def fill_feed_dict(data_set, images_pl, labels_pl):
  """
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict



def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  
  #读取数据
  train_dataset,validation_dataset= input_data.read_data_sets()

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    
    #生成占位符
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)
    #利用所有测试集进行验证
    labels_placeholder_all= tf.placeholder(tf.int32, shape=(validation_dataset._num_examples))

    #logits_boost 占位符
    # logits_boost_place=tf.placeholder(tf.float32, shape=(None,tensor_ecg_model.LABEL_SIZE))

    # Build a Graph that computes predictions from the inference model.
    logits= tensor_ecg_model.inference(images_placeholder,
                             FLAGS.conv1,
                             FLAGS.conv2,FLAGS.full)

    # Add to the Graph the Ops for loss calculation.
    loss = tensor_ecg_model.loss(logits, labels_placeholder)


    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = tensor_ecg_model.training(loss, FLAGS.learning_rate)

    #计算准确率,batch
    accuracy,correct_int,labels_true=tensor_ecg_model.accuracy(logits, labels_placeholder)
    #all
    accuracy_all,correct_int_all,labels_true_all=tensor_ecg_model.accuracy(logits, labels_placeholder_all)
    
    # accuracy_boost,correct_int_boost,labels_true_boost=tensor_ecg_model.accuracy(logits_boost_place, labels_placeholder)


    # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.summary.merge(loss.op.name)
    Entropy_summary=tf.summary.scalar('loss', loss)
    training_summary = tf.summary.scalar("training_accuracy", accuracy)
    validation_summary = tf.summary.scalar("validation_accuracy", accuracy)

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    best_validation_ACCURACY=0
    # Start the training loop.
    for step in range(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(train_dataset,
                                 images_placeholder,
                                 labels_placeholder)
      feed_dict_validation = fill_feed_dict(validation_dataset,
                                 images_placeholder,
                                 labels_placeholder)

      #这需要新建一个placeholder，label的大小不符合
      all_feed_validation={images_placeholder:validation_dataset._images, labels_placeholder_all:validation_dataset._labels}

      # Run one step of the add_percent[:,i].assign(-percent_10)model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _,loss_value = sess.run([train_op,loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        
        #训练集准确率
        accu_train,Entropy_summary_str,training_summary_str = sess.run([accuracy,Entropy_summary,training_summary], feed_dict=feed_dict)
        
        summary_writer.add_summary(Entropy_summary_str, step)
        summary_writer.add_summary(training_summary_str, step)
        print("training accuracy: {}".format(accu_train))
        
        #验证集准确率,batch
        accu_validation,logits_get,labels_true_val,validation_summary_str = sess.run([accuracy,logits,labels_true,validation_summary], feed_dict=feed_dict_validation)
        #验证集，all
        logits_get_all,labels_true_val_all = sess.run([logits,labels_true_all], feed_dict=all_feed_validation)
        
        #计算各类的召回率及查准率,原始
        y_pred=np.argmax(logits_get,axis=1)
        report=classification_report(labels_true_val,y_pred,digits=4)
        print('综合报告： ',report)

        summary_writer.add_summary(validation_summary_str, step)       
        print("valid accuracy: {}".format(accu_validation))

        summary_writer.flush()

        
        y_pred=np.argmax(logits_get_all,axis=1)
        report=classification_report(labels_true_val_all,y_pred,digits=4)
        print('综合报告ALL： ')
        print(report)

        # print('before: ')
        # print(logits_get_all[:3])

        #计算各类的召回率及查准率,提升
        sum_column_logits=np.sum(logits_get_all,axis=1)
        # print(sum_column_logits[:3])
        # sum_logits_tile=np.tile(sum_column_logits.reshape(-1,1),tensor_ecg_model.LABEL_SIZE-1)
        # print(sum_logits_tile[:3])
        logits_get_all[:,0]-=np.fabs((sum_column_logits/BOOST_rate))
        # logits_get_all[:,range(1,tensor_ecg_model.LABEL_SIZE)]+=(sum_logits_tile/(BOOST_rate*(tensor_ecg_model.LABEL_SIZE-1)))

        # print('after: ')
        # print(logits_get_all[:3])

        y_pred=np.argmax(logits_get_all,axis=1)
        report=classification_report(labels_true_val_all,y_pred,digits=4)
        print('综合报告ALL（提升）： ')
        print(report)

        

        if accu_validation>best_validation_ACCURACY:
          saver.save(sess,os.path.join(FLAGS.train_dir, 'checkpoint'),global_step=step)
          print('save one model!')
          best_validation_ACCURACY=accu_validation
    print('best: {}'.format(best_validation_ACCURACY))
      # # Save a checkpoint and evaluate the model periodically.
      # if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
      #   checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
      #   saver.save(sess, checkpoint_file, global_step=step)
      #   # Evaluate against the training set.
      #   print('Training Data Eval:')
      #   do_eval(sess,
      #           eval_correct,
      #           images_placeholder,
      #           labels_placeholder,
      #           train_dataset)

      #   # Evaluate against the validation set.
      #   print('Validation Data Eval:')
      #   do_eval(sess,
      #           eval_correct,
      #           images_placeholder,
      #           labels_placeholder,
      #           validation_dataset)

      #   # Evaluate against the test set.
      #   print('Test Data Eval:')
      #   do_eval(sess,
      #           eval_correct,
      #           images_placeholder,
      #           labels_placeholder,
      #           test_dataset)


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()