import tensorflow as tf

from data_provider import read_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("epoch", 1, "Number of epoch")
tf.app.flags.DEFINE_integer("image_size", 48, "The size of image input")
tf.app.flags.DEFINE_boolean("is_train", True, "The size of image input")
tf.app.flags.DEFINE_integer("scale", 3, "the size of scale factor for preprocessing input image")
tf.app.flags.DEFINE_integer("c_dim", 3, "The size of channel")
tf.app.flags.DEFINE_integer("batch_size", 256, "the size of batch")
#tf.app.flags.DEFINE_string("test_img", "C:\\image_path\\1343240909729.jpg", "test_img")
tf.app.flags.DEFINE_float("learning_rate", 1e-5 , "The learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "C:\\Users\\Administrator\\git\\espcn2\\espcn2\\checkpoint", "Name of checkpoint directory")
tf.app.flags.DEFINE_string("inputdata", 'D:\\edsr-data800.tfrecord', "outdata")

def main(_):
    with tf.Session() as sess:
        with tf.name_scope('inputs'), tf.device('/cpu:0'):
            input_, label_, iterator_ = read_data(FLAGS)
        
        counter = 0
        tf.initialize_all_variables().run()
        sess.run(iterator_.initializer)
        label_mean=tf.reduce_mean(label_)
        input_mean=tf.reduce_mean(input_)
        while True:
            counter += 1
            print(counter)
            label_shape=sess.run(label_).shape
            
            print('label_mean',sess.run(label_mean))
            print('input_mean',sess.run(input_mean))
            if not label_shape[0]==FLAGS.batch_size:
                print('error',counter,label_shape[0])
            
    
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()