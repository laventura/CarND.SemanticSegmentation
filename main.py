import os.path
import tensorflow as tf
import helper
import warnings
import math 
import numpy as np
from tqdm import tqdm
from distutils.version import LooseVersion
import project_tests as tests
import time

## --- Config Parameters ---
NUM_IMAGES = 290                    # total num of images to process
KEEP_PROB  = 0.5
LEARNING_RATE  = 0.001              # will be decayed in training loop

EPOCHS      = 35 
BATCH_SIZE  = 4
SAVE_DIR    = './checkpoint_dir'    # where checkpoint will be saved
MODEL_NAME  = 'my-model'

## --- end params ---

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

## Create  SAVE_DIR if not exists
if not os.path.exists(SAVE_DIR):
    print('Creating dir to save checkpoints:', SAVE_DIR)
    os.mkdir(SAVE_DIR)


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load pre-trained VGG model
    print('Loading VGG16 model from:', vgg_path)
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # Get graph and the layers
    graph   = tf.get_default_graph()
    
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob   = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3      = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4      = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7      = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, layer3, layer4, layer7

tests.test_load_vgg(load_vgg, tf)


def tf_norm(sd=0.01):
    ''' Return Truncated normal initializer. Helper func '''
    return tf.truncated_normal_initializer(stddev=sd)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    #Implement function

    # 1x1 conv for three layers
    L7      = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, kernel_initializer=tf_norm(sd=0.01))
    L4      = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, kernel_initializer=tf_norm(sd=0.01))
    L3      = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, kernel_initializer=tf_norm(sd=0.01))

    ## Upsample Layer7 and add to Layer4; 
    net_input   = tf.layers.conv2d_transpose(L7, num_classes, 
                                kernel_size=4, strides=2, padding='same',
                                kernel_initializer=tf_norm(sd=0.01))
    netinput    = tf.add(net_input, L4)

    ## Upsample the net input, and add to Layer3
    net_input   = tf.layers.conv2d_transpose(net_input, num_classes,
                                kernel_size=4, strides=2, padding='same',
                                kernel_initializer=tf_norm(sd=0.01))
    net_input   = tf.add(net_input, L3)  

    ## Upsample and return
    net_input   = tf.layers.conv2d_transpose(net_input, num_classes,
                                kernel_size=16, strides=8, padding='same',
                                kernel_initializer=tf_norm(sd=0.01))

    return net_input

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Implement function

    ## reshape; flatten into num_classes 1D 
    logits          = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    correct_label   = tf.reshape(correct_label, (-1, num_classes), name='correct_label')

    ## Cross entropy logits / loss
    cross_entropy_logits = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    cross_entropy_loss   = tf.reduce_mean(cross_entropy_logits, name='cross_entropy_loss')

    ## Training operation
    training_op         = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss, name='train_op')

    ## FIXME: should this be (cross_entropy_logits, train_op, cross_entropy_loss) ?
    return logits, training_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    #  Implement function
    print('-- Starting Training --')
 
    # TODO: a) Save model (meta graph and weights) before train
    saver = tf.train.Saver(allow_empty=True)  # default saver

    print()
    print('--- Training ---')
    print("  params: epochs {}, batch_size {}, keep_prob {}, learning_rate {}".format(epochs, batch_size, KEEP_PROB, LEARNING_RATE))

    K = 0.01
    lr = LEARNING_RATE

    ## 1 - For each epoch...
    for epoch in range(1, epochs+1):
        lr = lr / (1 + K * epoch)   # decay the learning_rate
        print("Epoch: {} of {}, lr {:.9f}".format(epoch, epochs, lr))

        time_epoch_start = time.time()

        # 1a - Generate a Batch
        total_loss      = []
        count           = 0
        # 1b - ...for each Batch...
        for batch_image, batch_label in get_batches_fn(batch_size):
            count += 1

            # 1c - ...create a Feed Dict
            feed_dict = {
                input_image:    batch_image,
                correct_label:  batch_label,
                keep_prob:      KEEP_PROB,
                learning_rate:  lr 
            }

            t0 = time.time()
            # 1d - Train step and calculate Loss
            _, loss     = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            total_loss.append(loss)

            if count % 10 == 0:
                t1 = time.time()
                time_batch = (t1 - t0) / 10
                print("  Epoch {} / {}, counter {}, batch training loss: {:.5f}, secs {:.2f}" \
                        .format(epoch, epochs, count, loss, time_batch))
            
        # 1e - output loss
        mean_loss = np.sum(total_loss) / NUM_IMAGES
        time_epoch_end = time.time()
        time_epoch     = time_epoch_end - time_epoch_start

        print("Epoch Loss: mean {:.5f}, last loss {:.5f}, secs {:.2f}".format(mean_loss, loss, (time_epoch) ) )

        # TODO: b) Save model periodically
        if epoch % 5 == 0:
            print(' Saving model for epoch:', epoch)
            saver.save(sess, SAVE_DIR + '/' + MODEL_NAME, global_step=epoch, write_meta_graph=True )
    print(' Saving model for epoch:', epoch)
    saver.save(sess, SAVE_DIR + '/' + MODEL_NAME, global_step=epoch, write_meta_graph=True )
    print('--- Done ---\n')
    
    return

tests.test_train_nn(train_nn)

def get_latest_checkpoint_number(save_dir):
    ''' Helper func. Return the latest checkpoint number from save_dir
    '''
    epoch_num = 0
    kvpairs = []
    checkpoint_file = os.path.join(save_dir, 'checkpoint')
    """ Note: 'checkpoint' file format is:
        model_checkpoint_path: "my-model-2"
        all_model_checkpoint_paths: "my-model-0"
        all_model_checkpoint_paths: "my-model-1"
    """
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rt') as inf:
            for ln in inf:
                ln = ln.strip('\n')
                ln = ln.replace("\"", '')
                ln = ln.replace(' ', '')
                k, v = ln.split(':')
                if k == 'model_checkpoint_path':
                    chk_value = v
                    break
        epoch_num = int(chk_value.split('-')[-1])
    
    print(' Last epoch was: ', epoch_num)
    return epoch_num + 1

## --- Continue Training ---
def continue_training(sess, epochs, batch_size, get_batches_fn, last_epoch_num):
    ''' Continue training for given number of epochs using specified batch_size
        Called when the model has been trained once, and you want to further train using existing checkpoint
        Returns the following tensors: 
            input_image:   
            logits:
            keep_prob:

    '''
    print('=== Continuing Training === (for epochs: ', epochs, ')')

    ### new - get from collection
    keep_prob   = tf.get_collection('keep_prob')[0]
    input_image = tf.get_collection('input_image')[0]
    logits      = tf.get_collection('logits')[0]
    correct_label      = tf.get_collection('correct_label')[0]
    learning_rate      = tf.get_collection('learning_rate')[0]

    # Grab the Operations
    cross_entropy_loss  = tf.get_collection('cross_entropy_loss')[0]
    train_op            = tf.get_collection('train_op')[0]

     # a) Save model (meta graph and weights) before train
    saver = tf.train.Saver(allow_empty=True)

    ## Init the vars -- this is NOT needed
    # sess.run(tf.global_variables_initializer())
    print("  params: epochs {}, batch_size {}, keep_prob {}, learning_rate {}".format(epochs, batch_size, KEEP_PROB, LEARNING_RATE))

    K = 0.01
    lr = LEARNING_RATE

    ## 1 - For each epoch...
    total_epochs = epochs + last_epoch_num
    for epoch in range(last_epoch_num, total_epochs):
        lr = lr / (1. + K * (epoch-last_epoch_num))  # decay LR
        print("Epoch: {} of {}, lr {:.9f}".format(epoch, total_epochs, lr))

        time_epoch_start = time.time()

        # 1a - Generate a Batch
        total_loss      = []
        count           = 0
        # 1b - for each Batch
        for batch_image, batch_label in get_batches_fn(batch_size):

            count += 1
            # 1c - create a Feed Dict
            feed_dict = {
                input_image:    batch_image,
                correct_label:  batch_label,
                keep_prob:      KEEP_PROB,
                learning_rate:  lr 
            }

            t0 = time.time() 
            
            # 1d - Train and calc. Loss
            _, loss     = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            total_loss.append(loss)

            if count % 10 == 0:
                t5 = time.time()
                batch_time = (t5 - t0) / 10
                print("  Epoch {} / {}, counter {}, batch training loss: {:.5f}, secs {:.2f}" \
                    .format(epoch, total_epochs, count, loss, batch_time))
            
        # 1e - output loss
        mean_loss       = np.sum(total_loss) / NUM_IMAGES
        time_epoch_end  = time.time()
        time_epoch      = time_epoch_end - time_epoch_start
        print("Epoch Loss: mean {:.5f}, last loss {:.5f}, secs {:.2f}".format(mean_loss, loss, time_epoch) )

        #  b) Save model periodically
        if epoch % 5 == 0:
            print('Saving model for epoch:', epoch)
            saver.save(sess, SAVE_DIR + '/' + MODEL_NAME, global_step=epoch, write_meta_graph=True )
    print('Saving model...')
    saver.save(sess, SAVE_DIR + '/' + MODEL_NAME, global_step=epoch, write_meta_graph=True )
    print('--- Done ---\n')
    return input_image, logits, keep_prob

### --- main entry method ---
def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)


    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        print('Looking for VGG at:', vgg_path)
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        ## (1) used checkpointed model, if exists, else load default VGG
        print('..ALSO looking for latest checkpoint in:', SAVE_DIR)
        latest_ckpt_name = tf.train.latest_checkpoint(SAVE_DIR)
        print('  latest ckpt is:', latest_ckpt_name)
        if latest_ckpt_name is not None:
            META_GRAPH_NAME = latest_ckpt_name + '.meta'
            print('  looking for meta graph:', META_GRAPH_NAME)
            new_saver = tf.train.import_meta_graph(META_GRAPH_NAME)  

            print('  restoring model from latest ckpt:', latest_ckpt_name)
            new_saver.restore(sess, latest_ckpt_name)
            print('>>>>>>>>>>')
            print('  NOTE: TO START TRAINING FROM SCRATCH, remove/rename the dir:', SAVE_DIR)
            print('<<<<<<<<<<')

            last_epoch_num = get_latest_checkpoint_number(SAVE_DIR)

            #  Run training from using this checkpoint model
            input_image, logits, keep_prob  = continue_training(sess, EPOCHS, BATCH_SIZE, get_batches_fn, last_epoch_num)
    
        else:
            print('>>>>  No previous checkpoint found. Will train from scratch <<<<')

            # TODO OPTIONAL: Augment Images for better results
            #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

            ## Placeholder;  num_classes = 2
            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
            correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))  

            # TODO: Build NN using load_vgg, layers, and optimize function
            input_image, keep_prob, l3_out, l4_out, l7_out = load_vgg(sess, vgg_path)

            net_output = layers(l3_out, l4_out, l7_out, num_classes)

            logits, train_op, cross_entropy_loss = optimize(net_output, correct_label, learning_rate, num_classes)

            ## Add ops/tensors to collection - to save in the checkpoint for later use
            tf.add_to_collection('input_image', input_image)
            tf.add_to_collection('keep_prob', keep_prob)
            tf.add_to_collection('learning_rate', learning_rate)
            tf.add_to_collection('correct_label', correct_label)
            tf.add_to_collection('logits', logits)
            tf.add_to_collection('cross_entropy_loss', cross_entropy_loss)
            tf.add_to_collection('train_op', train_op)
           
            ## Init variables
            sess.run(tf.global_variables_initializer())

            #  Train NN using the train_nn function
            train_nn(sess, EPOCHS, BATCH_SIZE, 
                    get_batches_fn, train_op, cross_entropy_loss, 
                    input_image, correct_label, keep_prob, learning_rate)
                    
        ## Done Training; now Inference
        #  Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        
        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
