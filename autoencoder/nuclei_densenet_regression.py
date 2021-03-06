from __future__ import print_function
import tensorflow as tf

import sys
sys.path.insert(0, '/home/ing/projects/tfmodels')
from tfmodels.segmentation import Regression
from tfmodels.utilities.ops import *

"""
Implementation is loosely based on two papers:
Original DenseNet: https://arxiv.org/abs/1608.06993
Fully Convolutional DenseNet: https://arxiv.org/abs/1611.09326
"""
class DenseNet(Regression):
    densenet_defaults={
        ## Number of layers to use for each dense block
        'dense_stacks': [4, 8, 8, 16],
        ## The parameter k in the paper. Dense blocks end up with L*k kernels
        'growth_rate': 16,
        ## Kernel size for all layers. either scalar or list same len as dense_stacks
        'k_size': 3,
        'name': 'reg/densenet',
    }

    def __init__(self, **kwargs):
        self.densenet_defaults.update(**kwargs)

        ## not sure sure it's good to do this first
        for key, val in self.densenet_defaults.items():
            setattr(self, key, val)

        ## The smallest dimension after all downsampling has to be >= 1
        self.n_dense = len(self.dense_stacks)
        print('Requesting {} dense blocks'.format(self.n_dense))
        start_size = min(self.x_dims[:2])
        min_dimension = start_size / np.power(2,self.n_dense+1)
        print('MINIMIUM DIMENSION: ', min_dimension)
        assert min_dimension >= 1

        super(DenseNet, self).__init__(**self.densenet_defaults)

        ## Check input shape is compatible with the number of downsampling modules


    """
    Dense blocks do not change h or w.

    Define the normal CNN transformation as x_i = H(x_(i-1)) .
    DenseNet uses an iterative concatenation for all layers:
        x_i = H([x_(i-1), x_(i-2), ..., x_0])

    Given x_0 ~ (batch_size, h, w, k_in)
    Return x_i ~ (batch_size, h, w, k_in + stacks*growth_rate)
    """
    def _dense_block(self, x_flow, n_layers, concat_input=True, keep_prob=0.8, block_num=0, name_scope='dense'):
        nonlin = self.nonlin
        conv_settings = {'n_kernel': self.growth_rate, 'stride': 1, 'k_size': self.k_size, 'no_bias': 0}
        conv_settings_b = {'n_kernel': self.growth_rate*4, 'stride': 1, 'k_size': 1, 'no_bias': 0}
        print('Dense block #{} ({})'.format(block_num, name_scope))

        concat_list = [x_flow]
        print('\t x_flow', x_flow.get_shape())
        with tf.variable_scope('{}_{}'.format(name_scope, block_num)):
            for l_i in range(n_layers):
                layer_name = 'd{}_l{}'.format(block_num, l_i)
                x_b = nonlin(conv(x_flow, var_scope=layer_name+'b', **conv_settings_b))
                x_b = tf.contrib.nn.alpha_dropout(x_b, keep_prob=keep_prob)
                x_hidden = nonlin(conv(x_b, var_scope=layer_name, **conv_settings))
                concat_list.append(x_hidden)
                x_flow = tf.concat(concat_list, axis=-1, name='concat'+layer_name)
                print('\t\t CONCAT {}:'.format(block_num, l_i), x_flow.get_shape())

            if concat_input:
                x_i = tf.concat(concat_list, axis=-1, name='concat_out')
            else:
                x_i = tf.concat(concat_list[1:], axis=-1, name='concat_out')

        return x_i


    ## theta is the compression factor, 0 < theta <= 1
    ## If theta = 1, then k_in = k_out
    def _transition_down(self, x_in, td_num, theta=0.5, keep_prob=0.8, name_scope='td'):
        nonlin = self.nonlin
        k_out = int(x_in.get_shape().as_list()[-1] * theta)
        print('\t Transition Down with k_out=', k_out)
        conv_settings = {'n_kernel': k_out, 'stride': 1, 'k_size': 1, 'no_bias': 0}

        with tf.variable_scope('{}_{}'.format(name_scope, td_num)):
            x_conv = nonlin(conv(x_in, var_scope='conv', **conv_settings))
            x_conv = tf.contrib.nn.alpha_dropout(x_conv, keep_prob=keep_prob)
            x_pool = tf.nn.max_pool(x_conv, [1,2,2,1], [1,2,2,1], padding='VALID', name='pool')

        return x_pool


    def _transition_up(self, x_in, tu_num, theta=0.5, keep_prob=0.8, name_scope='tu'):
        k_out = int(x_in.get_shape().as_list()[-1] * theta)
        print('\t Transition Up with k_out=', k_out)
        deconv_settings = {'n_kernel': k_out, 'upsample_rate': 2, 'k_size': 3, 'no_bias': 0}

        with tf.variable_scope('{}_{}'.format(name_scope, tu_num)):
            x_deconv = deconv(x_in, var_scope='TU', **deconv_settings)
            x_deconv = tf.contrib.nn.alpha_dropout(x_deconv, keep_prob=keep_prob)

        return x_deconv


    """
    x_in is (batch_size, h, w, channels)

    Similar to Table 2 in https://arxiv.org/abs/1611.09326
    """
    def model(self, x_in, keep_prob=0.8, reuse=False, training=False):
        print('DenseNet Model')
        nonlin = self.nonlin
        print('Non-linearity:', nonlin)

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print('\t x_in', x_in.get_shape())

            ## First convolution gets the ball rolling with a pretty big filter
            dense_ = nonlin(conv(x_in, n_kernel=self.growth_rate*2, stride=2, k_size=5, var_scope='conv1'))
            dense_ = tf.nn.max_pool(dense_, [1,2,2,1], [1,2,2,1], padding='VALID', name='c1_pool')

            ## Downsampling path
            self.downsample_list = []
            for i_, n_ in enumerate(self.dense_stacks[:-1]):
                dense_i = self._dense_block(dense_, n_, keep_prob=keep_prob, block_num=i_, name_scope='dd')
                dense_ = tf.concat([dense_i, dense_], axis=-1, name='concat_down_{}'.format(i_))
                self.downsample_list.append(dense_)
                print('\t DENSE: ', dense_.get_shape())

                dense_ = self._transition_down(dense_, i_, keep_prob=keep_prob)


            print("DOWNSAMPLING LIST:")
            for ds_ in self.downsample_list:
                print(ds_)

            ## bottleneck dense layer
            dense_ = self._dense_block(dense_, self.dense_stacks[-1],
                keep_prob=keep_prob, block_num=len(self.dense_stacks)-1)

            print('\t Bottleneck: ', dense_.get_shape())

            ## Upsampling path -- concat skip connections each time
            for i_, n_ in enumerate(reversed(self.dense_stacks[:-1])):
                dense_ = self._transition_up(dense_, tu_num=i_, keep_prob=keep_prob)

                print('\t Concatenating ', self.downsample_list[-(i_+1)])
                dense_ = tf.concat([dense_, self.downsample_list[-(i_+1)]],
                    axis=-1, name='concat_skip_{}'.format(i_))
                print('\t skip_{}: '.format(i_), dense_.get_shape())

                dense_ = self._dense_block(dense_, n_, concat_input=False,
                    keep_prob=keep_prob, block_num=i_, name_scope='du')
                print('\t dense_up{}: '.format(i_), dense_.get_shape())

            ## Classifier layer
            y_hat_0 = nonlin(deconv(dense_, n_kernel=self.growth_rate*4, k_size=5, pad='SAME', var_scope='y_hat_0'))
            y_hat = deconv(y_hat_0, n_kernel=1, k_size=3, pad='SAME', var_scope='y_hat')
            # y_hat = nonlin(deconv(y_hat_0, n_kernel=self.growth_rate*4, k_size=3, pad='SAME', var_scope='y_hat'))
            # y_hat = conv(y_hat, n_kernel=1, k_size=1, stride=1, pad='SAME', var_scope='y_hat_c')
            y_hat = tf.abs(y_hat)
            # y_hat = tf.nn.sigmoid(y_hat)

        return y_hat


class DenseNetTraining(DenseNet):
    train_defaults = { 'mode': 'TRAIN' }

    def __init__(self, **kwargs):
        self.train_defaults.update(**kwargs)
        super(DenseNetTraining, self).__init__(**self.train_defaults)


class DenseNetInference(DenseNet):
    inference_defaults = { 'mode': 'TEST' }

    def __init__(self, **kwargs):
        self.inference_defaults.update(**kwargs)
        super(DenseNetInference, self).__init__(**self.inference_defaults)
