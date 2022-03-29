import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from utils.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

class ConvNet(object):
  """
  [conv-relu-max pool]xN - [affine-relu]xM - affine - softmax
  N=2, M=1
  """
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):

    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    # initialize weights and biases

    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    C, H, W = input_dim
    
    # conv 1
    self.params['W1'] = weight_scale * np.random.randn(num_filters,C,filter_size,filter_size)
    self.params['b1'] = np.zeros(num_filters)
    
    # dimensions of conv 1 output
    H_conv1 = (H-filter_size+2*conv_param['pad'])//conv_param['stride'] + 1
    W_conv1 = (W-filter_size+2*conv_param['pad'])//conv_param['stride'] + 1

    # dimensions of pool 1 output
    H_pool1 = (H_conv1-pool_param['pool_height'])//pool_param['stride'] + 1
    W_pool1 = (W_conv1-pool_param['pool_width'])//pool_param['stride'] + 1

    # conv 2
    self.params['W2'] = weight_scale * np.random.randn(num_filters,num_filters,filter_size,filter_size)
    self.params['b2'] = np.zeros(num_filters)

    # dimensions of conv 2 output
    H_conv2 = (H_pool1-filter_size+2*conv_param['pad'])//conv_param['stride'] + 1
    W_conv2 = (W_pool1-filter_size+2*conv_param['pad'])//conv_param['stride'] + 1

    # dimensions of pool 2 output
    H_pool2 = (H_conv2-pool_param['pool_height'])//pool_param['stride'] + 1
    W_pool2 = (W_conv2-pool_param['pool_width'])//pool_param['stride'] + 1

    # hidden affine
    self.params['W3'] = weight_scale * np.random.randn(num_filters*H_pool2*W_pool2,hidden_dim)
    self.params['b3'] = np.zeros(hidden_dim)

    # output affine
    self.params['W4'] = weight_scale * np.random.randn(hidden_dim,num_classes)
    self.params['b4'] = np.zeros(num_classes)

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    # forward pass
    out_conv1, cache_conv1 = conv_forward_fast(X, W1, b1, conv_param)
    out_relu1, cache_relu1 = relu_forward(out_conv1)
    out_pool1, cache_pool1 = max_pool_forward_fast(out_relu1, pool_param)

    out_conv2, cache_conv2 = conv_forward_fast(out_pool1, W2, b2, conv_param)
    out_relu2, cache_relu2 = relu_forward(out_conv2)
    out_pool2, cache_pool2 = max_pool_forward_fast(out_relu2, pool_param)

    # print(out_pool2.shape)
    out_aff1, cache_aff1 = affine_forward(out_pool2, W3, b3)
    out_relu3, cache_relu3 = relu_forward(out_aff1)
    out_aff2, cache_aff2 = affine_forward(out_relu3, W4, b4)

    scores = out_aff2

    if y is None:
      return scores
    
    loss, grads = 0, {}

    # backward pass
    sm_loss, dL = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * (np.sum(self.params['W1']*self.params['W1']) + np.sum(self.params['W2']*self.params['W2']) + np.sum(self.params['W3']*self.params['W3']) + np.sum(self.params['W4']*self.params['W4']))
    loss = sm_loss + reg_loss

    d_aff2, grads['W4'], grads['b4'] = affine_backward(dL, cache_aff2)
    d_relu3 = relu_backward(d_aff2, cache_relu3)
    d_aff1, grads['W3'], grads['b3'] = affine_backward(d_relu3, cache_aff1)

    d_pool2 = max_pool_backward_fast(d_aff1, cache_pool2)
    d_relu2 = relu_backward(d_pool2, cache_relu2)
    d_conv2, grads['W2'], grads['b2'] = conv_backward_fast(d_relu2, cache_conv2)

    d_pool1 = max_pool_backward_fast(d_conv2, cache_pool1)
    d_relu1 = relu_backward(d_pool1, cache_relu1)
    d_conv1, grads['W1'], grads['b1'] = conv_backward_fast(d_relu1, cache_conv1)

    grads['W1'] += self.reg*self.params['W1']
    grads['W2'] += self.reg*self.params['W2']
    grads['W3'] += self.reg*self.params['W3']
    grads['W4'] += self.reg*self.params['W4']

    return loss, grads

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #

    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    C, H, W = input_dim
    
    # conv
    self.params['W1'] = weight_scale * np.random.randn(num_filters,C,filter_size,filter_size)
    self.params['b1'] = np.zeros(num_filters)

    # dimensions of conv output
    H_conv = (H-filter_size+2*conv_param['pad'])//conv_param['stride'] + 1
    W_conv = (W-filter_size+2*conv_param['pad'])//conv_param['stride'] + 1

    # dimensions of pool output
    H_pool = (H_conv-pool_param['pool_height'])//pool_param['stride'] + 1
    W_pool = (W_conv-pool_param['pool_width'])//pool_param['stride'] + 1

    # hidden affine
    self.params['W2'] = weight_scale * np.random.randn(num_filters*H_pool*W_pool,hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)

    # output affine
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim,num_classes)
    self.params['b3'] = np.zeros(num_classes)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    
    out_conv, cache_conv = conv_forward_fast(X, W1, b1, conv_param)
    out_relu1, cache_relu1 = relu_forward(out_conv)
    out_pool, cache_pool = max_pool_forward_fast(out_relu1, pool_param)
    out_aff1, cache_aff1 = affine_forward(out_pool, W2, b2)
    out_relu2, cache_relu2 = relu_forward(out_aff1)
    out_aff2, cache_aff2 = affine_forward(out_relu2, W3, b3)

    scores = out_aff2

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #
    
    sm_loss, dL = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * (np.sum(self.params['W1']*self.params['W1']) + np.sum(self.params['W2']*self.params['W2']) + np.sum(self.params['W3']*self.params['W3']))
    loss = sm_loss + reg_loss

    d_aff2, grads['W3'], grads['b3'] = affine_backward(dL, cache_aff2)
    d_relu2 = relu_backward(d_aff2, cache_relu2)
    d_aff1, grads['W2'], grads['b2'] = affine_backward(d_relu2, cache_aff1)
    d_pool = max_pool_backward_fast(d_aff1, cache_pool)
    d_relu1 = relu_backward(d_pool, cache_relu1)
    d_conv, grads['W1'], grads['b1'] = conv_backward_fast(d_relu1, cache_conv)

    grads['W1'] += self.reg*self.params['W1']
    grads['W2'] += self.reg*self.params['W2']
    grads['W3'] += self.reg*self.params['W3']

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
pass
