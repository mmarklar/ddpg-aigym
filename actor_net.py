import numpy as np
import tensorflow as tf
import math

LEARNING_RATE = 0.0001
BATCH_SIZE = 64
TAU = 0.001
class ActorNet:
    """ Actor Network Model of DDPG Algorithm """
    
    def __init__(self,num_states,num_actions):
        self.g=tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()

            # Create the actor network
            self.wc1, self.wc2, self.wd1, self.out_weight, \
                self.bc1, self.bc2, self.bd1, self.out_bias, \
                self.in_tensor, self.output = self.create_actor_net()

            # Create the target network
            self.t_wc1, self.t_wc2, self.t_wd1, self.t_out_weight, \
                self.t_bc1, self.t_bc2, self.t_bd1, self.t_out_bias, \
                self.t_in_tensor, self.t_output = self.create_actor_net()
            
            
            #cost of actor network:
            self.q_gradient_input = tf.placeholder("float",[None,num_actions]) #gets input from action_gradient computed in critic network file
            self.actor_parameters = [self.wc1, self.wc2, self.wd1, self.out_weight, \
                self.bc1, self.bc2, self.bd1, self.out_bias]

            self.parameters_gradients = tf.gradients(self.output, self.actor_parameters, -self.q_gradient_input)#/BATCH_SIZE) 
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.actor_parameters))  
            #initialize all tensor variable parameters:
            self.sess.run(tf.initialize_all_variables())    
            
            #To make sure actor and target have same intial parmameters copy the parameters:
            # copy target parameters
            self.sess.run([
                self.t_wc1.assign(self.wc1),
                self.t_wc2.assign(self.wc2),
                self.t_wd1.assign(self.wd1),
                self.t_out_weight.assign(self.out_weight),

                self.t_bc1.assign(self.bc1),
                self.t_bc2.assign(self.bc2),
                self.t_bd1.assign(self.bd1),
                self.t_out_bias.assign(self.out_bias)])

            # Update the target network with a combination of the 
            # actor and the target weights/biases
            # TODO: Figure out exactly how they are blended
            self.update_target_actor_op = [
                self.t_wc1.assign(TAU * self.wc1 + (1 - TAU) * self.t_wc1),
                self.t_wc2.assign(TAU * self.wc2 + (1 - TAU) * self.t_wc2),
                self.t_wd1.assign(TAU * self.wd1 + (1 - TAU) * self.t_wd1),
                self.t_out_weight.assign(TAU * self.out_weight + (1 - TAU) * self.t_out_weight),

                self.t_bc1.assign(TAU * self.bc1 + (1 - TAU) * self.t_bc1),
                self.t_bc2.assign(TAU * self.bc2 + (1 - TAU) * self.t_bc2),
                self.t_bd1.assign(TAU * self.bd1 + (1 - TAU) * self.t_bd1),
                self.t_out_bias.assign(TAU * self.out_bias + (1 - TAU) * self.t_out_bias)]

    def create_actor_net(self, num_states=4, num_actions=7):
        # Create the weights for the convolutions and the dense layers
        # 2x conv2d layers, 2x dense with the second being the output layer
        wc1 = tf.Variable(tf.random_normal([5, 5, 2, 32]))
        wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
        wd1 = tf.Variable(tf.random_normal([180000, 1024]))
        out_weight = tf.Variable(tf.random_normal([1024, num_actions]))

        # Create the biases for the layers
        bc1 = tf.Variable(tf.random_normal([32]))
        bc2 = tf.Variable(tf.random_normal([64]))
        bd1 = tf.Variable(tf.random_normal([1024]))
        out_bias = tf.Variable(tf.random_normal([num_actions]))

        # Create the model
        in_tensor = tf.placeholder(tf.float32, [None, 300, 300, 2])

        # Convolution layer with maxpooling
        conv1 = conv2d(in_tensor, wc1, bc1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Convolution layer with maxpooling
        conv2 = conv2d(conv1, wc2, bc2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Flatten the convolution output
        # And run it through the fully connected layer
        fc1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, wd1), bd1)
        fc1 = tf.nn.relu(fc1)

        # Get the output
        output = tf.add(tf.matmul(fc1, out_weight), out_bias)
        return wc1, wc2, wd1, out_weight, bc1, bc2, bd1, out_bias, in_tensor, output
        
    def evaluate_actor(self,state_t):
        return self.sess.run(self.output, feed_dict={self.in_tensor:state_t})        
        
    def evaluate_target_actor(self,state_t_1):
        return self.sess.run(self.t_output, feed_dict={self.t_in_tensor: state_t_1})
        
    def train_actor(self,actor_state_in,q_gradient_input):
        self.sess.run(self.optimizer, feed_dict={ self.in_tensor: actor_state_in, self.q_gradient_input: q_gradient_input})
    
    def update_target_actor(self):
        self.sess.run(self.update_target_actor_op)    

# Taken from conv2d example: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
