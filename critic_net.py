import numpy as np
import tensorflow as tf
import math

TAU = 0.001
LEARNING_RATE= 0.001
BATCH_SIZE = 64
class CriticNet:
    """ Critic Q value model of the DDPG algorithm """
    def __init__(self,num_states,num_actions):
        
        self.g=tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()

            #critic_q_model parameters:
            self.W1_c, self.B1_c, self.W2_c, self.W2_action_c, self.B2_c, self.W3_c, self.B3_c,\
            self.critic_q_model, self.critic_state_in, self.critic_action_in = self.create_critic_net(num_states, num_actions)
            #self.wc1, self.wc2, self.wd1, self.wd2, self.wh1, \
            #self.bc1, self.bc2, self.bd1, self.bd2, self.bd3, self.bh1, \
            #self.critic_q_model, self.state_in, self.action_in = self.create_critic_net(num_states, num_actions)
                                   
            #create target_q_model:
            #self.t_wc1, self.t_wc2, self.t_wd1, self.t_wd2, self.t_wh1, \
            #self.t_bc1, self.t_bc2, self.t_bd1, self.t_bd2, self.t_bd3, self.t_bh1, \
            #self.t_critic_q_model, self.t_state_in, self.t_action_in = self.create_critic_net(num_states, num_actions)

            self.t_W1_c, self.t_B1_c, self.t_W2_c, self.t_W2_action_c, self.t_B2_c, self.t_W3_c, self.t_B3_c,\
            self.t_critic_q_model, self.t_critic_state_in, self.t_critic_action_in = self.create_critic_net(num_states, num_actions)
            
            self.q_value_in=tf.placeholder("float",[None,1]) #supervisor
            #self.l2_regularizer_loss = tf.nn.l2_loss(self.W1_c)+tf.nn.l2_loss(self.W2_c)+ tf.nn.l2_loss(self.W2_action_c) + tf.nn.l2_loss(self.W3_c)+tf.nn.l2_loss(self.B1_c)+tf.nn.l2_loss(self.B2_c)+tf.nn.l2_loss(self.B3_c) 
            self.l2_regularizer_loss = 0.0001*tf.reduce_sum(tf.pow(self.W2_c,2))+ 0.0001*tf.reduce_sum(tf.pow(self.B2_c,2))             
            self.cost=tf.pow(self.critic_q_model-self.q_value_in,2)/BATCH_SIZE + self.l2_regularizer_loss#/tf.to_float(tf.shape(self.q_value_in)[0])
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)
            
            #action gradient to be used in actor network:
            #self.action_gradients=tf.gradients(self.critic_q_model,self.critic_action_in)
            #from simple actor net:
            self.act_grad_v = tf.gradients(self.critic_q_model, self.critic_action_in)
            self.action_gradients = [self.act_grad_v[0]/tf.to_float(tf.shape(self.act_grad_v[0])[0])] #this is just divided by batch size
            #from simple actor net:
            self.check_fl = self.action_gradients             
                       
            #initialize all tensor variable parameters:
            self.sess.run(tf.initialize_all_variables())
            
            #To make sure critic and target have same parmameters copy the parameters:
            # copy target parameters
            """
				self.t_wc1.assign(self.wc1),
				self.t_wc2.assign(self.wc2),
				self.t_wd1.assign(self.wd1),
				self.t_wd2.assign(self.wd2),
				self.t_wh1.assign(self.wh1),

				self.t_bc1.assign(self.bc1),
				self.t_bc2.assign(self.bc2),
				self.t_bd1.assign(self.bd1),
				self.t_bd2.assign(self.bd2),
				self.t_bd3.assign(self.bd3),
				self.t_bh1.assign(self.bh1),
            """
            self.sess.run([
				self.t_W1_c.assign(self.W1_c),
				self.t_B1_c.assign(self.B1_c),
				self.t_W2_c.assign(self.W2_c),
				self.t_W2_action_c.assign(self.W2_action_c),
				self.t_B2_c.assign(self.B2_c),
				self.t_W3_c.assign(self.W3_c),
				self.t_B3_c.assign(self.B3_c)
			])
            '''
            self.t_wc1.assign(TAU * self.wc1 + (1 - TAU) * self.t_wc1),
            self.t_wc2.assign(TAU * self.wc2 + (1 - TAU) * self.t_wc2),
            self.t_wd1.assign(TAU * self.wd1 + (1 - TAU) * self.t_wd1),
            self.t_wd2.assign(TAU * self.wd2 + (1 - TAU) * self.t_wd2),
            self.t_wh1.assign(TAU * self.wh1 + (1 - TAU) * self.t_wh1),

            self.t_bc1.assign(TAU * self.bc1 + (1 - TAU) * self.t_bc1),
            self.t_bc2.assign(TAU * self.bc2 + (1 - TAU) * self.t_bc2),
            self.t_bd1.assign(TAU * self.bd1 + (1 - TAU) * self.t_bd1),
            self.t_bd2.assign(TAU * self.bd2 + (1 - TAU) * self.t_bd2),
            self.t_bd3.assign(TAU * self.bd3 + (1 - TAU) * self.t_bd3),
            self.t_bh1.assign(TAU * self.bh1 + (1 - TAU) * self.t_bh1)
            '''
            
            self.update_target_critic_op = [

                self.t_W1_c.assign(TAU*self.W1_c+(1-TAU)*self.t_W1_c),
                self.t_B1_c.assign(TAU*self.B1_c+(1-TAU)*self.t_B1_c),
                self.t_W2_c.assign(TAU*self.W2_c+(1-TAU)*self.t_W2_c),
                self.t_W2_action_c.assign(TAU*self.W2_action_c+(1-TAU)*self.t_W2_action_c),
                self.t_B2_c.assign(TAU*self.B2_c+(1-TAU)*self.t_B2_c),
                self.t_W3_c.assign(TAU*self.W3_c+(1-TAU)*self.t_W3_c),
                self.t_B3_c.assign(TAU*self.B3_c+(1-TAU)*self.t_B3_c)
            ]
            
    def create_critic_net(self, num_states=4, num_actions=1):
        '''
        # Create the weights for the convolutions and the dense layers
        # 2x conv2d layers, 2x dense with the second being the output layer
        # Conv layers
        wc1 = tf.Variable(tf.random_normal([5, 5, 2, 32]))
        wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
        # Dense layers
        wd1 = tf.Variable(tf.random_normal([180000, 1024]))
        wd2 = tf.Variable(tf.random_normal([num_actions, 1024]))
        # Hidden layers
        wh1 = tf.Variable(tf.random_normal([1024, 1]))
        out_weight = tf.Variable(tf.random_normal([1024, num_actions]))

        # Create the biases for the layers
        # Conv layers
        bc1 = tf.Variable(tf.random_normal([32]))
        bc2 = tf.Variable(tf.random_normal([64]))
        # Dense layers
        bd1 = tf.Variable(tf.random_normal([1024]))
        bd2 = tf.Variable(tf.random_normal([1024]))
        bd3 = tf.Variable(tf.random_normal([1024]))
        # Hidden layers
        bh1 = tf.Variable(tf.random_normal([1024]))
        out_bias = tf.Variable(tf.random_normal([num_actions]))

        # Create the model
        state_in = tf.placeholder(tf.float32, [None, 300, 300, 2])
        action_in = tf.placeholder(tf.float32, [None, num_actions])

        # Convolution layer with maxpooling
        conv1 = conv2d(state_in, wc1, bc1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Convolution layer with maxpooling
        conv2 = conv2d(conv1, wc2, bc2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Flatten the convolution output
        # And run it through the fully connected layer
        fc1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, wd1), bd1)
        fc1 = tf.nn.relu(fc1)

        # Feed the actions through a fully connected layer
        fc2 = tf.add(tf.matmul(action_in, wd2), bd2)
        fc2 = tf.nn.relu(fc2)

        combined = fc1 + fc2

        hidden1 = tf.nn.tanh(combined + bd3)
        critic_q_model = tf.matmul(hidden1, wh1) + bh1

        return wc1, wc2, wd1, wd2, wh1, bc1, bc2, bd1, bd2, bd3, bh1, critic_q_model, state_in, action_in
        '''

        N_HIDDEN_1 = 400
        N_HIDDEN_2 = 300
        # TODO: Fix this hack, we probably need cnn here
        num_states = 180000
        critic_state_in = tf.placeholder("float",[None,num_states])
        critic_action_in = tf.placeholder("float",[None,num_actions])    
    
        W1_c = tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        B1_c = tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        W2_c = tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))    
        W2_action_c = tf.Variable(tf.random_uniform([num_actions,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))    
        B2_c= tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions))) 
        W3_c= tf.Variable(tf.random_uniform([N_HIDDEN_2,1],-0.003,0.003))
        B3_c= tf.Variable(tf.random_uniform([1],-0.003,0.003))
    
        H1_c=tf.nn.softplus(tf.matmul(critic_state_in,W1_c)+B1_c)
        #print("critic_state_in *****************", critic_state_in.shape)
        #print("H1_c *****************", H1_c.shape)
        temp1 = tf.matmul(H1_c, W2_c)
        temp2 = tf.matmul(critic_action_in, W2_action_c)
        temp3 = temp1 + temp2
        #print(temp1.shape, temp2.shape, temp3.shape, B2_c.shape)
        H2_c = tf.nn.tanh(temp3 + B2_c)
        #H2_c=tf.nn.tanh(tf.matmul(H1_c,W2_c)+tf.matmul(critic_action_in,W2_action_c)+B2_c)
            
        critic_q_model=tf.matmul(H2_c,W3_c)+B3_c
            
       
        return W1_c, B1_c, W2_c, W2_action_c, B2_c, W3_c, B3_c, critic_q_model, critic_state_in, critic_action_in
    
    def train_critic(self, state_t_batch, action_batch, y_i_batch ):
        state_t_batch = np.reshape(state_t_batch, [-1, 180000])
        self.sess.run(self.optimizer, feed_dict={self.critic_state_in: state_t_batch, self.critic_action_in:action_batch, self.q_value_in: y_i_batch})
             
    
    def evaluate_target_critic(self,state_t_1,action_t_1):
        state_t_1 = np.reshape(state_t_1, [-1, 180000])
        #print(action_t_1.shape)
        return self.sess.run(self.t_critic_q_model, feed_dict={self.t_critic_state_in: state_t_1, self.t_critic_action_in: action_t_1})    
        
    def compute_delQ_a(self,state_t,action_t):
#        print '\n'
#        print 'check grad number'        
#        ch= self.sess.run(self.check_fl, feed_dict={self.critic_state_in: state_t,self.critic_action_in: action_t})
#        print len(ch)
#        print len(ch[0])        
#        raw_input("Press Enter to continue...")        
        state_t = np.reshape(state_t, [-1, 180000])
        return self.sess.run(self.action_gradients, feed_dict={self.critic_state_in: state_t,self.critic_action_in: action_t})

    def update_target_critic(self):
        self.sess.run(self.update_target_critic_op)

# Taken from conv2d example: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
