import csv

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

def make_batches(n, batch_size):
    """Generator to create slices containing batch_size elements, from 0 to n.

    The last slice may contain less than batch_size elements, when batch_size
    does not divide n.
    """
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n) 

class MLP_Regressor_With_Autoencoder():
    
    def __init__(self, input_size, 
             hidden_1_size, 
             hidden_2_size, 
             output_size, 
             hidden_1_activation = tf.nn.leaky_relu, 
             hidden_2_activation = tf.nn.leaky_relu, 
             ae_output_activation = tf.nn.leaky_relu,
             output_activation = tf.nn.leaky_relu,
             tensorboard_path = './logs/tensorboard/',
             verbose = True, 
             verbose_debug = False, 
             weight_init_mean = 0,
             weight_init_std = 0.25,
             l2_norm_beta = 0.02,
             learning_rate_ae = 0.01,
             learning_rate_mlp = 0.005,
             stopping_criteria_ae_1 = 0.001,
             stopping_criteria_ae_2 = 0.00001,
             stopping_criteria_mlp_1 = 0.001,
             stopping_criteria_mlp_2 = 0.00001,             
             epochs_ae = 10,
             epochs_mlp = 300,
             batch_size_ae = 1000,
             batch_size_mlp = 100,
             display_freq_batch_ae = 100,
             display_freq_epoch_ae = 100,
             display_freq_batch_mlp = 100,
             display_freq_epoch_mlp = 100):
                     
        self.input_size = input_size
        self.hidden_1_size = hidden_1_size
        self.hidden_2_size = hidden_2_size
        self.output_size = output_size
        self.hidden_1_activation = hidden_1_activation
        self.hidden_2_activation = hidden_2_activation
        self.ae_output_activation = ae_output_activation
        self.output_activation = output_activation
        self.tensorboard_path = tensorboard_path
        self.verbose = verbose
        self.verbose_debug = verbose_debug
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.l2_norm_beta = l2_norm_beta
        self.learning_rate_ae = learning_rate_ae
        self.learning_rate_mlp = learning_rate_mlp
        self.stopping_criteria_ae_1 = stopping_criteria_ae_1
        self.stopping_criteria_ae_2 = stopping_criteria_ae_2
        self.stopping_criteria_mlp_1 = stopping_criteria_mlp_1
        self.stopping_criteria_mlp_2 = stopping_criteria_mlp_2         
        self.epochs_ae = epochs_ae
        self.epochs_mlp = epochs_mlp
        self.batch_size_ae = batch_size_ae
        self.batch_size_mlp = batch_size_mlp
        self.display_freq_batch_ae = display_freq_batch_ae
        self.display_freq_epoch_ae = display_freq_epoch_ae
        self.display_freq_batch_mlp = display_freq_batch_mlp
        self.display_freq_epoch_mlp = display_freq_epoch_mlp
        
        #Initialize Graph
        self.build_graph()
        self.initialize_graph()
        self.ae_fitted = False
        self.mlp_fitted = False
        
        
    def build_graph(self):
        ''' Set up Tensorflow graph '''
        tf.reset_default_graph()
        self._weight_initializer = tf.truncated_normal_initializer(mean=self.weight_init_mean ,stddev=self.weight_init_std)

        self._l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_norm_beta)
        
        self._global_step = global_step = tf.Variable(0, trainable = False)
        self._increment_global_step_op = tf.assign(self._global_step, self._global_step + 1)
        self._ae_step = tf.Variable(0, trainable = False)
        self._mlp_step = tf.Variable(0, trainable = False)
        
        #  Inputs
        with tf.variable_scope('Input'):
            self._x = tf.placeholder(tf.float32, shape=[None, self.input_size], name='X')
            self._y = tf.placeholder(tf.float32, shape=[None, self.output_size], name='Y')
            
        # Hidden Layer 1
        with tf.variable_scope('H1'):
            self._W_H1 = tf.get_variable('W_H1', dtype=tf.float32, shape=[self.input_size, self.hidden_1_size], initializer=self._weight_initializer, regularizer=self._l2_regularizer)
            self._b_H1 = tf.get_variable('b_H1', dtype=tf.float32, initializer=tf.constant(0., shape=[self.hidden_1_size], dtype=tf.float32))
            tf.summary.histogram('W_H1', self._W_H1)
            tf.summary.histogram('b_H1', self._b_H1)
            self._h1 = tf.matmul(self._x, self._W_H1) + self._b_H1
            self._h1 = self.hidden_1_activation(self._h1)
            
        # AE Output
        with tf.variable_scope('AE_Out'):
            self._W_AE = tf.get_variable('W_AE', dtype=tf.float32, shape=[self.hidden_1_size, self.input_size], initializer=self._weight_initializer, regularizer=self._l2_regularizer)
            self._b_AE = tf.get_variable('b_AE', dtype=tf.float32, initializer=tf.constant(0., shape=[self.input_size], dtype=tf.float32))
            tf.summary.histogram('W_AE', self._W_AE)
            tf.summary.histogram('b_AE', self._b_AE)
            self._AE_Out = tf.matmul(self._h1, self._W_AE) + self._b_AE
            self._AE_Out = self.ae_output_activation(self._AE_Out)
        
        
        # Hidden Layer 2
        with tf.variable_scope('H1'):
            self._W_H2 = tf.get_variable('W_H2', dtype=tf.float32, shape=[self.hidden_1_size, self.hidden_2_size], initializer=self._weight_initializer, regularizer=self._l2_regularizer)
            self._b_H2 = tf.get_variable('b_H2', dtype=tf.float32, initializer=tf.constant(0., shape=[self.hidden_2_size], dtype=tf.float32))
            tf.summary.histogram('W_H2', self._W_H2)
            tf.summary.histogram('b_H2', self._b_H2)
            self._h2 = tf.matmul(self._h1, self._W_H2) + self._b_H2
            self._h2 = self.hidden_2_activation(self._h2)
            
        # Output Layer
        with tf.variable_scope('Output'):
            self._W_Output = tf.get_variable('W_Output', dtype=tf.float32, shape=[self.hidden_2_size, self.output_size], initializer=self._weight_initializer, regularizer=self._l2_regularizer)
            self._b_Output = tf.get_variable('b_Output', dtype=tf.float32, initializer=tf.constant(0., shape=[self.output_size], dtype=tf.float32))
            tf.summary.histogram('W_Output', self._W_Output)
            tf.summary.histogram('b_Output', self._b_Output)
            self._output = tf.matmul(self._h2, self._W_Output) + self._b_Output
            #self._output = tf.nn.leaky_relu(self._output)
        
        # Layer 1 autoencoder trainer
        with tf.variable_scope("AE_Trainer"):
            self._ae_trainable_weights = [self._W_H1, self._b_H1, self._W_AE, self._b_AE]
            with tf.variable_scope("Loss"):
                self._ae_loss = tf.reduce_mean(tf.losses.mean_squared_error(self._x, self._AE_Out), name = 'MSE_Loss')
                tf.summary.scalar("ae_loss", self._ae_loss)
            with tf.variable_scope("Optimizer"):
                self._ae_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ae, name='Adam-op-1')
                self._ae_trainer = self._ae_optimizer.minimize(self._ae_loss, var_list=self._ae_trainable_weights,  global_step=self._ae_step)
        
        # Full Network Trainer
        with tf.variable_scope("Full_Trainer"):
            self._mlp_trainable_weights = [self._W_H2, self._b_H2, self._W_Output, self._b_Output]
            with tf.variable_scope("Loss"):
                self._mse_loss = tf.reduce_mean(tf.losses.mean_squared_error(self._y, self._output), name = 'MSE_Loss')
                self._l2_loss = tf.contrib.layers.apply_regularization(self._l2_regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                self._total_loss = self._mse_loss + self._l2_loss
                tf.summary.scalar("MSE_Loss", self._mse_loss)
                tf.summary.scalar("l2_loss", self._l2_loss)
                tf.summary.scalar("total_loss", self._total_loss)
            with tf.variable_scope("Optimizer"):
                self._mlp_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_mlp, name='Adam-op-2')
                self._mlp_trainer = self._mlp_optimizer.minimize(self._total_loss, var_list=self._mlp_trainable_weights,  global_step=self._mlp_step)

    def initialize_graph(self):
        self._merged_summaries = tf.summary.merge_all()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self._summary_writer = tf.summary.FileWriter(self.tensorboard_path, self.sess.graph)  
        self._graph_initialized = True
        
    def fit_autoencoder(self, X_Train, X_Test):
        # Initialize lists to store loss evaluations
        if not self.ae_fitted:
            self.ae_batch_losses = []
            self.ae_batch_step_numbers = []
            self.ae_evaluation_train_losses = []
            self.ae_evaluation_test_losses = []
            self.ae_evaluation_step_numbers = []
        prev_loss = 1000
        last_learning_rate_change = 0
        
        if self.verbose: print("Training AutoEncoder")
        for epoch in range(int(self.epochs_ae)):
            batch_iteration = 1
            np.random.shuffle(X_Train)
            for batch_slice in make_batches(len(X_Train), self.batch_size_ae):
                X_batch = X_Train[batch_slice]
                
                if self.verbose_debug and self._ae_step.eval(self.sess) == 0: print('AE_debug_1')
                    
                feed_dict = {self._x: X_batch, self._y: np.zeros([len(X_batch), self.output_size])}
                _, batch_summary, loss_batch = self.sess.run([self._ae_trainer, self._merged_summaries, self._ae_loss], feed_dict = feed_dict)
                
                if self.verbose_debug and self._ae_step.eval(self.sess) == 1: print('AE_debug_2')                
                
                self.ae_batch_losses.append(loss_batch)
                self.ae_batch_step_numbers.append(self._ae_step.eval(self.sess))
                
                if self.verbose and batch_iteration % self.display_freq_batch_ae == 0:
                    print(f"Epoch: {epoch + 1} iter {batch_iteration}:\t Loss={round(loss_batch,8)}")
                
                batch_iteration += 1
                
                if self.verbose_debug and self._ae_step.eval(self.sess) == 1:  print('AE_debug_3')             
        
            self._summary_writer.add_summary(batch_summary, self._global_step.eval(self.sess))
            
            loss_train = self.sess.run(self._ae_loss, feed_dict={self._x: X_Train, self._y: np.zeros([len(X_Train), self.output_size])})    
            loss_test = self.sess.run(self._ae_loss, feed_dict={self._x: X_Test, self._y: np.zeros([len(X_Test), self.output_size])})
            self.ae_evaluation_train_losses.append(loss_train)
            self.ae_evaluation_test_losses.append(loss_test)
            self.ae_evaluation_step_numbers.append(self._ae_step.eval(self.sess))
            
            if (epoch+1) % self.display_freq_epoch_ae == 0:
                if self.verbose:
                    print(f"Epoch: {epoch + 1}, train loss: {loss_train}")
                    print(f"Epoch: {epoch + 1}, test loss:  {loss_test}")
                    print('---------------------------------------------------------')
                
                if abs(prev_loss - loss_train) < self.stopping_criteria_ae_1 and (epoch - last_learning_rate_change) > 3: 
                    self._ae_optimizer._lr /= 2
                    if self.verbose: print(f"Reducing LR to {self._ae_optimizer._lr}")
                    last_learning_rate_change = epoch
                if abs(prev_loss - loss_train) < self.stopping_criteria_ae_2:    
                    if self.verbose: print(f"Exit due to stopping criteria.  Improvement less than {self.stopping_criteria_ae_2}")
                    break
                prev_loss = loss_train
            
            self.sess.run(self._increment_global_step_op)
            
        self.ae_fitted = True
            
        
    def fit(self, X_Train, Y_Train, X_Test, Y_Test):
        
        if not self.ae_fitted:
            self.fit_autoencoder(X_Train, X_Test)
        
        if not self.mlp_fitted:
            self.mlp_batch_losses = []
            self.mlp_batch_step_numbers = []
            self.mlp_evaluation_train_losses = []
            self.mlp_evaluation_test_losses = []
            self.mlp_evaluation_gap = []
            self.mlp_evaluation_step_numbers = []
        prev_loss = 1000
        last_learning_rate_change = 0   
        
        if self.verbose: print("Training MLP")
        for epoch in range(int(self.epochs_mlp)):
            batch_iteration = 1
            shuffle(X_Train, Y_Train)
            for batch_slice in make_batches(len(X_Train), self.batch_size_mlp):
                X_Batch = X_Train[batch_slice]
                Y_Batch = Y_Train[batch_slice]
                
                # Run optimization op (backprop)
                feed_dict = {self._x: X_Batch, self._y: Y_Batch}
                _, batch_summary, loss_batch = self.sess.run([self._mlp_trainer, self._merged_summaries, self._mse_loss], feed_dict = feed_dict)
                
                self.mlp_batch_losses.append(loss_batch)  
                self.mlp_batch_step_numbers.append(self._mlp_step.eval(self.sess))
                
                if self.verbose and batch_iteration % self.display_freq_batch_mlp == 0:
                    # display the batch loss and accuracy
                    print(f"Epoch: {epoch + 1} iter {batch_iteration}:\t Loss={round(loss_batch,8)}")
                batch_iteration += 1
                
            # write summary every epoch
            self._summary_writer.add_summary(batch_summary, self._global_step.eval(self.sess))
    
            # Run validation after every epoch
            loss_train = self.sess.run(self._mse_loss, feed_dict={self._x: X_Train, self._y: Y_Train})    
            loss_test = self.sess.run(self._mse_loss, feed_dict={self._x: X_Test, self._y: Y_Test})
            self.mlp_evaluation_train_losses.append(loss_train)
            self.mlp_evaluation_test_losses.append(loss_test)
            self.mlp_evaluation_gap.append(loss_test - loss_train)
            self.mlp_evaluation_step_numbers.append(self._mlp_step.eval(self.sess))
            
            if (epoch+1) % self.display_freq_epoch_mlp == 0:
                if self.verbose:
                    print(f"Epoch: {epoch + 1}, train loss: {loss_train}")
                    print(f"Epoch: {epoch + 1}, test loss:  {loss_test}")
                    print('---------------------------------------------------------')
                
                if abs(prev_loss - loss_train) < self.stopping_criteria_mlp_1 and (epoch - last_learning_rate_change) > 250: 
                    self._mlp_optimizer._lr /= 2
                    if self.verbose: print(f"Reducing LR to {self._mlp_optimizer._lr}")
                    last_learning_rate_change = epoch
                if abs(prev_loss - loss_train) < self.stopping_criteria_mlp_2:    
                    if self.verbose: print(f"Exit due to stopping criteria.  Improvement less than {self.stopping_criteria_mlp_2}")
                    break
                prev_loss = loss_train
            
            self.sess.run(self._increment_global_step_op)
        
        self.mlp_fitted = True
            
    def predict(self, X):
        return self.sess.run(self._output, feed_dict = {self._x: X})
    
    def compute_mse(self, X, Y):
        return self.sess.run(self._mse_loss, feed_dict = {self._x: X, self._y: Y})      
    
    def save_model(self, save_path):
        TFSaver = tf.train.Saver(tf.trainable_variables())
        TFSaver.save(self.sess, save_path)
        
    def restore_model(self, save_path):
        TFSaver = tf.train.Saver(tf.trainable_variables())
        TFSaver.restore(self.sess, save_path)        
        
    def plot_training_loss_curves(self, save_path = None):
        fig = plt.figure();
        fig.set_size_inches(12,8)
        ax = plt.gca()
        ax.plot(self.ae_batch_losses, label='Batch Loss')
        ax.plot(self.ae_evaluation_step_numbers, self.ae_evaluation_train_losses, label='Training Set Loss')
        ax.plot(self.ae_evaluation_step_numbers, self.ae_evaluation_test_losses, label='Test Set Loss')
        ax.set_title("AutoEncoder MSE Loss Function")
        ax.set_xlabel('Batch Number')
        ax.set_ylabel('Mean Squared Error')
        ax.set_ybound(-0.01, 0.1)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path + 'AE_loss_curves.png')
        
        fig = plt.figure();
        fig.set_size_inches(12,8)
        ax = plt.gca()
        ax.plot(self.mlp_batch_step_numbers, self.mlp_batch_losses, label='Batch Loss')
        ax.plot(self.mlp_evaluation_step_numbers, self.mlp_evaluation_train_losses, label='Training Set Loss')
        ax.plot(self.mlp_evaluation_step_numbers, self.mlp_evaluation_test_losses, label='Test Set Loss')
        ax.plot(self.mlp_evaluation_step_numbers, self.mlp_evaluation_gap, label = 'Train-Test Gap')
        ax.set_title("MLP MSE Loss Function")
        ax.set_xlabel('Batch Number')
        ax.set_ylabel('Mean Squared Error')
        ax.set_ybound(-0.01, 0.1)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path + 'MLP_loss_curves.png')
            
    def save_hyperparameters(self, save_path):
        with open(save_path + 'hyperparameters.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['save_location', save_path])
            w.writerow(['input_size', 	self.input_size])
            w.writerow(['hidden_1_size', 	self.hidden_1_size])
            w.writerow(['hidden_2_size', 	self.hidden_2_size])
            w.writerow(['output_size', 	self.output_size])
            w.writerow(['hidden_1_activation', 	self.hidden_1_activation])
            w.writerow(['hidden_2_activation', 	self.hidden_2_activation])
            w.writerow(['ae_output_activation', 	self.ae_output_activation])
            w.writerow(['output_activation', 	self.output_activation])
            w.writerow(['weight_init_mean', 	self.weight_init_mean])
            w.writerow(['weight_init_std', 	self.weight_init_std])
            w.writerow(['l2_norm_beta', 	self.l2_norm_beta])
            w.writerow(['learning_rate_ae', 	self.learning_rate_ae])
            w.writerow(['learning_rate_mlp', 	self.learning_rate_mlp])
            w.writerow(['stopping_criteria_ae_1', 	self.stopping_criteria_ae_1])
            w.writerow(['stopping_criteria_ae_2', 	self.stopping_criteria_ae_2])
            w.writerow(['stopping_criteria_mlp_1', 	self.stopping_criteria_mlp_1])
            w.writerow(['stopping_criteria_mlp_2', 	self.stopping_criteria_mlp_2])
            w.writerow(['epochs_ae', 	self.epochs_ae])
            w.writerow(['epochs_mlp', 	self.epochs_mlp])
            w.writerow(['batch_size_ae', 	self.batch_size_ae])
            w.writerow(['batch_size_mlp', 	self.batch_size_mlp])
            w.writerow(['display_freq_batch_ae', 	self.display_freq_batch_ae])
            w.writerow(['display_freq_epoch_ae', 	self.display_freq_epoch_ae])
            w.writerow(['display_freq_batch_mlp', 	self.display_freq_batch_mlp])
            w.writerow(['display_freq_epoch_mlp', 	self.display_freq_epoch_mlp])


    
        
                
   