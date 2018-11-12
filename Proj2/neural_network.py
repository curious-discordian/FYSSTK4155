
## TWO Options;
## The first is a very simplistic model,
## the second is the alternative example from the lecture notes.
class SimpleNeurons:
    """
    ## For the case where we essentially are looking for how much neighbors are
    ## related, the simplest case is often the easiest.
    ##
    ## We want a simple type of neural network that can do two things, which is
    ## really one thing: It should discern the coupling coefficient of items
    ## in an array.
    ##
    ## What we are interested in then is what the pattern is, i.e. it should learn
    ## by iteratively altering the coupling matrix, and finally returning said matrix
    ## for us to see that it indeed finds the "neighbor" matrix. (this is what a
    ## different neural network would have as a hidden component)
    ##
    ## In some sense also we wish for this to work like an RNN, where it can continuously 
    ## feed back into itself, 
    """
    def __init__(self, X,Y,layers,f=None,df=None):
        self.X = X
        self.Y = Y
        self.output = np.zeros(Y.shape)
        self.layers = layers
        
        if f and df:
            if callable(f) and callable(df):
                self.f = f
                self.df = df
            else:
                print "activation-function and derivative needs to be callable" 
        else:
            #Simple ReLu as default.  
            self.f = lambda x: np.maximum(x,0)
            self.df = lambda x: 1 if x > 0 else 0
        
        
    def feed_forward(self, f):
        # This will simply charge forward without storing the intermittent layers
        # anywhere: 
        # f ==  activation function
        f = self.f
        initialized = [np.array(self.X)] + self.layers
        self.output = reduce(lambda z,w: f(np.dot(z,w)), initialized)

    def backprop(self,df):
        df = self.df
        # simplest way of sloping;
        
        pass
        

class Simple2d(SimpleNeurons):
    # Subclass of simple-neurons.
    #
    pass




## Using the Neural Network example from lecture notes. 

class NeuralNetwork:
    ## From Lecture Notes, and modified to suit temperament.
    def __init__(self,
                 X_data,
                 Y_data,
                 # Default activation set to ReLu and deriv of ReLu 
                 activation_function=lambda x: np.maximum(x,0),
                 D_activation_function=lambda x: np.where(x<=0,0,1),
                 n_hidden_neurons=50,
                 n_categories=10,
                 epochs=10,
                 batch_size=100,
                 eta=0.1,
                 lmbd=0.0,):
        self.X_data_full = X_data
        self.Y_data_full = Y_data

        ## Supply activation function and derivative of
        ## activation functionn.
        self.activation = activation_function
        self.Dactivation = D_activation_function
        
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.activation(self.z_h)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.activation_function(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data
        deriv = self.Dactivation
        error_hidden = np.matmul(error_output, self.output_weights.T) * deriv(a_h)
        # self.a_h * (1 - self.a_h) # = derivative of sigmoid

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()
