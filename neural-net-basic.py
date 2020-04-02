import numpy as np    # necessary for matrix manipulation

class NeuralNetwork():

    def __init__(self):
        # Seed the random number generator
        np.random.seed(1)

        # Set synaptic weights to a 3x1 matrix,
        # with values from -1 to 1 and mean 0
        self.synaptic_weights = 2 * np.random.random((3,1)) - 1


    def sigmoid(self,x):
        """
        Takes in weighted sum of the inputs and normalizes
        them through between 0 and 1 through a sigmoid function
        """
        return 1/ (1+np.exp(-x))

    def sigmoid_derivate(self,x):
        """
        The derivative of the sigmoid function used to
        calculate necessary weight adjustments
        """
        return x * (1-x)

    def train(self,training_inputs,training_outpus,train_iterations):
        """
        We train the model through trial and error, adjusting the
        synaptic weights each time to get a better result
        """
        for i in range(train_iterations):   
            output  = self.think(training_inputs)
            error = training_outpus - output            # calculating error difference 
            # Multiply error by input and gradient of the sigmoid function
            # Less confident weights are adjusted more through the nature of the function
            adjustments = np.dot(training_inputs.T,error*self.sigmoid_derivate(output))   
            self.synaptic_weights += adjustments                       # change weights to new one 



    def think(self,inputs):
        """
        Pass inputs through the neural network to get output
        """
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs,self.synaptic_weights))   # dot product of input with weights and pass to actiation fucntion sigmoid
        return output 


if __name__ == "__main__":      

    neural_network = NeuralNetwork()
    print("Random Synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set, with 4 examples consisting of 3
    # input values and 1 output value
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]]) 

    training_outpus = np.array([[0,1,1,0]]).T        # training outputs map to train inputs and since it is 1 row and 4 col we take transpose so to make it 4*1

    #train our neural network
    neural_network.train(training_inputs,training_outpus,100)
    print("Synaptics weights after training: ")
    print(neural_network.synaptic_weights)

    #take test data adn predict
    A  = str(input("Input one: "))
    B  = str(input("Input two: "))
    C  = str(input("Input three: "))

    print("New test inpt data: ",A,B,C)
    print("Output prediction: ")
    print(neural_network.think(np.array([A,B,C])))

