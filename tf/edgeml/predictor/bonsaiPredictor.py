#######################################################
# Imports
import numpy as np


#######################################################

class bonsaiPredictor:
    # constructor function
    def __init__(self,dir_path,log_level="warn"):

        """
            W [numClasses*totalNodes, projectionDimension]
            V [numClasses*totalNodes, projectionDimension]
            Z [projectionDimension, dataDimension + 1]  => features + bias of 1
            T [internalNodes, projectionDimension]

            internalNodes = 2**treeDepth - 1
            totalNodes = 2*internalNodes + 1
        """
        self.hyperparameters = np.load(dir_path.rstrip("/")+'/hyperParam.npy',encoding="latin1")
        self.W = np.load(dir_path.rstrip("/")+'/W.npy')
        self.Z = np.load(dir_path.rstrip("/")+'/Z.npy')
        self.V = np.load(dir_path.rstrip("/")+'/V.npy')
        self.T = np.load(dir_path.rstrip("/")+'/T.npy')
        self.sigmaI = 1e9
        self.logLevel = log_level.lower()

    # Function 2
    # pass in the numpy array containing the values to be predicted.
    def predict(self, test_X):
        # get the hyperparameter dictionary
        hyperparm_dict = self.hyperparameters.item()
        print(hyperparm_dict)

        # compute the number of internal node and total nodes based on depth of tree.
        internalnodes = 2 ** hyperparm_dict['depth'] - 1
        totalnodes = 2 * internalnodes + 1
        print(internalnodes, totalnodes)

        # printing the shape of W/Z/V/T
        print('shape of W >>>> ' + str(self.W.shape))
        print('shape of Z >>>> ' + str(self.Z.shape))
        print('shape of V >>>> ' + str(self.V.shape))
        print('shape of T >>>> ' + str(self.T.shape))
        print('shape of input dataset X >>> ' + str(test_X.shape))

        # create the bias array
        bias_array = np.ones((test_X.shape[0], 1))
        if self.logLevel == "debug":
            self.print_array(bias_array, "\n\nBias Array shape")

        # Normalize the inout dataset
        normalized_X = (test_X - hyperparm_dict['mean']) / hyperparm_dict['std']
        if self.logLevel == "debug":
            self.print_array(normalized_X, "\n\nOur Normalized dataset")

        # add the bias to the  dataset to be predicted.
        X = np.concatenate((normalized_X, bias_array), axis=1)
        if self.logLevel == "debug":
            self.print_array(X, "\n\n Normalized dataset after adding the Bias")

        # matrix dotproduct
        X_ = np.dot(self.Z, X.T) / hyperparm_dict['projDim']

        # list to hold the probablities
        nodeProb = []

        print("num_classes",hyperparm_dict['numClasses'])

        # Compute the root node score
        W_ = self.W[0:(hyperparm_dict['numClasses'])]
        V_ = self.V[0:(hyperparm_dict['numClasses'])]
        # assign the node's prob to be one for root borrowed from tensorflow
        nodeProb.append(1)
        # score of the root node..
        score = nodeProb[0] * np.multiply(np.dot(W_, X_), np.tanh(hyperparm_dict['sigma'] * np.dot(V_, X_)))
        if self.logLevel == "debug":
            self.print_array(score, "\n\n Root node score")

        # compute the scores of the other nodes.
        for i in range(1, totalnodes):

            W_ = self.W[i * hyperparm_dict['numClasses']:((i + 1) * hyperparm_dict['numClasses'])]
            if self.logLevel == "debug":
                self.print_array(W_, "\n\n W_ array")

            V_ = self.V[i * hyperparm_dict['numClasses']:((i + 1) * hyperparm_dict['numClasses'])]
            if self.logLevel == "debug":
                self.print_array(V_, "\n\n V_ array")

            T_ = np.reshape(self.T[int(np.ceil(i / 2) ) - 1], [-1, hyperparm_dict['projDim']])
            #T_ = np.reshape(self.T[int(np.ceil(i / 2) - 1)], [-1, hyperparm_dict['projDim']])
            if self.logLevel == "debug":
                self.print_array(T_, "\n\n T_ array")

            # compute the probablity of the individual node
            prob = (1 + ((-1) ** (i + 1)) * np.tanh(np.multiply(self.sigmaI, np.dot(T_, X_))))
            prob = np.divide(prob, 2)
            if self.logLevel == "debug":
                self.print_array(prob, "\n\n Probablity of the node before multiplying with the probablity of parent "
                                       "node")
            prob = nodeProb[int(np.ceil(i / 2) - 1)] * prob
            nodeProb.append(prob)
            if self.logLevel == "debug":
                self.print_array(prob, "\n\n Probablity of the node after multiplying with the probablity of parent "
                                       "node")

            print("Score computed for node >> " + str(i))
            print(nodeProb[i] * np.multiply(np.dot(W_, X_), np.tanh(hyperparm_dict['sigma'] * np.dot(V_, X_))))

            score += nodeProb[i] * np.multiply(np.dot(W_, X_), np.tanh(hyperparm_dict['sigma'] * np.dot(V_, X_)))

            print("New Score computed for node >> " + str(i))

            print("\n\n ############################################################")
            print(score)

        # return the final predictions.
        return score

    # Function 3
    # debug function just to dump the shape
    def print_array(self, arr, message):
        print(message)
        print(arr)
        print(arr.shape)
