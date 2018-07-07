#######################################################
# Imports
import numpy as np
#######################################################

class bonsaiPredictorOptim:
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

        # # matrix dotproduct
        X_ = np.dot(self.Z, X.T) / hyperparm_dict['projDim']
        if self.logLevel == "debug":
            self.print_array(X_, "\n\n X_Hat computed is ")


        #compute the scores
        node_id = 0
        score = 0
        parent_prob = 1

        for i in range(0, hyperparm_dict['depth']+1):
             print("##########################################################")
             print("\n\nCurrently working on tree at depth "+str(i))
             print("Tree Node at which we are present is "+str(node_id))


             #W_ = self.W[node_id:node_id+1]
             W_ = self.W[node_id]
             if self.logLevel == "debug":
                 self.print_array(W_, "\n\n W_ array")

             #V_ = self.V[node_id:node_id+1]
             V_ = self.V[node_id]
             if self.logLevel == "debug":
                 self.print_array(V_, "\n\n V_ array")

             prob = 1
             # formula  s = s+ wzx * tanh (sigma * vzx)
             print("\n\n Score before adding to the previous score is >>> "
                   +str(prob*np.multiply(np.dot(W_, X_), np.tanh(hyperparm_dict['sigma'] * np.dot(V_, X_)))))

             score = score + prob *(np.multiply(np.dot(W_, X_), np.tanh(hyperparm_dict['sigma'] * np.dot(V_, X_))))
             print("\n\n Score after adding to the previous score is >>>"+str(score))

             if i < hyperparm_dict['depth']:

                 T_ = np.reshape(self.T[node_id], [-1, hyperparm_dict['projDim']])
                 if self.logLevel == "debug":
                    self.print_array(T_, "\n\n T_ array")

                 T_X = np.dot(T_,X_)

                 if self.logLevel == "debug":
                     self.print_array(T_X, "\n\n T_X array")
                 theta_value = T_X.item(0, 0)
                 print("value removed from the numpy array >>> "+str(theta_value))

                 # move left or right base on the theta_value
                 if theta_value >= 0:
                      node_id = 2 * node_id + 1
                 else:
                      node_id = 2 * node_id + 2

        print("\n\n###########################################################################")
        print("Final Computed  score  after traversing through the tree is >>> "+str(score))

        # return the final predictions.
        return score


    # Function 3
    # debug function just to dump the shape
    def print_array(self, arr, message):
        print(message)
        print(arr)
        print(arr.shape)
