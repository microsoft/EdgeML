######################################################################
# imports
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../')
from edgeml.predictor.bonsaiPredictor import bonsaiPredictor
from edgeml.predictor.bonsaiPredictorOptim import bonsaiPredictorOptim


# testing...
def prediction_test(PATH):
    # read the features which needs to be passed to predictor..
    test_X = np.load(PATH.rstrip("/")+'/test.npy')
    filtered_X = test_X[:, 1:]

    bonsai_predictor = bonsaiPredictor(PATH.rstrip("/")+"/",log_level="debug")

    predictions = bonsai_predictor.predict(filtered_X)
    print(predictions)

    #append the predictions to test_x
    print(test_X.shape)
    print(predictions.shape)
    final_X = np.concatenate((test_X,predictions.T),axis=1)

    # just for testing
    import pandas as pd
    # convert the numpy array to dataframe
    df = pd.DataFrame(final_X)
    # testin
    # assign the columns
    df.columns=['truth','irradiation','module_temperature','windspeed','ambient_temperature','minute','prediction']
    print(df.head())
    # write it as a csv file.
    df.to_csv(PATH.rstrip("/")+"/predictions.csv",index=False)



def prediction_test_optim(PATH):
    # read the features which needs to be passed to predictor..
    test_X = np.load(PATH.rstrip("/") + '/test.npy')
    filtered_X = test_X[:, 1:]

    #bonsai_predictor = bonsaiPredictorOptim(PATH.rstrip("/") + "/", log_level="debug")
    bonsai_predictor = bonsaiPredictor(PATH.rstrip("/") + "/", log_level="debug")

    # loop through each of the datapoing
    final_array = np.empty((0,1))
    for array in filtered_X:
        predictions = bonsai_predictor.predict(array.reshape(1, filtered_X.shape[1]))
        print(predictions)
        final_array = np.concatenate((final_array,predictions))

    print(test_X.shape)
    print(final_array.shape)
    final_X = np.concatenate((test_X, final_array),axis=1)

    # just for testing
    import pandas as pd
    # convert the numpy array to dataframe
    df = pd.DataFrame(final_X)
    # testin
    # assign the columns
    print(df)
    df.columns = ['truth', 'irradiation', 'module_temperature', 'windspeed', 'ambient_temperature', 'minute',
                  'prediction']
    print(df.head())
    # write it as a csv file.
    df.to_csv(PATH.rstrip("/") + "/predictions_optim.csv", index=False)


def predicion_debug(PATH):
    # read the features which needs to be passed to predictor..
    test_X = np.load(PATH.rstrip("/") + '/test.npy')
    print(test_X.shape)
    print(test_X[0,:])
    filtered_X = test_X[:, 1:]

    bonsai_predictor = bonsaiPredictorOptim(PATH.rstrip("/") + "/", log_level="debug")
    #bonsai_predictor = bonsaiPredictorOptim(PATH.rstrip("/") + "/", log_level="debug")
    #bonsai_predictor = bonsaiPredictor(PATH.rstrip("/") + "/", log_level="debug")

    i = 0
    prediction_array=np.zeros(shape=(test_X.shape[0]))
    while i < 3:
        predictions = bonsai_predictor.predict(filtered_X[i].reshape(1, filtered_X.shape[1]))
        prediction_array[i]=(predictions)
        i = i + 1
    print (str(prediction_array))
    np.savetxt(PATH+"results.csv", prediction_array, delimiter=",")


if __name__ == "__main__":
    #PATH="/Users/Tanuja/PycharmProjects/EdgeML_TF_updated/EdgeML_TF/bonsai_predictor_values/revision_dataset_BONSAI/test/"
    PATH = sys.argv[1]
    #prediction_test_optim(PATH)
    predicion_debug(PATH)
