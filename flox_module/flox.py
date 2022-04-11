import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time

from funcx.sdk.client import FuncXClient
from funcx.sdk.executor import FuncXExecutor


def get_edge_weights(sample_counts):
    """
    Takes an array of numbers and returns their fractions of the total number of samples
    Can be used to find weights for the weighted_average 

    Parameters
    ----------
    sample_counts: numpy array of integers

    Returns
    -------
    fractions: numpy array
    
    """
    total = sum(sample_counts)
    fractions = sample_counts/total
    return fractions

def eval_model(m, x, y, silent=False):
    """
    Evaluate the model on a datset

    Parameters
    ----------
    m: Tensorflow model

    x: numpy array
        dataset entries, e.g. x_test
    
    y: numpy array
        labels for the entries, e.g., y_test
    
    """
    score = m.evaluate(x, y, verbose=0)
    if not silent:
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
    return score[0], score[1]

def training_function(json_model_config, 
                      global_model_weights, 
                      num_samples=None,
                      epochs=10,     
                      data_source="keras",
                      keras_dataset="mnist",
                      preprocess=True,
                      path_dir='/home/pi/datasets', 
                      x_train_name="mnist_x_train.npy", 
                      y_train_name="mnist_y_train.npy",
                      input_shape=(32, 28, 28, 1),
                      loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"]                                               
):
    """
    This function gets deployed to the given endpoints with corresponding parameters. 
    Returns the updated model weights and number of samples it was trained on.

    Parameters
    ----------
    json_model_config: str
        configuration of the TF model retrieved using model.to_json()

    global_model_weights: numpy array
        a numpy array with weights of the TF model

    num_samples: int
        if data_source="keras", randomly samples n data points from (x_train, y_train)

    epochs: int
        the number of epochs to train the model for

    data_source: str
        the function supports three data sources: "local", "keras"
        for "local" and "keras", see get_local_data and get_keras_data functions

    keras_dataset: str
        specifies one of the default keras datasets to use if using "keras" data source
        ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'reuters', 'boston_housing']

    preprocess: boolean
            if True, will attempt to preprocess your data from "keras" data sources

    path_dir: str
        path to the folder with x_train and y_train; needed when data_sourse="local"

    x_train_name: str
        filename for x_train; needed when data_sourse="local"

    y_train_name: str
        file name for y_train; needed when data_sourse="local"

    input_shape: tupple
        input shape for the provided model
    
    loss: str
        loss for TF's model.fit() function

    optimizer: str
        optimizer for for TF's model.fit() function

    metrics: str/list 
        metrics for TF's model.fit() function. E.g, metrics=["accuracy"],

    Returns
    -------
    model_weights: numpy array
        updated TF model weights after training on the local data

    samples_counts: int
        the number of samples the model was trained on. 
        Can be used to find the weighted average by # of samples seen

    """
    # import all the dependencies required for funcX functions)
    from tensorflow import keras
    import numpy as np
    import os

    # retrieve (and optionally process) the data
    if data_source == 'keras':
        available_datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'reuters', 'boston_housing']
        dataset_mapping= {
            'mnist': keras.datasets.mnist,
            'fashion_mnist': keras.datasets.fashion_mnist,
            'cifar10': keras.datasets.cifar10,
            'cifar100': keras.datasets.cifar100,
            'imdb': keras.datasets.imdb,
            'reuters': keras.datasets.reuters,
            'boston_housing': keras.datasets.boston_housing
        }
        image_datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']

        # check if the Keras dataset exists
        if keras_dataset not in available_datasets:
            raise Exception(f"Please select one of the built-in Keras datasets: {available_datasets}")

        else:
            # load the data
            (x_train, y_train), _ = dataset_mapping[keras_dataset].load_data()

            # take a random set of images
            if num_samples:
                idx = np.random.choice(np.arange(len(x_train)), num_samples, replace=True)
                x_train = x_train[idx]
                y_train = y_train[idx]

            if preprocess:
                # do default image processing for built-in Keras images    
                if keras_dataset in image_datasets:
                    # Scale images to the [0, 1] range
                    x_train = x_train.astype("float32") / 255

                    # Make sure images have shape (num_samples, x, y, 1) if working with MNIST images
                    if x_train.shape[-1] not in [1, 3]:
                        x_train = np.expand_dims(x_train, -1)

                    # convert class vectors to binary class matrices
                    if keras_dataset == 'cifar100':
                        num_classes=100
                    else:
                        num_classes=10
                        
                    y_train = keras.utils.to_categorical(y_train, num_classes)

    elif data_source == 'local':        
        # construct the path
        x_train_path_file = os.sep.join([path_dir, x_train_name])
        y_train_path_file = os.sep.join([path_dir, y_train_name])

        # load the files
        with open(x_train_path_file, 'rb') as f:
            x_train = np.load(f)
            
        with open(y_train_path_file, 'rb') as f:
            y_train = np.load(f)

        # if preprocess is True & the function is valid, preprocess the data
        if preprocess:
            # check if a valid function was given
            depth = input_shape[3]
            image_size_y = input_shape[2]
            image_size_x = input_shape[1]

            # take a limited number of samples, if indicated
            if num_samples:
                idx = np.random.choice(np.arange(len(x_train)), num_samples, replace=True)
                x_train = x_train[idx]
                y_train = y_train[idx]
            
            # reshape and scale to pixel values to 0-1
            x_train = x_train.reshape(len(x_train), image_size_x, image_size_y, depth)
            x_train = x_train / 255.0

    else:
        raise Exception("Please choose one of data sources: ['local', 'keras']")

    # create the model
    model = keras.models.model_from_json(json_model_config)

    # compile the model and set weights to the global model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    try:
        model.set_weights(global_model_weights)
    except:
        # some older TF versions require the model to be 'built' first
        model.build(input_shape=input_shape)
        model.set_weights(global_model_weights)

    # train the model on the local data and extract the weights
    model.fit(x_train, y_train, epochs=epochs)
    model_weights = model.get_weights()

    # transform to a numpy array
    np_model_weights = np.asarray(model_weights, dtype=object)

    # return the updated weights and number of samples the model was trained on
    return {"model_weights":np_model_weights, "samples_count": x_train.shape[0]}

def federated_learning(global_model, 
                      endpoint_ids, 
                      num_samples=100,
                      epochs=5,
                      loops=1,
                      time_interval=0,
                      aggregation_mode="weighted_average",
                      data_source: str = "keras",
                      keras_dataset = "mnist",  
                      preprocess=False,
                      path_dir='/home/pi/datasets', 
                      x_train_name="mnist_x_train.npy", 
                      y_train_name="mnist_y_train.npy",
                      input_shape=(32, 28, 28, 1),
                      loss="categorical_crossentropy",
                      optimizer="adam", 
                      metrics=["accuracy"],
                      evaluation_function=eval_model,
                      x_test=None,
                      y_test=None
                      ):
    """
    Facilitates Federated Learning for *loops* rounds. 

    Parameters
    ----------
    global_model: TF model object
        compiled TF model that will be deployed for training on the endpoints

    endpoint_ids: list of str
        a list with endpoint_ids to include in the FL process. 
        The ids can be retrieved by running 'funcx-endpoint list' on participating devices

    num_samples: int or list
        indicates how many samples to get for training on endpoints
        if int, applies the same num_samples to all endpoints. 
        if list, it will use the corresponding number of samples for each device
        the list should have the same number of entries as the number of endpoints


    epochs: int or list
        indicates how many epochs to use for training on endpoints
        if int, applies the same number of epochs to all endpoints. 
        if list, it will use the corresponding number of epochs for each device
        the list should have the same number of entries as the number of endpoints

    loops: int
        defines how many FL rounds to run. Each round consists of deploying the mode, training
        aggregating the updates, and reassigning new weights to the model. 

    time_interval: int
        defines the pause between FL rounds in seconds (default=0). Can be useful if you want to run 
        a round every minute/hour/day.

    aggregation_mode: str
        defines the aggregation mode from a choice of a simple 'average' or a 'weighted_average'
        the 'weighted_average' aggregates the updates based on how many samples each updates has 
        been trained on. This gives more weight to updates from devices that have seen more data.

    data_source: str
        the function supports three data sources: "local", "keras"
        for "local" and "keras", see get_local_data and get_keras_data functions

    keras_dataset: str
        specifies one of the default keras datasets to use if using "keras" data source
        ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'reuters', 'boston_housing']

    preprocess: boolean
            if True, will attempt to preprocess your data from "keras" data sources

    path_dir: str
        path to the folder with x_train and y_train; needed when data_sourse="local"

    x_train_name: str
        filename for x_train; needed when data_sourse="local"

    y_train_name: str
        file name for y_train; needed when data_sourse="local"

    input_shape: tupple
        input shape for the provided model
    
    loss: str
        loss for TF's model.fit() function

    optimizer: str
        optimizer for for TF's model.fit() function

    metrics: str/list 
        metrics for TF's model.fit() function. E.g, metrics=["accuracy"],

    evaluation_function: function
        if supplied, evaluates the model on x_test and y_test

    x_test: list/numpy array/tensors
        x_test data for testing

    y_test: list
        y_test labels for x_test


    Returns
    -------
    global_model: TF model
        the original model but with updated weights after all the FL rounds

    """
    # instantiate the FuncXExecutor
    fx = FuncXExecutor(FuncXClient())

    # if num_samples or epochs is an int, convert to list so the same number can be applied to all endpoints
    if type(num_samples) == int:
        num_samples = [num_samples]*len(endpoint_ids)

    if type(epochs) == int:
        epochs = [epochs]*len(endpoint_ids)
    
    # start running FL loops
    for i in range(loops):

        # get the model's architecture and weights
        json_config = global_model.to_json()
        gm_weights = global_model.get_weights()
        gm_weights_np = np.asarray(gm_weights, dtype=object)

        #fx = FuncXExecutor(FuncXClient())
        tasks = []

        # submit the corresponding parameters to each endpoint for a round of FL 
        for e, num_s, num_epoch, path_d in zip(endpoint_ids, num_samples, epochs, path_dir): 
            tasks.append(fx.submit(training_function, 
                                   json_model_config=json_config, 
                                    global_model_weights=gm_weights_np, 
                                    num_samples=num_s,
                                    epochs=num_epoch,
                                    data_source=data_source,
                                    keras_dataset=keras_dataset,
                                    preprocess=preprocess,
                                    path_dir=path_d,
                                    x_train_name=x_train_name,
                                    y_train_name=y_train_name,
                                    input_shape=input_shape,
                                    loss=loss,
                                    optimizer=optimizer,
                                    metrics=metrics,
                                    endpoint_id=e))
        
        # extract model updates from each endpoints once they are available
        model_weights = [t.result()["model_weights"] for t in tasks]
        
        # aggregate the updates using simple 'average' or 'weighted_average'
        if aggregation_mode == "average":
            average_weights = np.mean(model_weights, axis=0)

        elif aggregation_mode == "weighted_average":
            # get the weights of model updates based on how much data they have been trained on
            sample_counts = np.array([t.result()["samples_count"] for t in tasks])
            edge_weights = get_edge_weights(sample_counts)
            
            # find weighted average
            average_weights = np.average(model_weights, weights=edge_weights, axis=0)

        else:
            raise Exception(f"Federated mode {aggregation_mode} is not recognized. \
                 Please select one of the available modes: ['average', 'weighted_average']")
            
        # assign the updated weights to the global_model
        global_model.set_weights(average_weights)

        print(f'Epoch {i}, Trained Federated Model')

        # if the all paramters are supplied, evaluate the model
        if x_test is not None and y_test is not None and evaluation_function and callable(evaluation_function):
            loss_eval, accuracy = evaluation_function(global_model, x_test, y_test)

        # if time_interval is supplied, wait for *time_interval* seconds
        if time_interval > 0:
            time.sleep(time_interval)

    return global_model


