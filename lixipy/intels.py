if __name__ == "__main__":
    import basics
    import ops
    import times
    import freqs
else:
    from lixipy import basics
    from lixipy import ops
    from lixipy import times
    from lixipy import freqs

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import datetime
import json
import os
from sklearn.model_selection import train_test_split

# --  Model Training

#MISSING: Check Definition and polish the training for already compiled models
def model_train(model, X, y, epochs, optimizer = 'adam', loss = 'categorical_crossentropy', compile_model = True,
                validation_set_mode = "split", val_part = 0.33, val_X = None, val_y = None, 
                batch_size = 32, random_state=42,
                metrics = ["accuracy", "loss"], plot_metrics = True):
    
    """   
    Description:
        Train the Tensorflow keras Model with the training data.
        If plot_loss and/or plot_accuracy are set as True, plots the accuracy and loss of the model trhoughout the training, both on validation and training set.
        If "split", then the validation set will be obtained as a fraction val_part of the training set
        If "load", then the validation set will be load as val_X and val_y. In this case val_X and val_y cannot be None
        If a value different from "split" or "load" is passed, an error will be raised.
    
    Inputs:
        - model (tensorflow.keras.Sequential): model to train
        - X (numpy.ndarray): array containing features or characteristics of the model to train
        - y (numpy.ndarray): array containing the labels or targets of the model to train
        - epochs (int): number of epochs to train the model
        - optimizer (tensorflow.keras.optimizers or str): optimizer for the training of the model, following tensorflow workflow
        - loss (tensorflow.keras.optimizers or str)): indicate the loss function to use
        - validation_set_mode (str): string with values "split" or "load" indicating how the validation set will be obtained 
        - val_part (float): fraction of the data that will be used as validation set to test the model, in validation_set_mode = "split" case
        - val_X (numpy.ndarray): features to use for validation in validation_set_mode = "load" case
        - val_y (numpy.ndarrat): labels to use for validation in validation_set_mode = "load" case
        - batch_size (int): size of the batch
        - random_state (int): repeatability of the validation set.
        - metrics (list of str): metrics to include in the training process, refer to https://keras.io/api/metrics/
        - plot_metrics (boolean or list of booleans): wether or not to plot each metric
    
    Output:
        - model (tensorflow.keras.Sequential): trained model
        - history (tensorflow.keras.History): object containing info from the training process.
        
    """
    
    assert validation_set_mode == "split" or validation_set_mode == "load", str(validation_set_mode) + " is not a valid option for validation_set_mode"
    
    if validation_set_mode == "split":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_part, random_state=random_state)
    
    if validation_set_mode == "load":
        assert (val_X is not None) and (val_y is not None), "If the Validation Set will be loaded, features and labels should be loaded in val_X and val_y respectively"
        X_train = X
        y_train = y
        X_test = val_X
        y_test = val_y
        
    if type(metrics) == str: #add metrics as a tf.keras.metrics.Metrics object or a custom function
        metrics = [metrics]
    elif type(metrics) == list:
        if not all([type(m) == str for m in metrics]):
            raise ValueError("Training metrics should be passed as a list of strings of the keras module metrics")
    else:
        raise ValueError("Training metrics should be passed as a list of strings of the keras module metrics")
    
    if type(plot_metrics) == bool:
        plot_metrics = [plot_metrics]*len(metrics)
    elif type(plot_metrics) == list:
        assert len(metrics) == len(plot_metrics), "The length of the metrics requested for and the length of the plot metrics boolean don't coincide."
    else:
        raise ValueError("plot_metrics should be passed as a bool (in case the decision is the same for every metric) or a list of bools (one for each metric).")

    if compile_model:
        model.compile(optimizer=optimizer, loss=loss, metrics=[m for m in metrics if m != "loss"])
   
        print(model.summary())
    
    history = model.fit(X_train, y_train,
                        epochs=epochs, validation_data=(X_test, y_test), shuffle=True,
                        batch_size = batch_size
                        )
    
    for m, b in zip(metrics, plot_metrics):
        if b:
            plt.figure()
            plt.plot(history.history[m])
            plt.plot(history.history['val_'+m])
            plt.title(m+" Plot")
            plt.ylabel(m)
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()  
    
    return model, history

#--- Model Test
def model_buffer_test(model, X_test, y_test, predictions = 10, buffer_size = 30):
    
    """   
    Description:
        Function that takes an amount of predections buffer_size and make this predictions. 
        then it makes the mode, and compares it with the labels mode.
      
    Inputs:
        - model (tensorflow.keras.Sequential): trained model
        - X_test (): features to use to test the model
        - y_test (): labels to use to test the model
        - predictions (int): number of predictions to make
        - buffer_size (int): size of the buffer who contains the features and labels to make predictions  
        
    Output:
        - None.
    """
    correct = 0
    
    for i in range(predictions):
        n = np.random.randint(0, X_test.shape[0] - predictions) #we substract predictions so that the random index does not overpass the FV amount.
        pred_matrix = np.zeros((buffer_size, y_test.shape[1])) #create a matrix of zeros of size (buffer, classes).
        label_matrix = np.zeros((buffer_size, y_test.shape[1])) 
        for j in range(buffer_size):
            pred_matrix[j, :] = model.predict(X_test[n+j:n+j+1, :])
            label_matrix[j, :] = y_test[n+j:n+j+1, :]
        
        print("Prediction no. ", i+1)
        #print(pred_matrix)
        #print(label_matrix)
        #print(np.argmax(pred_matrix, axis = 1))
        #print(np.argmax(label_matrix, axis = 1))
        pred = stats.mode(np.argmax(pred_matrix, axis = 1))
        label = stats.mode(np.argmax(label_matrix, axis = 1))
    
        print("Prediction: \n", pred)
        print("Label: \n", label)
        if pred[0][0] == label[0][0]:
            print("Correcto!!\n\n-\n\n")
            correct += 1
        else:
            print("Incorrecto!!\n\n-\n\n")  
    
    print(str(correct) + " out of " + str(predictions) + " predictions were correct.")
    print("That is " + str(100*correct/predictions) + "% Correct.")

#--- Weights Save

def save_model_weights(model, history, training_files, folder_dir):
    """
    
    Description:
    Save de weights of the model, exept those which came from Dropout layers.
    Also save a json object with the model characteristics and the loss and accuracy results.

    Inputs:
        - model (tensorflow.keras.Sequential): trained model
        - history (): object containing info from the training process.
        - folder_dir(str): directory of the folder to save the weights
        
    Output:
        - None.
        
    """
    
    n = len(model.layers) # total number of layers
    
    li = 0 # Dense layers counter
    files = {} #creating an empty dictionary 
    for i in range(n):
        if ("Dropout" in str(type(model.layers[i])) ): #Dropout layers are only useful during training and it's weights are no longer needed in the OpenBCI model
            pass # discard, nothing to save
        else:
            wi = model.layers[i].get_weights()[0] #this should return an array with the weights of the i-eth layer
            bi = model.layers[i].get_weights()[1] #this should return an array with the biases of the i-eth layer

            files['w'+str(li)] = wi #"appending" the values of the weights of each layer to the dictionary with the name wi for
            files['b'+str(li)] = bi #the weights of the i-eth layer and bi for the biases

            li += 1
    
    #since to create the JSON object we need to get info on the model, we use it to create the folder too
    model_dict_tojson, dirName = save_model_json(history, folder_dir, training_files, model = model)
    
    # Save weights and bias in separate files        
    for f in files.keys():                                      #the .keys method return a dict_keys object with contains the keys (the names) of each element inside the dict
        np.savetxt(dirName + "/" + f + '.csv', files[f], delimiter=',') #since those keys are wi and bi for the weights and biases of the i-eth layer 
        print(files[f].shape)                                     #a csv for each array is saved
        
    model.save(dirName + "/" + "SavedModel")
    
# -- Description of the Model Saved in JSON file

def save_model_json(history, folder_dir, training_files, **kwargs): #kwargs: model to get the layers from or layers_list with the layers numbers
    
    """
    
    Description:
    Take the layer info from the model and save the json 
    In **kwargs you could pass the model to get the layers from, or a layers_list with the layers parameters.
    Also save the training model date.

    Inputs:
        - history (): object containing info from the training process.
        - folder_dir (str): directory of the folder to save the weights
        - training_files (list of str): list containing the names of the files used to train this model.
        - **kwargs:
            > model(tf.keras.model): the model to get the layers from
            > layers_list(list): list with the number of units in each layer.

    Output:
        - current_model_dict (dict): dictionary to save th JSON
            a."acc"(float): accuracy of the model
            b."val_acc"(float): validation accuracy of the model
            c. "loss"(float): loss of the model
            d. "val_loss"(float): validation loss
            e. "input_size"(int): input size
            f. "hidden_layers_no"(int): hidden layers amount
            g. "hidden_size"(int): hidden layers unit amount
            h. "output_size"(int): output size
            i. "train_date"(string): date of training
        - dirName (str): directory for saving model JSON 
        
    """
    #First get model units numbers through the model itself or a list
    if "model" in kwargs.keys():
        model = kwargs["model"]
        input_size = model.layers[0].input_shape[-1]
        hidden_list = []
        for lay in model.layers[:-1]:
            if ("Dropout" in str(type(lay))):
                pass
            else:
                hidden_list.append(lay.output_shape[-1])
        output_size = model.layers[-1].output_shape[-1]
        
    elif "layers_list" in kwargs.keys():
        layers_list = kwargs["layers_list"]
        input_size = layers_list[0]
        hidden_list = layers_list[1:-1]
        output_size = layers_list[-1]
        
    else:
        raise ValueError("No object to get layers from please input model or layers_list")
    
    #Then get when the model was trained and the results
    train_date = str(datetime.datetime.now()).split(" ")[0]
    folder_name = str(input_size) + "-" + str(hidden_list) + "-" + str(output_size) + " " + train_date
    
    current_model_dict = {"acc": float(history.history['accuracy'][-1]),
                        "val_acc": float(history.history['val_accuracy'][-1]),
                        "loss" : float(history.history['loss'][-1]),
                        "val_loss" : float(history.history['val_loss'][-1]),
                        "input_size" : input_size,
                        "hidden_layers_no":len(hidden_list),
                        "hidden_size": hidden_list,
                        "output_size": output_size,
                        "training_files": training_files,
                        "train_date": train_date}
    
    #Reading other models present in the file to compare,
    #if the model is better than others it's name will be changed
    #adding "Best" to it and removing it from the latest best.
    folder_name = read_model_json(current_model_dict, folder_dir, folder_name, only_best = True)
    
    # Create a directory for saving model JSON and return it to save_weights to save the model weights
    dirName = folder_dir + "/" + folder_name
    print(dirName)
    
    try:
        os.mkdir(dirName) #this function creates a new directory, and since it's embedded in the try function, if the file already exists it raises an error and data is directly saved there
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        pass
    
    #Saving the JSON file
    with open(dirName + "/model_info.json", "w") as output:
        json.dump(current_model_dict, output)
        
    return current_model_dict, dirName

# Read other JSON files with model descriptions and compare to current
# if only_best = True then it will only compare the model to the ones with "Best" name

def read_model_json(current_model_dict, folder_dir, folder_name, only_best = True):
    """
        
    Description:
        Read inside the folder where the model is saved, and comparee the structure. If it's the same, 
        then reads the accuracy and loss and write "Best" in the folder name in case that the current 
        model has better numbers.

    Inputs:
        - current_model_dict (dict): dictionary to save the JSON
            a."acc"(float): accuracy of the model
            b."val_acc"(float): validation accuracy of the model
            c. "loss"(float): loss of the model
            d. "val_loss"(float): validation loss
            e. "input_size"(int): input size
            f. "hidden_layers_no"(int): hidden layers amount
            g. "hidden_size"(int): hidden layers unit amount
            h. "output_size"(int): output size
            i. "train_date"(string): date of training
        - forlder_dir (str): directory of the folder to save the weights
        - folder_name (str): name of the folder containing the actual JSON
        - only_best (bool): boolean to determinate if only the folders that have "Best" will be compared
    
    Output:
        - folder_name (str): name of the folder, with or without the "Best" depending on the case.
        
    """
    nothing_tocompare = True #this boolean is used if there is no other model with the same parameters to compare to, then the current one will be the best in its category
    for elem in os.listdir(folder_dir): #run through folders of models in the file
        if "Best" in elem or not only_best: #taking only the ones that have "Best" on them, if only_best = True, if not we take all
            with open(folder_dir + "/" + elem + "/model_info.json",) as f:
                dict_fromjson = json.load(f) #load JSON into dict
                f.close() #close the file, if not we wont be able to change folder names because files inside them will be opened
                
                if (current_model_dict["input_size"] == dict_fromjson["input_size"]) and (current_model_dict["hidden_size"] == dict_fromjson["hidden_size"]) and (current_model_dict["output_size"] == dict_fromjson["output_size"]):
                    nothing_tocompare = False #if there is at least one similar model then we have something to compare to
                    if model_comparison(current_model_dict, dict_fromjson): #run model_comparison
                        print("The model is better than the latest best, so it will replace it as the best one.")
                        basics.remove_from_foldername(folder_dir + "/" + elem, "Best") #if its true we remove "Best" from the folder of the latest best
                        folder_name = "Best " + folder_name #and add "Best" to the current model folder_name
                        break #if we found the one that our model is best to, then there's no reason to keep searching
                    else:
                        print("The model is not better than the latest best, it will still be saved as itself.")
                        
    if nothing_tocompare: #if there is no other model to compare to or if the ones that exist don't have the same architecture, then our current model is the best on its category
        folder_name = "Best " + folder_name
        print("The model is the first of it's category, it will be saved as best.")
    
    return folder_name

# Model Comparison Score

def model_comparison(current_model_dict, model_tocompare, keys_tocompare = ["acc", "val_acc", "loss", "val_loss"]):
    """
    
    Description:
        Compare the current model with another saved model in JSON format and return True if the current model is better,
        or False in the other case.
        This function cheks keys_to_compare in each model dictionary to determinate which one is the best
        If the keys_to_compare are from validation (val_acc, val_loss), they will be weighted by 2

    Inputs:
        - current_model_dict (dict): actual dictionary to save the JSON and compare
            a."acc": (float) accuracy of the model
            b."val_acc": (float) validation accuracy of the model
            c. "loss" : (float) loss of the model
            d. "val_loss" : (float) validation loss
            e. "input_size" : input_size
            f. "hidden_layers_no":len(hidden_list)
            g. "hidden_size": hidden_list
            h. "output_size": output_size
            i. "train_date": date
        - model_tocompare (dict): JSON dictionary to be compared with the current model
        - keys_tocompare (list): keys from dictionary to compare both models
            a. acc: accuracy
            b. val_acc: validation accuracy
            c. loss
            d. val_loss: validation loss
    
    Output:
        - is_best (bool): boolean that describes if the model is the beest or not
        
    """  
    
    score = 0
    for key in keys_tocompare:
        if "loss" in key: #loss is better when is lowest, so we make exceptions in the scoring
            delta = model_tocompare[key] - current_model_dict[key]
        else:
            delta = current_model_dict[key] - model_tocompare[key]
            
        if "val" in key: #Since validation accuracy and loss are more important for us because they represent the model generalization power, we give them more importance
            score += 2*delta
        else:
            score += delta
    
    if score >= 0:
        is_best = True
        
    else:
        is_best = False
        
    return is_best