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
import matplotlib as plt
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split

# --  Model Training

#MISSING: Check Definition and polish the training for already compiled models
def model_train(model, X, y, epochs, optimizer = 'adam', loss = 'categorical_crossentropy', compile_model = True,
                validation_set_mode = "split", val_part = 0.33, val_X = None, val_y = None, 
                batch_size = 32, random_state=42,
                plot_loss = True, plot_accuracy = True):
    
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
        - plot_loss (boolean): wether or not to plot loss.
        - plot_accuracy (boolean): wether or not to plot accuracy.
    
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
        
    if compile_model:
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
   
        print(model.summary())
    
    history = model.fit(X_train, y_train,
                        epochs=epochs, validation_data=(X_test, y_test), shuffle=True,
                        batch_size = batch_size
                        )
    
    
    # summarize history for loss
    if plot_loss:
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
    #summarize history for accuracy
    if plot_accuracy:
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()        
    
    return model, history

#--- Model Test
#MISSING: change to buffer test or smth
def model_test(model, X_test, y_test, predictions = 10, buffer_size = 30):
    
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