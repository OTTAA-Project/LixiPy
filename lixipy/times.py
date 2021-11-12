if __name__ == "__main__":
    import basics
    import ops
    import freqs
    import intels    
else:
    from lixipy import basics
    from lixipy import ops
    from lixipy import freqs
    from lixipy import intels

import numpy as np
import matplotlib.pyplot as plt

def fast_feature_matrix_gen(signal, labels, startpoint_timestamps,
                            stim_size = 0, stim_delay = 0, stride = 8,
                            endpoint_timestamps = None,
                            window_size = 512, 
                            print_data_loss = True
                            ):

    """
    Description:
        This method is faster than the safe_feature_matrix_gen, but may alter data with a stride that is not an even divisor of the window size.
    
    Inputs:
        See doc for feature_matrix_gen. Only difference lies in that every time interval or vector size is only passed in sample measure, use feature_matrix_gen with method="fast" to pass it in seconds.
        
    Outputs:
        - fv_matrix(np.array): matrix full of feature vector
        - label_matrix(np.array): matrix with all the labels.

    """

    total_dataloss = 0
    fv_matrix_list = []
    labels_matrix_list = []
    for i in range(len(labels)):
      each_label_matrix_size = 0
      for t in startpoint_timestamps[i]:
        if endpoint_timestamps is None:
          vector = signal[t+stim_delay:t+stim_size]
        else:
          vector = signal[t+stim_delay:endpoint_timestamps[i][startpoint_timestamps[i].index(t)]+1]
        end_list_for_dataloss = []
        vector_reshaped_list = []
        final_shape = 0
        for start in range(0, window_size, stride):
          quant = (vector.shape[0]-start)//window_size
          end = quant*window_size+start
          end_list_for_dataloss.append(end)
          vector_reshaped = vector[start:end].reshape(quant, window_size)
          #print("Vector reshaped:\n", vector_reshaped)
          final_shape += vector_reshaped.shape[0]
          vector_reshaped_list.append(vector_reshaped.copy())
        #print("dataloss: ",  vector.shape[0] - np.array(end_list_for_dataloss).max())
        total_dataloss += vector.shape[0] - np.array(end_list_for_dataloss).max()
        final_reshaped_vector = np.empty((final_shape, window_size))
        temp = len(vector_reshaped_list)
        for n, v in enumerate(vector_reshaped_list):
          final_reshaped_vector[n:n+v.shape[0]*temp:temp,:] = v
        #print(final_reshaped_vector.shape)
        fv_matrix_list.append(final_reshaped_vector.copy())
        each_label_matrix_size += final_reshaped_vector.shape[0]
      temp_labels = np.zeros((each_label_matrix_size, len(labels)))
      temp_labels[:, i] = 1.0
      labels_matrix_list.append(temp_labels)

    fv_matrix = np.concatenate(fv_matrix_list, axis=0)
    onehot_labels_matrix = np.concatenate(labels_matrix_list, axis=0)
    
    if print_data_loss:
      print("The selected parameters led to the loss of ", str(total_dataloss), " samples.")

    return fv_matrix, onehot_labels_matrix

def feature_matrix_gen(signal, sample_rate, labels, startpoint_timestamps,
                        method = "fast", channels = "all", channels_return = "concat",
                        stimulation_time = 10, stimulation_time_inseconds = True, 
                        stimulation_delay = 0, stimulation_delay_inseconds = True,
                        endpoint_timestamps = None,
                        stride = 8, stride_inseconds = False,
                        window_size = 512, window_size_inseconds = False, 
                        print_data_loss = True,
                        plot_average = False):

  """
    Description:
        Generates a Feature Vector Matrix in the time spectrum of the signal by croppin the signal.
        All time durations can be written in seconds or samples.
    
    Inputs:
        - signal(txt): signal to generate the feature vector
        - sample_rate(int): sampling frequency the signal was obtained in.
        - labels(list): labels of the feature vector
        - startpoint_timestamps(list):
        - stimulation_time(int or float): stimulation time to select an interval. It could be in second or samples
        - stimulation_time_inseconds(bool): to specify whether stimulation_time is written in seconds or in samples
        - stimulation_delay(int or float): to select a delay from timestartpoint_timestamps
        - stimulation_delay_inseconds(bool): to specify whether stimulation_delay is written in seconds or in samples
        - stride(int): stride of the window in each step
        - stride_inseconds(bool): to specify whether stride is written in seconds or in samples
        - window_size(int): window size
        - window_size_inseconds(bool): to specify whether window_size is written in seconds or in samples
        - print_data_loss(bool): specifies the amount of samples that are lost for the window not coinciding exactly with the signal shape.
        - plot_average(bool): to plot the feature vector average
        
    Outputs:
        - fv_matrix(np.array): matrix full of feature vector
        - label_matrix(np.array): matrix with all the labels.
        
    """

  assert len(labels) == len(startpoint_timestamps), "label list and startpoint_timestamps list must be of equal length"

  #Window size
  if window_size_inseconds:
    N = window_size * sample_rate
  else:
    N = window_size
  
  #Stride size
  if stride_inseconds:
    stride = stride*sample_rate

  #Stimulation size
  if stimulation_time_inseconds:
    stim_size = stimulation_time*sample_rate
  else:
    stim_size = stimulation_time
    
  if stimulation_delay_inseconds:
    stim_delay = stimulation_delay*sample_rate
  else:
    stim_delay = stimulation_delay

  if channels == "all":
    channels = range(signal.shape[0])
  
  fv_matrix_list = []

  if method == "fast" and window_size%stride != 0:
    print("\n", "-"*100)
    print("WARNING: this method is intended for strides that are even divisors of the window_size, if that's not the case the results may be altered, but no noticeable error will be raised. If you'd like to use this parameters without the alterations better opt for the safe_feature_matrix_gen instead.")
    print("-"*100, "\n\n")
    if basics.yn_quest("\tDo you wish to continue with the fast method anyways? (If N will continue with the safe method)"):
      pass
    else:
      method = "safe"

  for ch in channels:
    if method == "fast":
      temp = fast_feature_matrix_gen(signal[ch, :], labels, startpoint_timestamps, 
                                    stim_size, stim_delay, stride, endpoint_timestamps,
                                    N, print_data_loss, plot_average
                                    )
      fv_matrix_list.append(temp[1])
    elif method == "safe":
      print("Safe method not fully developed yet")
      fv_matrix_list.append(temp[1])
    else:
      print("Invalid method type for feature matrix generation.")
      raise KeyboardInterrupt
  
  #if plot_average: #Not ready yet.
  #  plt.show()
  fv_bins = temp[0] #for all channels the bins chosen and label matrix are the same
  label_matrix = temp[2]
  
  if channels_return == "concat":
    return fv_bins, np.concatenate(fv_matrix_list, axis = 1), label_matrix
  else:
    return fv_bins, fv_matrix_list, label_matrix