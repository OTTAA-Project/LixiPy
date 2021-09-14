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

def fast_feature_matrix_gen(signal, labels, startpoint_timestamps,
                            stim_size = 0, stim_delay = 0, stride = 8,
                            endpoint_timestamps = None,
                            filter_window = None, window_size = 512, 
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