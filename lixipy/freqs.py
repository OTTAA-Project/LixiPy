if __name__ == "__main__":
    import basics
    import ops
    import times
    import intels    
else:
    from lixipy import basics
    from lixipy import ops
    from lixipy import times
    from lixipy import intels

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn

#--- FFT Calculation

def fft_calc(windowed_signal, sample_rate, norm = "basic", filter_window = None):
    """
    Description:
        Function that calculates the fft using numpy
        If filter_window = np.array, then filter_window.shape must be the same as N (windowed_signal.shape)
       
    Inputs:
        - windowed_signal(np.array): windowed signal
        - sample_rate(int): sample frequency
        - norm(str): type of normalization to use
        - filter_window(str or np.ndarray): filter window to apply 
       
    Outputs:
        - fft_freq(numpy.ndarray): fft frequencies
        - fft_values(numpy.ndarray): fft values
        
    """
    N = int(windowed_signal.shape[-1])
    
    if filter_window == None:
      windowed_signal = windowed_signal
    elif type(filter_window) == str:
      filter_window_fromscipy = sgn.get_window(filter_window, N)
      windowed_signal = windowed_signal*filter_window_fromscipy
    elif type(filter_window) == np.ndarray:
      assert filter_window.shape[0] == N
      windowed_signal = windowed_signal*filter_window 
    else:
      raise ValueError("Incorrect argument passed for filter_window, see docs for more info.")    
    
    if norm == "ortho":
      norm_factor = 1/np.sqrt(N)
    elif norm == "basic":
      norm_factor = 2/N #https://www.mathworks.com/help/matlab/ref/fft.html actually the extremes (DC and Nyquist) should only be multiplied by 2
    else:
      norm_factor = 1 #replace by RaiseError
  
    fft_values = abs(np.fft.rfft(windowed_signal, axis = -1)*norm_factor)
    fft_freq = np.fft.rfftfreq(n = N, d = 1/sample_rate)
  
    return fft_freq, fft_values

#--- Feature Vector Generator

def feature_vector_gen(fft_freq, fft_values, interest_freqs, neighbour_or_interval = "neighbour",
                       include_harmonics = True, apply_SNR = False, same_bw_forSNR = True, bw_forSNR = 1.0,
                       interest_bw = 1.0, max_bins = 5, #config for neighbour method
                       include_extremes = True): #config for interval method
    """
    Description:
        General function where you chose if the feature vector is generated using an interval around the main freq or neighbours.
        If neighbour_or_interval = "neighbour", a list of freq must be given on interest_freqs. This list will be as long as frequencies of interest exist 
        If neighbour_or_interval = "interval", a tuple of minumum and maximum values must be given on interest_freqs. It will be a tuple of size 0 to 2.
        If apply_SNR = True, then feature vector values are divided by the mean of the values of the frequencies that are not included inside the fv.
    
    Inputs:
        - fft_freq(np.ndarray): fft frequencies
        - fft_values(np.ndarray): fft values
        - interest_freqs(list or tuple): contain the interest frequencies on a list or a tuple with the minimun and maximun values. Depending on the case
        - neighbour_or_interval(str): Define if the feature vector is generated usin an interval or neighbour. One of this two options must be choosed
        - include_harmonics(bool): Boolean to choose if onclude harmonics or not. 
        - apply_SNR(bool): Boolean to choose if apply SNR or not. 
        - same_bw_forSNR(bool): 
        - bw_forSNR(float):
        - interest_bw(float): bandwidth to configure neighbour method
        - max_bins(int): maximum number of bins, to configure neighbour method
        - include_extremes(bool): Choose if include extremes or not if neighbour_orinterval = "interval"
            
    Outputs:
        - feature_vector_bins(np.array): feature vector bins
        - feature_vector(np.array): feature vector values
    """
    assert neighbour_or_interval == "neighbour" or neighbour_or_interval == "interval", "The choice for Interval or Neighbour method isn't compatible"
    
    if neighbour_or_interval == "neighbour":
        assert type(interest_freqs) == list, "For Neighbour method, a list must be given on interest_freqs"
        feature_vector_bins, feature_vector = feature_vector_gen_neighbour(fft_freq, fft_values, interest_freqs,
                                                                           include_harmonics, apply_SNR, same_bw_forSNR, bw_forSNR,
                                                                           interest_bw, max_bins)
        
    elif neighbour_or_interval == "interval": #shouldn't be necessary since we have the assertion above, but..
        assert type(interest_freqs) == tuple, "For Interval method, a tuple must be given on interest_freqs with minimum and maximum values"
        feature_vector_bins, feature_vector = feature_vector_gen_interval(fft_freq, fft_values, interest_freqs,
                                                                          apply_SNR, include_extremes)
        
    return feature_vector_bins, feature_vector
        
def feature_vector_gen_interval(fft_freq, fft_values, interest_freqs,
                                apply_SNR = False, include_extremes = True):
    """
    Description:
        Generate a mask with the interval freqs and apply it to fft_values anda fft_freq to generate feature_vector anda feature_vector_bins respectively
        If interest_freqs = None, then that extreme of the interval goes to max or min accordingly
        & is a bitwise AND operation.
        ~ is a bitwise NOT operation.
        If apply_SNR = True, then feature vector values are divided by the mean of the values of the frequencies that are not included inside the fv.
        
    Inputs:
        - fft_freq(np.ndarray): fft frequencies
        - fft_values(np.ndarray): fft values
        - interest_freqs(tuple): contain a tuple with the minimun and maximun values. 
        - apply_SNR(bool): Boolean to choose if apply SNR or not. 
        - include_extremes(bool): Choose if include extremes or not.
            
    Outputs:
        - feature_vector_bins(np.array): feature vector bins
        - feature_vector(np.array): feature vector values
        
    """
    #if a value in interest_freq is None, then that extreme of the interval goes to max or min accordingly
    if (interest_freqs[0] is None) and (interest_freqs[1] is None):
        interval_min = fft_freq[0]
        interval_max = fft_freq[-1]
    elif interest_freqs[0] is None:
        interval_min = fft_freq[0]
        interval_max = interest_freqs[1]
    elif interest_freqs[1] is None:
        interval_min = interest_freqs[0]
        interval_max = fft_freq[-1]
    else:
        interval_min = interest_freqs[0]
        interval_max = interest_freqs[1]
        
    if include_extremes:
        mask = (fft_freq >= interval_min) & (fft_freq <= interval_max) #& is a bitwise AND operation, it doesn't raise arror like and operator, see: https://www.geeksforgeeks.org/difference-between-and-and-in-python/
    else:
        mask = (fft_freq > interval_min) & (fft_freq < interval_max)
    
    if len(fft_values.shape) == 1:
      feature_vector = fft_values[mask]
      feature_vector_bins = fft_freq[mask]
      
      noise_vector = fft_values[~mask] #~ is a bitwise NOT operation, it doesn't raise arror like not operator, see: https://www.geeksforgeeks.org/python-bitwise-operators/
    elif len(fft_values.shape) == 2:
      feature_vector = fft_values[:, mask]
      feature_vector_bins = fft_freq[mask]

      noise_vector = fft_values[:, ~mask]
    else:
      print("Invalid fft_values shape.")
      raise ValueError

    if apply_SNR:
        rate = np.mean(np.array(noise_vector), axis = -1)
        if len(noise_vector.shape) == 2:
          rate = rate.reshape((rate.shape[0], 1))
    else:
        rate = 1.0
        
    feature_vector = np.array(feature_vector) / rate #If any dimension coincide to be broadcasted togheter, the element-wise operation will go through that dimension, in this case doing matrix_row/mean_of_that_row
    feature_vector_bins = np.array(feature_vector_bins)
    
    return feature_vector_bins, feature_vector

def feature_vector_gen_neighbour(fft_freq, fft_values, interest_freqs, 
                                  include_harmonics = True, apply_SNR = False, same_bw_forSNR = True, bw_forSNR = 1.0,
                                  interest_bw = 1, max_bins = 40):
    """
    Description:
        Calculate the feature vector bins and values
        If apply_SNR = True, then feature vector values are divided by the mean of the values of the frequencies that are not included inside the fv.
        If same_bw_forSNR = False, then you have to determinate the particular bw in bw_forSNR(float).
        If include_harmonics = True, then the harmonics will also be used to generate an interval around the freq and the harmonics.
        Each neighbour will be generated with an especific bandwidth and maximun number of bins. If the bw is too long, and the 
        max_bins too small, maybe not all the bw will be used. And vice versa.
            
    Inputs:
        - fft_freq(np.ndarray): fft frequencies
        - fft_values(np.ndarray): fft values
        - interest_freqs(list): contain the interest frequencies on a list 
        - include_harmonics(bool): Boolean to choose if onclude harmonics or not. 
        - apply_SNR(bool): Boolean to choose if apply SNR or not. 
        - same_bw_forSNR(bool): To choose if use the same bandwidth for the SNR or not.
        - bw_forSNR(float): To choose a particular bandwidth for SNR calc.
        - interest_bw(float): bandwidth
        - max_bins(int): maximum number of bins
                
    Outputs:
        - feature_vector_bins(np.array): feature vector bins
        - feature_vector(np.array): feature vector values
        
    """
    N = fft_freq.shape[-1]
    mask = []
    maskSNR = []
    
    filtered_freqs = [freqs for freqs in interest_freqs if freqs is not None]
    bins_sum = 0
    harm_bins_sum = 0
    for i in range(N):
        fft_bin = fft_freq[i] 
        if any([fft_bin <= (particular_freqs + interest_bw) and fft_bin >= (particular_freqs - interest_bw) for particular_freqs in filtered_freqs]) and bins_sum < max_bins:
            mask.append(True)
            #bins_sum += 1 REDO: used to loop on each freq so max_bins worked for each frequency, now its global, change or delete
        elif any([fft_bin <= (2*particular_freqs + interest_bw) and fft_bin >= (2*particular_freqs - interest_bw) for particular_freqs in filtered_freqs]) and harm_bins_sum < max_bins and include_harmonics: 
            mask.append(True)
            #harm_bins_sum += 1 REDO: used to loop on each freq so max_bins worked for each frequency, now its global, change or delete
        else:
            mask.append(False)
        
        if apply_SNR and not same_bw_forSNR:
            if any([fft_bin >= (particular_freqs + bw_forSNR) or fft_bin <= (particular_freqs - bw_forSNR) for particular_freqs in filtered_freqs]) or (any([fft_bin >= (2*particular_freqs + bw_forSNR) or fft_bin <= (2*particular_freqs - bw_forSNR) for particular_freqs in filtered_freqs]) and include_harmonics):
                maskSNR.append(True)
            else:
                maskSNR.append(False)
    
    mask = np.array(mask)
    maskSNR = np.array(maskSNR)

    if len(fft_values.shape) == 1:
        feature_vector = np.array(fft_values[mask])
        feature_vector_bins = np.array(fft_freq[mask])

        if same_bw_forSNR:
            noise_vector = np.array(fft_values[~mask])
        else:
            noise_vector = np.array(fft_values[maskSNR])
    elif len(fft_values.shape) == 2:
        feature_vector = np.array(fft_values[:, mask])
        feature_vector_bins = np.array(fft_freq[mask])

        if same_bw_forSNR:
            noise_vector = np.array(fft_values[:, ~mask])
        else:
            noise_vector = np.array(fft_values[:, maskSNR])
    else:
        print("Invalid fft_values shape.")
        raise ValueError
        
    if apply_SNR:
        rate = np.mean(noise_vector, axis = -1)
        if len(fft_values.shape) == 2:
          rate = rate.reshape((rate.shape[0], 1))
    else:
        rate = 1.0
        
    feature_vector = np.array(feature_vector) / rate
    feature_vector_bins = np.array(feature_vector_bins)
    
    return feature_vector_bins, feature_vector

#--  Feature Matrix Generator

def safe_feature_matrix_gen(signal, sample_rate, labels, startpoint_timestamps, interest_freqs, neighbour_or_interval,
                            stim_size, stim_delay, stride, filter_window = None, window_size = 512, 
                            norm = "basic", max_bins = 5, interest_bw = 1.0, include_harmonics = True, 
                            apply_SNR = False, same_bw_forSNR = True, bw_forSNR = 1.0, 
                            include_extremes = True, print_data_loss = True,
                            plot_average = False):
    """
    Description:
        This method is slower than the fast_feature_matrix_gen, but works perfectly with a stride that is not an even divisor of the window size.
    
    Inputs:
        See doc for feature_matrix_gen. Only difference lies in that every time interval or vector size is only passed in sample measure, use feature_matrix_gen with method="safe" to pass it in seconds.
        
    Outputs:
        - feature_vector_bins(np.array): feature vector bins
        - fv_matrix(np.array): matrix full of feature vector
        - label_matrix(np.array): matrix with all the labels.
        
    """
    sample_loss = 0
    fv_matrix = None
    #FFT Calculation
    if plot_average:
      plt.figure()
    for i in range(len(labels)):
      l = labels[i]
      onehot = np.zeros(len(labels), dtype = float)
      onehot[i] = 1.0
      
      fft_average_matrix  = None
  
      for t in startpoint_timestamps[i]:
        j = 0 #this represents the stride on each particular window
        while True:
          start_sample = t + j + stim_delay
          end_sample = t + j + stim_delay + window_size
  
          if end_sample > (stim_size+t): #in case stride is not a divisor of the stimulation interval
            if start_sample < (stim_size+t):
              sample_loss += stim_size + t - (end_sample - stride) #to count the amount of samples that did not get transformed. a method to overcome this loss is to take for that window only end_sample = stim_size and start_sample = stim_size-N
            break
          elif end_sample > (signal.shape[-1]): #in case signal gets to the end and a new interval doesn't fit correctly
            if start_sample < (signal.shape[-1]):
              sample_loss += signal.shape[-1] - (end_sample - stride) #to count the amount of samples that did not get transformed. a method to overcome this loss is to take for that window only end_sample = stim_size and start_sample = stim_size-N
            break
          elif end_sample == (stim_size+t): #in case stride is a divisor of the stimulation interval
            break
          elif end_sample == (signal.shape[-1]): #in case signal gets to the end
            break
          else:
            fft_freq, fft_val = fft_calc(signal[start_sample:end_sample], sample_rate, norm, filter_window)
            fv_bins, fv = feature_vector_gen(fft_freq, fft_val, interest_freqs, neighbour_or_interval, include_harmonics, apply_SNR, same_bw_forSNR, bw_forSNR, interest_bw, max_bins, include_extremes)
          
          if fv_matrix is None:
            fv_matrix = np.array([fv])
            label_matrix = np.array([onehot])
                  
          else:
            fv_matrix = np.append(fv_matrix, [fv], axis = 0)
            label_matrix = np.append(label_matrix, [onehot], axis = 0)
  
          if plot_average:
              if fft_average_matrix is None:
                  fft_average_matrix = np.array([fft_val])
              else:
                  fft_average_matrix = np.append(fft_average_matrix, [fft_val], axis = 0)
          
          j += stride
                            
      if plot_average and (fft_average_matrix is not None):
          plt.plot(fft_freq, np.mean(fft_average_matrix, axis = 0))
          plt.legend(["label " + str(lab) for lab in labels if startpoint_timestamps[labels.index(lab)] != []]) 
    
    if print_data_loss:
      print("The selected parameters led to the loss of ", str(sample_loss), " samples.")
  
    return fv_bins, fv_matrix, label_matrix

def fast_feature_matrix_gen(signal, sample_rate, labels, startpoint_timestamps, interest_freqs, neighbour_or_interval,
                            stim_size = 0, stim_delay = 0, endpoint_timestamps = None, stride = 8, filter_window = None, window_size = 512, 
                            norm = "basic", max_bins = 5, interest_bw = 1.0, include_harmonics = True, 
                            apply_SNR = False, same_bw_forSNR = True, bw_forSNR = 1.0, 
                            include_extremes = True, print_data_loss = True,
                            plot_average = False):
    """
    Description:
        This method is faster than the safe_feature_matrix_gen, but may alter data with a stride that is not an even divisor of the window size.
    
    Inputs:
        See doc for feature_matrix_gen. Only difference lies in that every time interval or vector size is only passed in sample measure, use feature_matrix_gen with method="fast" to pass it in seconds.
        
    Outputs:
        - feature_vector_bins(np.array): feature vector bins
        - fv_matrix(np.array): matrix full of feature vector
        - label_matrix(np.array): matrix with all the labels.
        
    """
    time_fv_matrix, onehot_labels_matrix = times.fast_feature_matrix_gen(signal, labels, startpoint_timestamps, 
                                                                        stim_size, stim_delay, stride, endpoint_timestamps,
                                                                        window_size, print_data_loss)

    fv_bins, fv_matrix = feature_matrix_from_times(time_fv_matrix, onehot_labels_matrix, sample_rate,
                                                  labels, interest_freqs, interest_bw, max_bins,
                                                  neighbour_or_interval, include_harmonics, include_extremes,
                                                  norm, filter_window,
                                                  apply_SNR, same_bw_forSNR, bw_forSNR,
                                                  plot_average)

    return fv_bins, fv_matrix, onehot_labels_matrix
    
def feature_matrix_from_times(time_fv_matrix, onehot_labels_matrix, sample_rate,
                              labels, interest_freqs, interest_bw = 1.0, max_bins = 40,
                              neighbour_or_interval = "neighbour", include_harmonics = True, include_extremes = True,
                              norm = "basic", filter_window = None,
                              apply_SNR = False, same_bw_forSNR = True, bw_forSNR = 1.0,
                              plot_average = False):

    fft_bins, fft_matrix = fft_calc(time_fv_matrix, sample_rate, norm, filter_window)
    
    if plot_average:
      plt.figure()
      for i in range(len(labels)):
        plt.plot(fft_bins, fft_matrix[onehot_labels_matrix[:, i] == 1].mean(axis = 0), label="Label " + str(labels[i]))
      plt.legend()
        
    fv_bins, fv_matrix = feature_vector_gen(fft_bins, fft_matrix, interest_freqs, neighbour_or_interval, include_harmonics, apply_SNR, same_bw_forSNR, bw_forSNR, interest_bw, max_bins, include_extremes)

    return fv_bins, fv_matrix

def feature_matrix_gen(signal, sample_rate, labels, startpoint_timestamps, interest_freqs, neighbour_or_interval = "neighbour",
                        method = "fast", channels = "all", channels_return = "concat",
                        stimulation_time = 10, stimulation_time_inseconds = True, stimulation_delay = 0, stimulation_delay_inseconds = True,
                        endpoint_timestamps = None,
                        stride = 8, stride_inseconds = False, filter_window = None,
                        window_size = 512, window_size_inseconds = False, 
                        norm = "basic", max_bins = 5, interest_bw = 1.0, include_harmonics = True, 
                        apply_SNR = False, same_bw_forSNR = True, bw_forSNR = 1.0, 
                        include_extremes = True, print_data_loss = True,
                        plot_average = False):

  """
    Description:
        Generates a Feature Vector Matrix in the frequency spectrum of the signal by obtaining the FFT amplitudes at the desired bins and appending a new example on each line.
        If neighbour_or_interval = "neighbour", a list of freq must be given on interest_freqs. This list will be as long as frequencies of interest exist 
        If neighbour_or_interval = "interval", a tuple of minumum and maximum values must be given on interest_freqs. It will be a tuple of size 0 to 2.
        If apply_SNR = True, then feature vector values are divided by the mean of the values of the frequencies that are not included inside the fv.
        Stimulation time could be writte in seconds or samples.
    
    Inputs:
        - signal(txt): signal to generate the feature vector
        - sample_rate(int): sampling frequency the signal was obtained in.
        - labels(list): labels of the feature vector
        - startpoint_timestamps(list):
        - interest_freqs(list or tuple): contain the interest frequencies on a list or a tuple with the minimun and maximun values. Depending on the case
        - neighbour_or_interval(str): Define if the feature vector is generated usin an interval or neighbour. One of this two options must be choose
        - stimulation_time(int or float): stimulation time to select an interval. It could be in second or samples
        - stimulation_time_inseconds(bool): to specify whether stimulation_time is written in seconds or in samples
        - stimulation_delay(int or float): to select a delay from timestartpoint_timestamps
        - stimulation_delay_inseconds(bool): to specify whether stimulation_delay is written in seconds or in samples
        - stride(int): stride of the window in each step
        - stride_inseconds(bool): to specify whether stride is written in seconds or in samples
        - filter_window(str): To select an especiifc window 
        - window_size(int): window size
        - window_size_inseconds(bool): to specify whether window_size is written in seconds or in samples
        - norm(str): normalization for the fft calculation
        - max_bins(int): maximum number of bins, to configure neighbour method
        - interest_bw(float): bandwidth to configure neighbour method
        - include_harmonics(bool): Boolean to choose if onclude harmonics or not to configure neighbour method. 
        - apply_SNR(bool): Boolean to choose if apply SNR or not. 
        - same_bw_forSNR(bool): To choose if use the same bandwidth for the SNR or not to configure neighbour method.
        - bw_forSNR(float): To choose a particular bandwidth for SNR calc to configure neighbour method.
        - include_extremes(bool): Choose if include extremes or not if neighbour_orinterval = "interval" 
        - plot_average(bool): to plot the feature vector average
        
    Outputs:
        - feature_vector_bins(np.array): feature vector bins
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
      temp = fast_feature_matrix_gen(signal[ch, :], sample_rate, labels, startpoint_timestamps, 
                                    interest_freqs, neighbour_or_interval, stim_size, stim_delay, endpoint_timestamps,
                                    stride, filter_window, N, norm, max_bins, interest_bw, include_harmonics, 
                                    apply_SNR, same_bw_forSNR, bw_forSNR, include_extremes, print_data_loss, plot_average
                                    )
      fv_matrix_list.append(temp[1])
    elif method == "safe":
      temp = safe_feature_matrix_gen(signal[ch, :], sample_rate, labels, startpoint_timestamps, 
                                    interest_freqs, neighbour_or_interval, stim_size, stim_delay, 
                                    stride, filter_window, N, norm, max_bins, interest_bw, include_harmonics, 
                                    apply_SNR, same_bw_forSNR, bw_forSNR, include_extremes, print_data_loss, plot_average
                                    )
      fv_matrix_list.append(temp[1])
    else:
      print("Invalid method type for feature matrix generation.")
      raise KeyboardInterrupt
  
  if plot_average:
    plt.show()
  fv_bins = temp[0] #for all channels the bins chosen and label matrix are the same
  label_matrix = temp[2]
  
  if channels_return == "concat":
    return fv_bins, np.concatenate(fv_matrix_list, axis = 1), label_matrix
  else:
    return fv_bins, fv_matrix_list, label_matrix

def command_simulation(signal_array = None, labels_array = None,
                      signal_pd = None, signal_column_name = ["Ch1", "Ch2", "Ch3", "Ch4"], labels_column_name = 'mlp_labels',
                      ):

    if signal_array is not None and labels_array is not None:
        signal = signal_array
        labels = labels_array
    elif signal_pd is not None:
        signal = signal_pd[signal_column_name].values
        labels = signal_pd[labels_column_name].values

    #THE IDEA IS TO USE THE FAST_FV_MATRIX_GEN for SIGNAL and the TIMES FAST_FEATURE_MATRIX_GEN for LABELS
    # with only one label = [0] and only one set of startpoint and endpoint timestamps = [[0]] & [[labels.shape[0]]]
    # then the gt can be chosen from the generated matrix of labels as the first sample, the last, or the average/median/etc
    # the pred is obtained from predicting on the SIGNAL MATRIX
    
    