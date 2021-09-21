if __name__ == "__main__":
    import basics
    import times
    import freqs
    import intels    
else:
    from lixipy import basics
    from lixipy import times
    from lixipy import freqs
    from lixipy import intels

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import scipy.signal as sgn

#---Deleting the points that are not useful for training

def signal_cleaning (sgn_pd, label_col_name='mlp_labels', labels=[1, 2, 3, 4], #labels to keep
                     just_np = False): #wether to actually do the cleaning or just redo the index and export Numpy Array
    """
    Description:
        Deletes the points that are not useful for training, and transform the signal into a numpy.array
        In the chosen pandas, only keep the chosen labels 
        If just_np, then the signal is not cleaned, just transformed into a numpy.array
    
    Inputs:
        - sgn_pd(pd): pandas Dataframe
        - label_col_name(str): name of the label column
        - labels(list): list with the labels to conserve
        - just_np(bool): boolean to choose if only transforma the signal into a np.array
       
    Outputs:
        - signal(numpy.array): numpy signal
        - to_signal(pd): cleaned pandas Dataframe
        
    """    
    if (not just_np):
      df_calibration = sgn_pd
      
      sgn_pd = df_calibration.loc[df_calibration[label_col_name].isin(labels)]
    
    #Rebuilding the index tag, since many values were dropped in the function above
    sgn_pd.index = range(sgn_pd.shape[0]) 
    
    #Drop the colmns and generate the signal np array, transposing it so that it's easier to work later
    to_signal = sgn_pd.copy()
    signal = np.asarray(to_signal.drop(columns = ["x", "y", "z", "mlp_labels", "time", "rand"]))
    signal = signal.transpose()
    
    return signal, to_signal

#--- Preprocessing

def preprocess(signal, fs=200, f_notch=50, Q_notch=30, bp_order=5, bp_type='butter', bp_flo=15, bp_fhi=80):
    """
    Description:
        Apply filters to the signal such as, notch and bp and returns the fitered signal
       
    Inputs:
        - signal(np.array): numpy array signal to preprocess
        - fs(int): sample frequency
        - f_notch(float): notch filter frequency 
        - Q_notch(float): Quality factor that characterizes the notch filteer
        - bp_order(int): bandpass filter order
        - bp_type(str): type of bandpass filter
        - bp_flo(float): bandpass low frequency
        - bp_fhi(float): bandpass high frequency
       
    Outputs:
        - filt_signal(np.array): filtered signal
        
    """
    fn = fs/2
  
    #Detrend
    signal = sgn.detrend(signal, type='constant')
  
    #Notch Filter
    b, a = sgn.iirnotch(f_notch, Q_notch, fs)
    signal = sgn.filtfilt(b, a, signal)
  
    #Band-pass Filter
    b, a = sgn.iirfilter(bp_order, [bp_flo/fn, bp_fhi/fn], ftype=bp_type)
    filt_signal = sgn.filtfilt(b, a, signal)
  
    return filt_signal

#--- Timestamps Location

def interval_timestamps_loc(dataframe, label_column_name = 'mlp_labels', labels = [1, 2], 
                            export_dict = True, export_list = True, list_end = False):
    """
    Description:
        Find the index of the start and end of the intervals of labels in a column label_column_name inside dataframe.
        If export_dict, then returns a dictionary of dictionaries where the first keys will be the label
        the second keys will choose between startpoint or endpoint timestamps and the value will be a list with the timestamps.
        If export_list, then returns a list of lists of the startpoint timestamps if list_end is False, or endpoint_timestamps if list_end is True,
        in the same order they appear in the dictionary mentioned above order. 
        Take into consideration that, using the inclusive:exclusive paradigm of Python on interval declaration, the endpoint timestamps must be increased by 1
        to include the whole interval, since this are the last index of each labels intervals.
        
    Inputs:
        - dataframe(pd): pandas Dataframe
        - label_column_name(str): name of the label column
        - labels(list): list with the labels 
        - export_dict(bool): boolean to choose if export as a dictionary 
        - export_list(bool): boolean to choose if export as a list of lists
        - end_list(bool): boolean to choose wether to export the startpoint or endpoint timestamps in the export list.
       
    Outputs:
        - timestamps_dict(dict): timestamps dictionary
        - timestamps_list(list): timestamps list of lists
    """
    timestamps_dict = {}
    for l in labels:
        timestamps_dict["Label " + str(l)] = {"Startpoint":[], "Endpoint":[]}

    prev_tag = None
    for i in dataframe.index:
        tag = dataframe[label_column_name][i]
        if tag != prev_tag:
            if tag in labels:
                timestamps_dict["Label " + str(tag)]["Startpoint"].append(i)
            if i!=0 and prev_tag in labels:
                timestamps_dict["Label " + str(prev_tag)]["Endpoint"].append(i-1)
        prev_tag = tag

    if prev_tag in labels:
        timestamps_dict["Label " + str(prev_tag)]["Endpoint"].append(i)
    
    if list_end:
        timestamps_list = [element["Endpoint"] for element in timestamps_dict.values()]
    else:
        timestamps_list = [element["Startpoint"] for element in timestamps_dict.values()]
    
    if export_dict and export_list:
        return timestamps_dict, timestamps_list
    
    elif export_dict:
        return timestamps_dict
    
    elif export_list:
        return timestamps_list

def markers_timestamps_loc(dataframe, label_column_name = "mlp_labels", labels = [1,2],
                            export_dict = True, export_list = True, 
                            list_end = False, list_others=False, interval_duration=None):

    """
    Description:
        Find the index for marker labels in a column label_column_name inside dataframe.
        If and interval_duration is provided, then this markers can be interpreted as interval beginings,
        and return both the end of each interval marked and the begining and end of intervals in between.
        If export_dict, then returns a dictionary of dictionaries where the first keys will be the label
        the second keys will choose between startpoint or endpoint timestamps and the value will be a list with the timestamps.
        If export_list, then returns a list of lists of the startpoint timestamps if list_end is False, or endpoint_timestamps if list_end is True,
        in the same order they appear in the dictionary mentioned above order. 
        Take into consideration that, using the inclusive:exclusive paradigm of Python on interval declaration, the endpoint timestamps must be increased by 1
        to include the whole interval, since this are the last index of each labels intervals.
        
    Inputs:
        - dataframe(pd): pandas Dataframe
        - label_column_name(str): name of the label column
        - labels(list): list with the labels 
        - export_dict(bool): boolean to choose if export as a dictionary 
        - export_list(bool): boolean to choose if export as a list of lists
        - list_end(bool): boolean to choose wether to export the startpoint or endpoint timestamps in the export list. Needs providing an interval_duration.
        - list_other(bool) : boolean to choose wether to include the intervals in between the marked ones in the analysis. Needs providing an interval_duration.
        - interval_duration(int) : defined duration for virtual interval of analysis that start on the marker appearance and end after interval_duration samples.

    Outputs:
        - timestamps_dict(dict): timestamps dictionary
        - timestamps_list(list): timestamps list of lists 
    """

    marker_dict, marker_list = interval_timestamps_loc(dataframe, label_column_name, labels)

    if list_end or list_others or interval_duration is not None:
        if interval_duration is None or interval_duration <= 0:
            raise ValueError("For the method to obtain the endpoint timestamps and/or any timestamp of the intervals in between, you need to provide a greater than zero integer as the interval duration.")
    
        for k in marker_dict.keys():
            marker_dict[k]["Endpoint"] = [t + interval_duration for t in marker_dict[k]["Startpoint"]]

        if list_others:

            #to identify the in-between intervals we'll recreate the labels vector with an specific tag for those "other" intervals
            others_label = 0
            while others_label in labels:
                others_label += 1
            
            aux_labels = np.full(dataframe[label_column_name].values.shape[0], others_label)

            for k in marker_dict.keys():
                lab = int(k.split(" ")[-1])

                for start, end in zip(marker_dict[k]["Startpoint"], marker_dict[k]["Endpoint"]):
                    aux_labels[start:end+1] = np.full(end+1-start, lab)

            aux_labels_df = pd.DataFrame(aux_labels, columns=[label_column_name])
            others_marker_dict = interval_timestamps_loc(aux_labels_df, label_column_name, labels=[others_label],
                                                        export_list = False)

            marker_dict["Others"] = others_marker_dict["Label " + str(others_label)]
            
    if list_end:
        timestamps_list = [element["Endpoint"] for element in marker_dict.values()]
    else:
        timestamps_list = [element["Startpoint"] for element in marker_dict.values()]

    timestamps_dict = marker_dict

    if export_dict and export_list:
        return timestamps_dict, timestamps_list
    
    elif export_dict:
        return timestamps_dict
    
    elif export_list:
        return timestamps_list

def label_interval_correspondance(timestamps_dict, index, other_return_value = "other"):

    """
    Description:
        Method for finding which label a certain index corresponds to based on the timestamps_dict
        obtained through the interval_timestamps_loc or markers_timestamps_loc method.
        Although it'll still work, if the timestamps for the fill-in intervals is not provided the method will be
        slower, since it'll discard being in any of the labeled intervals first.
        If the dictionary is obtained through other mediums, this is the structure it should have:
        {
        "Label X":
            {
            "Startpoint": [list of integers of startpoints for label X],
            "Endpoint": [list of integers of startpoints for label X]
            },
        "Label Y":
            {...}
        "Others": (Optional, the startpoints and enpoints of the fill-in intervals between intervals of labels)
            {...}
        }

    Inputs:
        - timestamps_dict(dict): dictionary with the timestamps for startpoint and enpoints of intervals of labels, obtained
        through the any of the timestamps_loc methods, or at least of the from shown above.
        - index(int): index that shall be analysied correspondance for.
        - others_return_value(Any): value to be returned when the index falls into an "Other" category, corresponding to
        the fill-in intervals, or the method doesn't find an interval that index falls into (which should be comparative circumstances)

    Outputs:
        - label(Any): label the index corresponds to, or other_return_value if the index corresponds to the "Other" category
        or doesn't fit in any of the intervals passed.        
    """
    for k in timestamps_dict.keys():
        if k == "Others":
            lab = other_return_value
        else:
            lab = int(k.split(" ")[-1])

        for start, end in zip(timestamps_dict[k]["Startpoint"], timestamps_dict[k]["Endpoint"]):
            if start <= index <= end:
                return lab
        
    return other_return_value

#Index Finder
def find_index(array, value, find_alternative = True, ascending_descending_array = True):
    """   
    Description:
        Looks for value inside the array and return which index it is in.
        If find_alternative then when the requested value does not exist in array, the algorithm will return the index with the closest value
        Example:
            >>> array = [21.5, 21.9, 22.3, 22.7, 23.1]
            >>> value = 22
            >>> find_alternative = True
            >>> find_index(array, value, find_alternative)
            Out: 1

    Inputs:
        - array(np.array or array-like object): array to search through.
        - value(int or float): value to search for.
        - find_alternative(bool): boolean to choose to find the nearest value in case the requested does not exist in array.
        - ascending_descending_array(bool): boolean to tell the array is in ascending or descending value order, in which case the value can be found faster
    Outputs:
        - i(int): index where the value (or it's alternative) was found
    """
    last_dif = float('inf')
    min_dif = float('inf')
    min_dif_index = None
    for i in range(len(array)):
        dif = abs(array[i]-value)
        
        if dif == 0:
            return i
        
        if find_alternative:
            if ascending_descending_array:
                if last_dif<=dif:
                    return i-1
                else:
                    last_dif = dif
            else:
                if dif<min_dif:
                    min_dif=dif
                    min_dif_index=i

    return min_dif_index
        #MISSING: agregar aca el else para el find alternative, checkear si esto se puede usar en las funciones de fv gen

# Laplacian Optimization: Objective Function

def objective_f(fft, fft_bins, freq, bw_bins = 2, bw_neighbour = 2):
    """
    Description:
        Takes, the fft, fft_bins and freq and calculates an objective function where result = (low_neighbour_average + upp_neighbour_average)/2 - main_freq_average
        main_freq_aveerage is an average of the interest frequencies taken from freq - bw_bins to freq + bw_bins
        low_neighbour_average is an average of non-interest frequencies taken:
            - from 0 to freq - bw_bins if bw_neighbour = "all"
            - from freq - bw_neighbour - bw_bins to freq - bw_neighbour + bw_bins if bw_neighbour is an int
        upp_neighbour_average is an average of non-interest frequencies taken:
            - from freq + bw_bins to the highest frequency if bw_neighbour = "all"
            - from freq + bw_neighbour - bw_bins to freq + bw_neighbour + bw_bins if bw_neighbour is an int
        bw_bins would be in frequency samples, not values
        bw_neighbour would be in frequency values, not samples
        
    Inputs:
        - fft(np.array): fft
        - fft_bins(np.array): frequency bins
        - freq(float): frequency of interest
        - bw_bins(int): bandwidht that will be used to calculate the main of interest frequencies 
        - bw_neighbour(int or str): bandwidht that will be used to calculate the main of non-interest frequencies 
       
    Outputs:
        - result(float): resulting value of calculating the objective function
        
    """    
    #bw_bins would be in frequency samples, not values
    low_bw_bins = abs(int(bw_bins))
    high_bw_bins = low_bw_bins + 1 #Because of intervals being [inclusive, exclusive]
    #bw_neighbour would be in frequency values, not samples
    
    main_freq_bin = find_index(fft_bins, freq, True)
    main_freq_average = np.mean(fft[main_freq_bin-low_bw_bins:main_freq_bin+high_bw_bins])
    
    if bw_neighbour == "all":
        low_neighbour_average = np.mean(fft[0:main_freq_bin - bw_bins])
        upp_neighbour_average = np.mean(fft[main_freq_bin + bw_bins: -1])        
    else:
        low_neighbour_bin = find_index(fft_bins, freq - bw_neighbour, True)
        upp_neighbour_bin = find_index(fft_bins, freq + bw_neighbour, True)
        low_neighbour_average = np.mean(fft[low_neighbour_bin-low_bw_bins:low_neighbour_bin+high_bw_bins])
        upp_neighbour_average = np.mean(fft[upp_neighbour_bin-low_bw_bins:upp_neighbour_bin+high_bw_bins])
    
    #Objective function calculation
    result = (low_neighbour_average + upp_neighbour_average)/2 - main_freq_average

    return result

# Laplacian Application with Objective Function

def laplacian_app(x, *argf):
    """
    Description:
        Apply the laplacian, spacial filtering or combination, and calculate the objective function (see objective_f)
       
    Inputs:
        - x(list): list of weights to apply the laplacian calculation
        - argf(list of positional arguments):
            - (np.array): signal to apply the laplacian 
            - (int or float): frequency of interest to calculate the objective function
            - (int or float): smapling frequency of signal
            - (bool): wether to plot results (progress) or not
            - (int or float): bw_bins for the objective function
            - (int float or str): bw_neighbour for the objectivee function
            
    Outputs:
        - objective_result(float): objective function result
        
    """
    w = np.array(x)
    func = argf[0]
    freq = argf[1]
    sample_freq = argf[2]
    plot_bool = argf[3]
    bw_bins = argf[4]
    bw_neighbour = argf[5]
    
    func2 = func.copy()

    for i in range(func.shape[0]):
        func2[i] = w[i]*func[i]
  
    sum_func = np.sum(func2, axis = 0)
    fft_bins = np.fft.rfftfreq(sum_func.shape[0], d=1/sample_freq)
    fft_vals = abs(np.fft.rfft(sum_func))
  
    if plot_bool:
        plt.figure()#(figsize = (15,5))
        plt.title("Plot for Optimization for Freq: " + str(freq) + " Hz")
        plt.plot(fft_bins, fft_vals)
        plt.xlim(0, 2*freq+1) #so as to catch the harmonics

    objective_result = objective_f(fft_vals, fft_bins, freq, bw_bins, bw_neighbour)

    return objective_result

# Optimization
def optimization_lap(freq, signal, sample_freq = 200, initial_values = [0.25, 0.25, 0.25, 0.25], 
                     bw_bins = 2, bw_neighbour = 2, max_iter = 50, method = "L-BFGS-B", bounds = 5,
                     plot_progress = True):
    """
    Description:
       Optimize the laplacian values
       Takes a set of weights, apply the laplacian, calculates the objective function and minimize its results through changing the weights values
       The optimization is done through the chosen method
       
    Inputs:
        - freq(int or float): frequency of interest
        - signal(np.array): signal of interest to optimize laplacian for
        - sample_freq(int or float): smapling frequency of signal
        - initial_values(list): initial values to start the laplacian from
        - bw_bins(int or float): bw_bins for the objective function
        - bw_neighbour(int float or str): bw_neighbour for the objectivee function
        - max_iter(int): maximun amount of iterations to allow the optimization to work on
        - method(str): mothod to work the optimization with (see scipy.optimize.minimize)
        - bounds(int or floats): bouderies for the values of the weights
        - plot_progress(bool): wether to plot progress or not
       
    Outputs:
        - results(list): final weights obtained after apply the optimization
        
    """    
    assert type(signal) == np.ndarray
    assert len(initial_values) == signal.shape[0]
    
    bound_list = []
    for i in range(len(initial_values)):
        bound_list.append((-bounds, +bounds))
        
    laplacian_app(initial_values, signal, freq, sample_freq, plot_progress, bw_bins, bw_neighbour)

    results = minimize(laplacian_app, x0 = initial_values, 
                   args=(signal, freq, sample_freq, False, bw_bins, bw_neighbour), 
                   method = "L-BFGS-B",
                   bounds = bound_list,
                   options = {'maxiter': max_iter,
                              'disp': True
                              }
                   )

    laplacian_app(results.x, signal, freq, sample_freq, plot_progress, bw_bins, bw_neighbour)

    return results