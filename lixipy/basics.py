if __name__ == "__main__":
    import ops
    import times
    import freqs
    import intels    
else:
    from lixipy import ops
    from lixipy import times
    from lixipy import freqs
    from lixipy import intels

import tkinter as tk
from tkinter import filedialog as tkFileDialog

import os

import pandas as pd

import json

#--- Yes No Question
def yn_quest(question, y_variant="Y", n_variant="N", case_sensitive=False):
    y = str(y_variant)
    n = str(n_variant)
    while True:
        answer = str(input(question + "[" + y_variant + "/" + n_variant + "]: "))
        if not case_sensitive:
            y = y.lower()
            n = n.lower()
            answer = answer.lower()
        if answer == y:
            return True
        elif answer == n:
            return False
        else:
            print("\n---> The answer is not valid, please use one of: " + y + " or " + n + "\n")

#--- File path and directory

def get_file(initial_dir = "/", window_title = "Select a File"):
    """
    Description:
        Search a File or a Directory and return the file name, file directory and the path
        
    Inputs:
        - initial_dir(str): initial directory for function
        - window_title(str): window title
       
    Outputs:
        - file_name(str): file name
        - file_dir(str): file directory
        - file_path(str): file path
            
    """
    root = tk.Tk()
  
    file_path = tkFileDialog.askopenfilename(initialdir = initial_dir, title = window_title, filetypes=(("Documento de texto", "*.txt"), ("Todos los archivos", "*.*")))
    file_dir , file_name = os.path.split(file_path)
    
    root.destroy()
    root.mainloop()
  
    return file_name, file_dir, file_path

def get_dir(initial_dir = "/", window_title = "Choose a Directory"):
    """     
    Description:
        Search a Directory and return it
     
    Inputs:
        - initial_dir(str): initial directory for function
        - window_title(str): window title
   
    Outputs:
        - dir_path(str): Directory path
        
    """
    root = tk.Tk()
  
    dir_path = tkFileDialog.askdirectory(initialdir = initial_dir, title = window_title)
    
    root.destroy()
    root.mainloop()
  
    return dir_path

def better_listdir(initial_dir="/", search_dir=True, name_filter=None, keep_files=True, keep_folders=False): #MISSING: analyze subdirectories/subfolders.
    """
    Description:
        Better version of the os.listdir function with file types and name filtering.
        If search_dir is True a window will open to search for the directory where to list directories from, if False that initial dir will be used.

    Inputs:
        - initial_dir (string): initial directory where to look for the directory to list directories from if search_dir is True, or straightly the directory if search_dir is False.
        - search_dir (boolean): wether or not to open a search window to look for the directory where to list directories from.
        - name_filter (string or list of string): filters for names of files that should be kept. If a list is passed, all conditions must be met for the file to be kept in the list.
        - keep_files (bool): filter to keep the file type directories.
        - keep_folders (bool): filter to keep the folder type directories.
    Outputs:
        - str: path where the search was done
        - list: list of directories that meet the specified criteria.
    """
    
    if search_dir:
        saving_path = get_dir(initial_dir=initial_dir, window_title="Where will you list directories from?")
    else:
        saving_path = initial_dir
    
    main_list=os.listdir(saving_path)
    folders_files_filtering = []
    name_filtering = []
    for elem in main_list:
        if "." in elem:
            if keep_files:
                folders_files_filtering.append(elem)
        else:
            if keep_folders:
                folders_files_filtering.append(elem)
    
    if name_filter is not None:
        if type(name_filter) == str:
            for elem in folders_files_filtering:
                if name_filter in elem:
                    name_filtering.append(elem)
        elif type(name_filter) == list:
            for elem in folders_files_filtering:
                if all([filt in elem for filt in name_filter]):
                    name_filtering.append(elem)
        else:
            raise ValueError("Incompatible type of name_filter parameter, should be string or list.")

        return saving_path, name_filtering
    else:
        return saving_path, folders_files_filtering

#--- DataFrame Loading

def load_dataframe(file_path, index_col = 0, sep = ",", header = None, skiprows= 10, 
                       names= ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'x', 'y', 'z', 'mlp_labels', 'time', 'rand'],
                       tag_column_name = 'mlp_labels', tag_column_index = 7):
    """
    Description:
        Combine get_file or get_dir depending on the case and load a file with the input parameters
        Generate the Dataframe and returns it
    
    Inputs:
        - file_path(str): file path
        - index_col(int): index column
        - sep(str): typo of separator you use
        - header(int or None): Defines if exist or not a row wich containg column names 
        - skiprows(int): number of rows to skip
        - names(list): columns name
        - has_tag_column(bool): if has tag column
        - tag_column_name(str): name of tag column
        - tag_column_index(int): index of tag column
       
    Outputs:
        - pd_df(pd): pandas Dataframe
        - tag_column_name(str): name of tag column
         
    """  
    if tag_column_name is not None:
      temp_tag_column_name = names[tag_column_index]
      if (temp_tag_column_name != tag_column_name):
        print("tag_column_name and tag_column_index do not coincide. Force index [Y/N]?")
        ans = input("If N, then tag_column will be chosen by tag_column_name: ")
        if (ans != "Y"):
          tag_column_name = temp_tag_column_name
    
    pd_df = pd.read_csv(file_path, sep = sep, header = header, index_col = index_col, names = names, skiprows = skiprows)
  
    return pd_df, tag_column_name

def build_dataframe(index_col = 0, sep = ",", header = None, skiprows= 10, 
                    names= ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'x', 'y', 'z', 'mlp_labels', 'time', 'rand'],
                    tag_column_name = 'mlp_labels', tag_column_index = 7, 
                    initial_dir = "/", search_dir = True):
    """  
    Description:
        Join get_file with dataframe_creation for full Dataframe Searching and Loading
        If search_dir is set to True then a window will be opened on the initial_dir to search
        to search for the dataframe file path; instead if False, initial_dir will be used as the dataframe file path.
    
    Inputs:
        - index_col(int): index column
        - sep(str): typo of separator you use
        - header(int or None): Defines if exist or not a row wich containg column names 
        - skiprows(int): number of rows to skip
        - names(list): columns name
        - has_tag_column(bool): if has tag column
        - tag_column_name(str): name of tag column
        - tag_column_index(int): index of tag column
        - initial_dir(str): initial directory to get_file
        - search_dir(bool): wether to use the initial_dir as the starting point from 
        which to search the dataframe path or as the dataframe path itself
       
    Outputs:
        - pd_df(pd): pandas Dataframe
        - pd_tag_column(pd): pandas tag column
        - pd_name(str): pandas name
        - pd_dir(str): pandas directory
        - pd_path(str): pandas path
    
    """   
    
    if search_dir:
        pd_name, pd_dir, pd_path = get_file(initial_dir)
    else:
        pd_path = initial_dir
        pd_dir, pd_name = os.path.split(pd_path)

    pd_df, pd_tag_column = load_dataframe(pd_path, index_col, sep, header, skiprows, 
                                              names, tag_column_name, tag_column_index)
    
    return pd_df, pd_tag_column, pd_name, pd_dir, pd_path

# Many DataFrame Searching and Loading from Dir

def build_many_dataframe(index_col = 0, sep = ",", header = None, skiprows= 10, 
                            names= ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'x', 'y', 'z', 'mlp_labels', 'time', 'rand'],
                            tag_column_name = 'mlp_labels', tag_column_index = 7,
                            filtering_list = [], filtering_type = "ignore",
                            initial_dir = "/", search_dir = True,
                            return_dict = False, return_joined = True):
    """
    Description:
        Generate a single dataframe with many signals at the same time using get_dir and dataframe_creation
        Analyze all .txt or .csv files into the directory and loads them in a single Dataframe
        return_dict and return_joined must be opposite 
        If return_dict, then returns a dictionary that in each key has the name of the file and in each value has the dataframe 
        If return_join, then returns a single Dataframe with everything attached. And tag_column_names only returns de first tag column name
        If search_dir is set to True then a window will be opened on the initial_dir to search
        to search for the dataframe files dir; instead if False, initial_dir will be used as the dataframe files dir.
    
    Inputs:
        - index_col(int): index column
        - sep(str): typo of separator you use
        - header(int or None): Defines if exist or not a row wich containg column names 
        - skiprows(int): number of rows to skip
        - names(list): columns name
        - has_tag_column(bool): if has tag column
        - tag_column_name(str): name of tag column
        - tag_column_index(int): index of tag 
        - filtering_list (list of str): list containing file names that should or shouldn´t (depending on filtering_type) be put in the dataframe. Defaults to an empty list because the filtering_type defaults to "ignore", that way if nothing is passed regarding filtering files, all files will be loaded.
        - filtering_type (string): if "keep" then only the files in the list will be loaded, if "ignore" then the files that are NOT in the list will be loaded. Defaults to "ignore" because the filtering_list defaults to an empty list, that way if nothing is passed regarding filtering files, all files will be loaded.
        - initial_dir(str): initial directory to get_file
        - search_dir(bool): wether to use the initial_dir as the starting point from 
        which to search the dataframe files dir or as the dataframe files dir itself
        - return_dict(bool): boolean to choose if return or not 
        - return_joined(bool): boolean to choose
            
    Outputs:
        - signal_dict(dict)(Optional): signal dictionary 
        - full_pd(pd)(Optional): full pandas Dataframe
        - tag_column_names(list): tag column names or name depending on the case
        - signal_names(list of str): list of names of the files in the directory, but not necessarily the ones loaded in the dataframe
    """    
    assert return_dict != return_joined, "return_dict and return_joined must be opposite since both or none of them can be return"
    
    if search_dir:
        files_dir = get_dir(initial_dir = initial_dir)
    else:
        files_dir = initial_dir
    
    signal_dict = {}
    tag_column_names =[]
    signal_names = []

    for file in os.listdir(files_dir):
        if (".txt" in file or ".csv" in file):
            file_path = os.path.join(files_dir, file)
            if (file not in filtering_list and filtering_type == "ignore") or (file in filtering_list and filtering_type == "keep"):
                print("Loaded Signal: " + file_path)
                file_name = file.split("-")[0]
                
                signal_pd, tag_column = load_dataframe(file_path = file_path, index_col = index_col, sep = sep, header = header, skiprows= skiprows, 
                       names= names, tag_column_name = tag_column_name, tag_column_index = tag_column_index)
            
                tag_column_names.append(tag_column)
                
                signal_dict[file_name] = signal_pd
                
            signal_names.append(file)
    
    if len(signal_dict.values()) != 0:
        full_pd = pd.concat([element for element in signal_dict.values()],
                            ignore_index = False
                            )
    else:
        full_pd = pd.DataFrame()
        tag_column_names.append(None)
    
    return_list = []

    if return_dict:
        return_list.append(signal_dict)    
    elif return_joined:
        return_list.append(full_pd)
    return_list.append(tag_column_names[0])
    return_list.append(signal_names)

    return return_list

#CSV loading
def load_csv(initial_dir = "/", search_dir=True, 
            sep=",", header=None, names=None, index_col=None, skiprows=0, lineterminator=None, delim_whitespace=False, 
            print_success=False):
    """
    Description:
        Load data from a CSV (comma-separated values) file.
        If search_dir is True a window will open to search for the directory where to load the CSV from, if False that initial dir will be used as the loading directory.

    Inputs:
        - initial_dir (string): initial directory where to look for the directory to load the file from if search_dir is True, or straightly the loading directory if search_dir is False.
        - search_dir (boolean): wether or not to open a search window to look for the directory where to load the CSV file from.
        - sep, header, names, index_col, skiprows, lineterminator, delim_whitespace: parameters used for the pandas.read_csv method, see https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html for more info on them.
        - print_success (bool): if True, a message will be printed when successfully loading the file.
    
    Outputs:
        - df (pandas.DataFrame): DataFrame object created from reading the CSV file
    """
    if search_dir:
        _, _, saving_path = get_file(initial_dir=initial_dir, window_title="Where will you load the CSV from?")
    else:
        saving_path = initial_dir

    df = pd.read_csv(saving_path, sep=sep, header=header, names=names, index_col=index_col, skiprows=skiprows, lineterminator=lineterminator, delim_whitespace=delim_whitespace)

    if print_success:
        print("Loaded file:", saving_path)
    return df

#CSV saving for arrays like FVs

def save_csv(data, sep = ",", columns=None, header=True, index=True, index_label=None, 
            search_dir=True, initial_dir = "/", file_name="CSVTable.txt", print_success=False):
    """
    Description:
        Save data as a CSV (comma-separated values). Preferably data should be passed as a pandas DataFrame or Series.
        If search_dir is True a window will open to search for the directory where to save the CSV, if False that initial dir will be used as the saving directory.

    Inputs:
        - data (DataFrame, Series, array-like): data to be saved as a CSV file.
        - sep, columns, header, index, index_label: parameters used for the pandas.DataFrame.to_csv method, see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html for more info on them.
        - search_dir (boolean): wether or not to open a search window to look for the directory where to save the CSV file.
        - initial_dir (string): initial directory where to look for the directory to save the file to if search_dir is True, or straightly the saving directory if search_dir is False.
        - file_name (string): name for the saved CSV file. Remember to include the extension!
        - print_success (bool): if True, a message will be printed when successfully saving the file.

    Outputs:
        - saving_path (string): path where the csv file was saved.
    """
    
    if type(data) != pd.DataFrame or type(data) != pd.Series: #MISSING: should check if the array has more than two dimensions, but I don't know how to do it with a list.
        saved_data = pd.DataFrame(data)
    else:
        saved_data = data
    
    #Exception on file_name
    if len(file_name.split(".")) < 2:
        raise ValueError("Please include the file extension in file_name so that the file is saved correctly.")
    if len(file_name.split(".")) > 2:
        raise ValueError("It would seem you include a dot (.) in the name of the file aside from the one separating the file extension, this might cause trouble when saving the file, please exclude it.")
    if any([symbol in file_name for symbol in ["\\", "/", ":", "?", "*", '"', "<", ">", "|"]]):
        raise ValueError('It would seem you included an invalid simbol in the name if the file, this are: \\, /, :, ?, *, ", <, >, |')
    
    if search_dir:
        saving_path = get_dir(initial_dir=initial_dir, window_title="Where will you save the CSV?") + "/" + file_name
    else:
        if initial_dir == "/":
            saving_path = initial_dir + file_name
        else:
            saving_path = initial_dir + "/" + file_name

    saved_data.to_csv(saving_path, sep=sep, columns=columns, header=header, index=index, index_label=index_label)
    
    if print_success:
        print("CSV saved to file", saving_path)
    
    return saving_path

#JSON to dictionary Loading
def load_json_todict(initial_dir = "/", search_dir=True,  
                    print_success=False):
    """
    Description:
        Load data from a JSON (JavaScript Object Notation) file to a dictionary object.
        If search_dir is True a window will open to search for the directory where to load the JSON from, if False that initial dir will be used as the loading directory.

    Inputs:
        - initial_dir (string): initial directory where to look for the directory to load the file from if search_dir is True, or straightly the loading directory if search_dir is False.
        - search_dir (boolean): wether or not to open a search window to look for the directory where to load the JSON file from.
        - print_success (bool): if True, a message will be printed when successfully loading the file.
    
    Outputs:
        - json_dict (dict): dict object created from reading the JSON file
    """
    if search_dir:
        _, _, saving_path = get_file(initial_dir=initial_dir, window_title="Where will you load the JSON from?")
    else:
        saving_path = initial_dir

    with open(saving_path, "r") as file:
        json_dict = json.load(file)

    if print_success:
        print("Loaded file:", saving_path)
    return json_dict

#JSON saving from dict
def save_json_fromdict(data, search_dir=True, initial_dir = "/", file_name="JSONdict.json", print_success=False):
    """
    Description:
        Save data as a JSON (JavaScript Object Notation). Preferably data should be passed as a dict.
        If search_dir is True a window will open to search for the directory where to save the JSON, if False that initial dir will be used as the saving directory.

    Inputs:
        - data (dict, dict-like): data to be saved as a JSON file.
        - search_dir (boolean): wether or not to open a search window to look for the directory where to save the JSON file.
        - initial_dir (string): initial directory where to look for the directory to save the file to if search_dir is True, or straightly the saving directory if search_dir is False.
        - file_name (string): name for the saved JSON file. Remember to include the extension!
        - print_success (bool): if True, a message will be printed when successfully saving the file.

    Outputs:
        - saving_path (string): path where the JSON file was saved.
    """
    
    if type(data) != dict: 
        saved_data = dict(data)
    else:
        saved_data = data
    
    #Exception on file_name
    if len(file_name.split(".")) < 2:
        raise ValueError("Please include the file extension in file_name so that the file is saved correctly.")
    if len(file_name.split(".")) > 2:
        raise ValueError("It would seem you include a dot (.) in the name of the file aside from the one separating the file extension, this might cause trouble when saving the file, please exclude it.")
    if any([symbol in file_name for symbol in ["\\", "/", ":", "?", "*", '"', "<", ">", "|"]]):
        raise ValueError('It would seem you included an invalid simbol in the name if the file, this are: \\, /, :, ?, *, ", <, >, |')
    
    if search_dir:
        saving_path = get_dir(initial_dir=initial_dir, window_title="Where will you save the JSON?") + "/" + file_name
    else:
        if initial_dir == "/":
            saving_path = initial_dir + file_name
        else:
            saving_path = initial_dir + "/" + file_name

    with open(saving_path, "w") as file:
        json.dump(saved_data, file)

    if print_success:
        print("JSON saved to file", saving_path)
    
    return saving_path

def remove_from_foldername(original_dir, to_remove):
    """
    Description:
        Function that deletes the "BEST" of the folder that was compared and is no longer the best. 
   
    Inputs:
        - original_dir(str): Directory of the model that is no longer the best
        - to_remove(str): word that will be removed
           
    Output:
        - None.
    """
    folder_name = original_dir.split("/")[-1] #we take the folder's name from the whole path
    folder_dir = original_dir.split(folder_name)[0] #and the folder's location
    if to_remove + " " in folder_name: #if the word we want to remove was spaced from the others we'll have an extra space that will look bad, so we create another exception
        new_folder_name = folder_name.replace(to_remove + " ", "")
        new_dir = folder_dir + new_folder_name
        while True: #when we change the folder name there might be another folder on the same location with the same name as what we want to replace our folders name to
            try:
                os.rename(original_dir, new_dir)
            except OSError: #this will raise an OSError, which we use to add the word "New" to the folder we are modyfing, if that exists to, we add another "New" and so on
                new_dir = new_dir + " New"
            else: #once we are able to change the name of the folder, we break from the infinite loop.
                break
            
    elif to_remove in folder_name:
        new_folder_name = folder_name.replace(to_remove, "")
        new_dir = folder_dir + new_folder_name
        while True: #when we change the folder name there might be another folder on the same location with the same name as what we want to replace our folders name to
            try:
                os.rename(original_dir, new_dir)
            except OSError: #this will raise an OSError, which we use to add the word "New" to the folder we are modyfing, if that exists to, we add another "New" and so on
                new_dir = new_dir + " New"
            else: #once we are able to change the name of the folder, we break from the infinite loop.
                break
    else: #if the word we want to remove is not in the folder's name, we just pass. this else wouldn't be necessary, but just in case we want to add something in the future
        pass