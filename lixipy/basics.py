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

#--- DataFrame Loading

def load_dataframe(file_path, index_col = 0, sep = ",", header = None, skiprows= 10, 
                       names= ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'x', 'y', 'z', 'mlp_labels', 'time', 'rand'],
                       has_tag_column = True, tag_column_name = 'mlp_labels', tag_column_index = 7):
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
                    has_tag_column = True, tag_column_name = 'mlp_labels', tag_column_index = 7, 
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
                                              names, has_tag_column, tag_column_name, tag_column_index)
    
    return pd_df, pd_tag_column, pd_name, pd_dir, pd_path

# Many DataFrame Searching and Loading from Dir

def build_dataframe_fromdir(index_col = 0, sep = ",", header = None, skiprows= 10, 
                            names= ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'x', 'y', 'z', 'mlp_labels', 'time', 'rand'],
                            has_tag_column = True, tag_column_name = 'mlp_labels', tag_column_index = 7,
                            filtering_list = [],
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
        - filtering_list (list of str): list containing file names that shouldn´t be put in the dataframe even though they are in the selected dir
        - initial_dir(str): initial directory to get_file
        - search_dir(bool): wether to use the initial_dir as the starting point from 
        which to search the dataframe files dir or as the dataframe files dir itself
        - return_dict(bool): boolean to choose if return or not 
        - return_joined(bool): boolean to choose
            
    Outputs:
        - signal_dict(dict): signal dictionary 
        - tag_column_names(list): tag column names or name depending on the case
        - signal_names(list of str): list of names of the files in the directory, but not necessarily the ones loaded in the dataframe
        - full_pd(pd): full pandas Dataframe
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
            if (file not in filtering_list):
                print("Loaded Signal: " + file_path)
                file_name = file.split("-")[0]
                
                signal_pd, tag_column = load_dataframe(file_path = file_path) #MISSING: FILL THIS WITH VALUES
            
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
    
    if return_dict:
        return signal_dict, tag_column_names, signal_names
    
    elif return_joined:
        return full_pd, tag_column_names[0], signal_names

# - FV Loading and Saving
#MISSING: Redo load FV and should generalize it or bring it to freqs and times separately
#MISSING: Basing on the save weights, save model JSON and load model JSON, should add here functions to read and save JSON and CSVs, and then put the export FV, Weights and models on freqs and intel

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