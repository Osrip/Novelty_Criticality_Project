import gzip
import pickle
import os
from os import listdir
from os.path import isfile, join
import re


def save_pickle_compressed(name, data):
    '''
    save data to a compressed pickle file
    '.pgz' is added to name as a suffix
    name includes whole file path
    '''
    with gzip.GzipFile(name + '.pgz', 'w') as f:
        pickle.dump(data, f)


def load_pickle_compressed(name, add_pgz=True):
    '''
    load compressed pickle file, adds '.pgz' to name by default.
    name includes complete file path
    '''
    if add_pgz:
        name = name + '.pgz'
    data = gzip.GzipFile(name , 'rb')
    data = pickle.load(data)
    return data


def mkdir(dir):
    '''
    Creates dir in case it does not exist yet
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)


def all_file_names_that_include(file_name_include, file_path):
    '''
    This function returns all file names in file_path, that include file_name_include.
    '''
    file_name_list = [f for f in listdir(file_path) if isfile(join(file_path, f)) and file_name_include in f]
    return file_name_list


def load_all_files_that_include(file_name_include, file_path):
    '''
    This function loads all files in file_path, that include file_name_include.
    '''
    file_name_list = [f for f in listdir(file_path) if isfile(join(file_path, f)) and file_name_include in f]
    return [load_pickle_compressed(file_path + name, add_pgz=False) for name in file_name_list]


def load_all_files_that_include_as_dict(file_name_include, file_path):
    '''
    This function loads all files in file_path, that include file_name_include.
    Further it looks for an integer and returns a dict with the integer as key and the loaded data as entry
    '''
    file_name_list = [f for f in listdir(file_path) if isfile(join(file_path, f)) and file_name_include in f]
    return {int(re.search(r'\d+', name).group()): load_pickle_compressed(file_path + name, add_pgz=False)
            for name in file_name_list}


def save_settings(global_dict):
    '''
    save a dict 'settings' both as pickle and as .csv
    '''
    # global_dict['device'] = ''
    path = '{}/settings/'.format(global_dict['save_dir'])
    mkdir(path)

    with open(path + 'global_dict.csv', 'w') as f:
        for key in global_dict.keys():
            f.write("%s,%s\n" % (key, global_dict[key]))
    pickle_out = open('{}global_dict.pickle'.format(path), 'wb')
    pickle.dump(global_dict, pickle_out)
    pickle_out.close()


def detect_int_after_substr(string, substring):
    '''
    returns integer after a given substring
    '''
    match = re.search(substring+'(\d+)', string)
    return int(match.group(1))


def load_files_ints_after_substr(file_path, substr, substr_int_dict, only_load_one_file=False):
    '''
    This function loads all files in file_path with substring substr, which have certain integers behind certain substrings
    that are specified by substr_int_dict {substr: int}
    All requirements in substr_int_dict hve to be fulfilled
    In case only_load_one_file == True, only one file is loaded (first in list, that fulfills requirements)
    '''
    file_names = all_file_names_that_include(substr, file_path)
    file_list = []
    for file_name in file_names:
        load_file = True
        for substr, int_wanted in substr_int_dict.items():
            int_actual = detect_int_after_substr(file_name, substr)
            load_file = load_file and int(int_actual)==int(int_wanted)
        if load_file:
            if only_load_one_file:
                return load_pickle_compressed('{}/{}'.format(file_path,file_name), add_pgz=False)
            file_list.append(load_pickle_compressed('{}/{}'.format(file_path,file_name), add_pgz=False))
    return file_list
