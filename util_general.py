import numpy as np
import sys

class BatchHelper():
    def __init__(self,n_data, batch_size=None):
        self.n_data = n_data
        if batch_size is None:
            self.batch_size = n_data
        else:
            self.batch_size = batch_size
        self.current_label = 0
        self.iterations = 0

    def next_batch_slices(self):
        start = self.current_label
        end = min(self.current_label + self.batch_size, self.n_data)
        if end >= self.n_data:
            self.current_label = 0
            self.iterations += 1
        else:
            self.current_label = end
        return start, end

    def reset(self):
        self.current_label = 0
        self.iterations = 0


def dl_to_ld(DL):
    """
    transforms a dictionary of lists to a list of dictionaries
    https://stackoverflow.com/a/33046935
    >>> dl_to_ld({"a":[1,2,3],"b":[2,4,5]})
    [{'a': 1, 'b': 2}, {'a': 2, 'b': 4}, {'a': 3, 'b': 5}]
    >>> ld_to_dl(dl_to_ld({"a":[1,2,3],"b":[2,4,5]}))
    {'a': [1, 2, 3], 'b': [2, 4, 5]}
    """
    return [dict(zip(DL,t)) for t in zip(*DL.values())]


def ld_to_dl(LD):
    """
    transforms a list of dictionaries to a dictionary of lists
    https://stackoverflow.com/a/33046935
    >>> ld_to_dl([{"a":1,"b":2},{"a":2,"b":4},{"a":3,"b":5}])
    {'a': [1, 2, 3], 'b': [2, 4, 5]}
    >>> dl_to_ld(ld_to_dl([{"a":1,"b":2},{"a":2,"b":4},{"a":3,"b":5}]))
    [{'a': 1, 'b': 2}, {'a': 2, 'b': 4}, {'a': 3, 'b': 5}]
    """
    return {k: [dic[k] for dic in LD] for k in LD[0]}


def sort_list_by(list, by):
    """
    sorts one list by the elements of another list.
    https://stackoverflow.com/a/6618543
    :param list: list to be sorted
    :param by: the list by which to sort
    :return: list
    >>> sort_list_by(["b", "d", "f", "c", "a"],[2, 4, 6, 3, 1])
    ['a', 'b', 'c', 'd', 'f']
    """
    return [x for y, x in sorted(zip(by, list),key = lambda x:x[0])]



def merge_dicts(d1,d2):
    """
    Merges two dictionaries into one.
     If a key appears in both dictionaries, the value from the second dictionary (d2) is used.
    :param d1: dictionary
    :param d2: dictionary
    :return: dictionary
    """
    return {**d1, **d2}


def try_get_dict_value(dict, key, default):
    if key in dict:
        return dict[key]
    else:
        return default


def match_with_default_dict(in_dict,default_dict):
    return {k: try_get_dict_value(in_dict, k, default_dict[k]) for k in default_dict.keys()}

def stdout_to_str(function):
    """
    Executa a function and collect all console output in a string.
    based on https://stackoverflow.com/a/21341209
    :param function: an executable function that  prints to console
    :return: function output, str
    """

    class ListStream:
        def __init__(self):
            self.data = ""

        def write(self, s):
            self.data += s

    sys.stdout = x = ListStream()

    fun_out = function()

    sys.stdout = sys.__stdout__
    console_out = x.data
    return fun_out, console_out


def merge_index_regions(indices, tolerance=0):
    # TODO: find anomaly peak detection based
    """
    Finds connected index regions based on threshold and statistical test,
    window size and step size chosen at class initialization
    Keyword arguments:
        :param indices: list int -- a list of indices e.g. from a list or 1 dimension of an array
        :param tolerance: int -- a tolerance value, defines small distance
        at which non-connected anomaly regions should still be merged.
        """
    indices.sort()
    index_regions = []
    current_start = None
    current_end = None
    for anomaly in indices:
        # initialize first anomaly region
        if current_start is None:
            # start at current anomaly
            current_start = anomaly
            # end at the end of the rolling window used to calculate anomaly measure
            current_end = anomaly + 1
        # does current anomaly lie within or connect to the previous anomaly region
        # or if not, is the distance smaller then the tolerance
        elif current_end + tolerance >= anomaly:
            # merge anomaly regions
            new_end = anomaly + 1
            # should rigorous testing be performed before merging?
            current_end = new_end
        else:
            # else anomaly region is complete, save and create new anomaly region
            index_regions += [(current_start, current_end)]
            current_start = anomaly
            current_end = anomaly + 1
    index_regions += [(current_start, current_end)]
    return index_regions