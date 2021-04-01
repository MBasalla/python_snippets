import numpy as np

def sort_n_2_d_array(a):
    """
    Sorts the rows of a 2d array such that elements of lower index.
    """
    # sort last dimension
    row_length = a.shape[1]
    a = a[a[:,-1].argsort()] # First sort doesn't need to be stable.
    for i in reversed(range(row_length-1)):
        # sort previous dimensions
        a = a[a[:,i-row_length].argsort(kind='mergesort')]
    return a


def match_2d_arrays(ar1, ar2, diff_threshold= None,used1=[],used2=[]):
    pairs = []
    index_pairs = []
    used2 = used2
    for i in range(len(ar1)):
        if i in used1:
            continue
        row1 = ar1[i]
        best_diff = np.infty
        best_idx = 0
        # print("used2", used2)
        for j in range(len(ar2)):
            if j in used2:
                continue
            row2 = ar2[j]
            # print("indices:", i,j)
            # print("rows:", row1,row2)
            # compute difference between current elements (sum element wise diff)
            curr_diff = np.sum(np.abs(row1 - row2))
            # print("curr_diff",curr_diff)
            # print("best_diff", best_diff)
            # if diff smaller:
            if curr_diff < best_diff:
                # update best diff
                best_diff = curr_diff
                best_idx = j

        if diff_threshold is None or best_diff <= diff_threshold:
            pairs.append((row1, ar2[best_idx]))
            index_pairs.append((i,best_idx))
            used2.append(best_idx)

    return pairs, index_pairs


def match_hierarchical(ar1, ar2,thresholds):
    used1 = []
    used2 = []
    index_pairs = []
    pairs = []
    for th in thresholds:
        p, ip = match_2d_arrays(ar1, ar2,th,used1,used2)
        pairs += p
        index_pairs += ip
        if len(index_pairs) > 0:
            used1 = list(np.array(index_pairs)[:,0])
            used2 = list(np.array(index_pairs)[:, 1])
    return pairs,index_pairs


def rolling_window_multi(a, window, axis=-1, step_size=1):
    '''Return a windowed array.
    source: https://gist.github.com/joe-antognini/ebef3ecfb7624d2980eae7ead8007cfc
    Slightly adapted by adding step size
    Parameters:
      a: A numpy array
      window: The size of the window

    Returns:
      An array that has been windowed
    '''

    if axis == -1:
        axis = a.ndim - 1

    if 0 <= axis <= a.ndim - 1:
        shape = (a.shape[:axis] + (a.shape[axis] - window + 1, window) +
                 a.shape[axis + 1:])
        strides = a.strides[:axis] + (step_size*a.strides[axis],) + a.strides[axis:]
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    else:
        raise ValueError('rolling_window: axis out of bounds')


def quadratic_subarray(indices, original):
    """
    Takes a 2D array and a 1D array of indices
    Returns a smaller 2D matrix containing elements [i,j]
    of the larger matrix for all indices i and j in the index array.
    Keyword arguments:
        :param indicies: 1D array of int-- array of indices (int) for creating the smaller matrix
        :param original: 2D array of any data type -- array from which sub-array is created
    """

    size=len(indices)
    sub = np.zeros([size,size])
    for i in range(size):
        if (i >= original.shape[0]) or (i >= original.shape[1]):
            IndexError("Indesx %i must not exceed both dimensions %i and %i of the original matrix!" %(
                i, original.shape[0], original.shape[1]))
        for j in range(i, size):
            if (j >= original.shape[0]) or (j >= original.shape[1]):
                IndexError("Indesx %i must not exceed both dimensions %i and %i of the original matrix!" % (
                j, original.shape[0], original.shape[1]))
            sub[i][j] = original[indices[i]][indices[j]]
            sub[j][i] = original[indices[j]][indices[i]]
    return sub


def rand_indices(num, end, start=0, step = 1):
    """
    Creates a set of random integers between values start and end without repetition of values.
    :param num: int -- number of random indices
    :param end: int -- maximum value of random indices
    :param start: int -- minimum value of random indices (default 0)
    :param step: int -- step size of the indices
    :return: 1D array of ints -- num values between start and end
    """

    # all possible indices
    remaining = list(range(start, end, step))
    # selected indices
    indices = []
    for i in range(num):
        # choose a random list element
        rand_element = remaining[random.randint(0, len(remaining) - 1)]
        # add to selected indices
        indices += [rand_element]
        # and remove from list so it will not be picked again.
        remaining.remove(rand_element)
        # list length decreases
        end -= 1
    return indices
