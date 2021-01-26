from scipy.signal import kaiserord, lfilter, firwin,  butter, filtfilt
import numpy as np

# TODO: take out delay from returned time series (move to the left by delay)
# based on https://scipy-cookbook.readthedocs.io/items/FIRFilter.html


# TODO: take out delay from returned time series (move to the left by delay)
# based on https://scipy-cookbook.readthedocs.io/items/FIRFilter.html
def low_pass_filter(sign, freq=16000, rel_width=5, ripple_db=60, cutoff_hz=1000.0):
    '''
    Implements a kaiser window low pass filter.
    sign; array, signal
    freq: int, signal sampling frequeny
    rel_width: int relative width of the transition
    ripple_db: int, acceptable ripple of the filter
    cuoff_hz: float, cutoff frequency
    '''
    # The Nyquist rate of the signal.
    nyq_rate = freq / 2.0
    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.
    width = rel_width / nyq_rate

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

    delay = 0.5 * (N - 1) / freq

    # Use lfilter to filter x with the FIR filter.
    return lfilter(taps, 1.0, sign), N, delay


# based on https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x, window_len=10, window='hanning'):
    '''
    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    import numpy as np
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    '''

    # assure that data has the correct shape
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    # assure that the window size is smaller than the data
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    # if window too small just return the original data
    if window_len < 3:
        return x

    # make sure window string corresponds to an window type that is implemented in scipy
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # create slices based on window size for convolution
    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]

    # create or load the window weights as an array
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    # convolce the slices with the window weights
    y = np.convolve(w / w.sum(), s, mode='same')

    return y[window_len - 1:-window_len + 1]


def butter_bandpass(lowcut, highcut, fs=16000, order=5):
    '''
    Implements a butterworth bandpass filter.
    based on https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    lowcut: lower bound of the isolated frequency band
    highcut: upper bound of the isolated frequency band
    fs: frequency of the signal
    order: oerder of the signal
    '''
    # nyquist frequency
    nyq = 0.5 * fs
    # low cut
    low = lowcut / nyq
    # high cut
    high = highcut / nyq
    # define filter
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_lowpass(highcut, fs=16000, order=5):
    '''
    Implements a butterworth highpass filter.
    based on https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    highcut: upper bound of the isolated frequency band
    fs: frequency of the signal
    order: oerder of the signal
    '''
    # nyquist frequency
    nyq = 0.5 * fs
    # high cut
    high = highcut / nyq
    # define filter
    b, a = butter(order, high)
    return b, a


#
def butter_highpass(lowcut, fs=16000, order=5):
    '''
    Implements a butterworth highpass filter.
    based on  based on https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    highcut: upper bound of the isolated frequency band
    fs: frequency of the signal
    order: oerder of the signal
    '''
    # nyquist frequency
    nyq = 0.5 * fs
    # low cut
    low = lowcut / nyq
    # define filter
    b, a = butter(order, low, btype='highpass')
    return b, a


# based on https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html


# based on https://scipy-cookbook.readthedocs.io/items/FiltFilt.html
def bandpass_filter(sign, lowcut, highcut, fs=16000, order=3):
    '''
    Applies a bandpass filter to the signal.
    lowcut: lower bound of the isolated frequency band
    highcut: upper bound of the isolated frequency band
    fs: frequency of the signal
    order: oerder of the signal
    '''
    # Create an order bandpass butterworth filter.
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)

    # Use filtfilt to apply the filter.
    return filtfilt(b, a, sign)


# based on https://scipy-cookbook.readthedocs.io/items/FiltFilt.html
def lowpass_filter(sign, highcut, fs=16000, order=3):
    '''
    Applies a lowpass filter to the signal.
    highcut: upper bound of the isolated frequency band
    fs: frequency of the signal
    order: oerder of the signal
    '''
    # Create a lowpass butterworth filter.
    b, a = butter_lowpass(highcut, fs, order=order)

    # Use filtfilt to apply the filter.
    return filtfilt(b, a, sign)


# based on https://scipy-cookbook.readthedocs.io/items/FiltFilt.html
def highpass_filter(sign, lowcut, fs=16000, order=3):
    '''
    Applies a highpass filter to the signal.
    lowcut: lower bound of the isolated frequency band
    fs: frequency of the signal
    order: oerder of the signal
    '''
    # Create an highpass butterworth filter.
    b, a = butter_highpass(lowcut, fs, order=order)

    # Use filtfilt to apply the filter.
    return filtfilt(b, a, sign)


