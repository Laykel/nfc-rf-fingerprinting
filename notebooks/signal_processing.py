import numpy as np


def complex_to_mag(m):
    """
    Takes in a numpy array of complex numbers and returns an array of
    the magnitudes of those numbers
    """
    return np.sqrt(np.real(m)**2 + np.imag(m)**2)

def shift_down(m, by):
    """Shifts every element of a numpy array of floats down"""
    return m - by

def binary_slicer(m):
    """
    Takes in a numpy array of floats and replaces every element under 0
    by a 0 and every other element by a 1
    """
    mask = m > 0
    return 1 * mask