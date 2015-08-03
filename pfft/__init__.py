from .core import *

def split_size_2d(s):
    """ Split `s` into two integers, 
        a and d, such that a * d == s and a <= d

        returns:  a, d
    """
    a = int(s** 0.5) + 1
    d = s
    while a > 1:
        if s % a == 0:
            d = s // a
            break
        a = a - 1 
    return a, d
