import functools
from time import process_time


def timer(func):
    """Print time in fractional seconds for the execution of the given function. (Use as a decorator.)
    :param func: The function to time
    :return: The timer wrapper around the given function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = process_time()
        value = func(*args, **kwargs)
        end = process_time()
        print(f"Execution time: {end - start}[s]")
        return value

    return wrapper
