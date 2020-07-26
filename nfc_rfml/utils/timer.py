from time import process_time


def timer(func):
    """Print time in fractional seconds for the execution of the given function. (Use as a decorator.)
    :param func: The function to time
    :return: The timer wrapper around the given function
    """
    def wrapper():
        start = process_time()
        func()
        print("\nExecution time: %s [s]" % (process_time() - start))

    return wrapper
