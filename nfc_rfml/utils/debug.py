import functools


def debug(func):
    """Print the arguments of the function and its return value to console to ease debugging.
    :param func: The function to debug
    :return: The debug wrapper around the given function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(arg) for arg in args]
        kwargs_repr = [f"{k}={v}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")

        value = func(*args, **kwargs)
        print(f"{func.__name__} returned {value}")

        return value

    return wrapper
