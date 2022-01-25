import random

def random_color():
    return "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
