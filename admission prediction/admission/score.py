import random

def score(a):
    if a<20:
        return a+80
    elif a>20 and a<50:
        return a+(random.randint(55,60))
    else:
        return a+(random.randint(40,45))