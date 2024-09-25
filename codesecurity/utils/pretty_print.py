


def percent(num, total=0):
    if total == 0:
        return "{:.2f}%".format(num*100)
    return "{:.2f}%".format(num/total*100)


