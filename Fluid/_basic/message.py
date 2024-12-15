import time

def log(*args, **kwargs):
    # get the current time and format it as [hh:mm:ss]
    timestamp = time.strftime("[%H:%M:%S]", time.localtime())
    # print the timestamp with all passed arguments, *args and **kwargs allow for any number of arguments
    print(timestamp, *args, **kwargs)