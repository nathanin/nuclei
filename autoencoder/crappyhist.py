import numpy as np

def crappyhist(a, bins=25, width=100):
    h, b = np.histogram(a, bins)

    for i in range(0, bins):
        print('{:10.5f} | {:{width}s} {}'.format(b[i], '#'*int(width*h[i]/np.amax(h)), h[i], width=width))

    print('{:10.5f} |'.format(b[bins]))
