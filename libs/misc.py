import numpy as np

def shuffle(a, b):
    assert len(a) == len(b)
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def dropout_mask(samples, nodes, threshold=0.5):
    mask = np.random.rand(samples, nodes)
    mask[mask<threshold] = 0
    mask[mask!=0] = 1
    return mask