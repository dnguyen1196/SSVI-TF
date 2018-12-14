import scipy.io as sio
from Tensor.real_tensor import RV_tensor
from Tensor.binary_tensor import binary_tensor
from Tensor.count_tensor import count_tensor
import numpy as np


def synthesize_tensor(dims, datatype, using_ratio, noise):
    real_dim = 20
    means = [np.ones((real_dim,)) * 1, np.ones((real_dim,)) * 4, np.ones((real_dim,)) * 2]
    covariances = [np.eye(real_dim) * 0.5, np.eye(real_dim) * 0.5, np.eye(real_dim) * 0.5]

    if datatype == "binary":
        tensor = binary_tensor()
    elif datatype == "real":
        tensor = RV_tensor()
    elif datatype == "count":
        tensor = count_tensor()

    """
    """
    tensor.synthesize_data(dims, means, covariances, real_dim, \
                           train=0.8, sparsity=1, noise=noise, noise_ratio=using_ratio)
    return tensor

datatype = "binary"
test_tensor = synthesize_tensor([10, 10, 10], datatype, True, 0.05)


"""
aving to a MATLAB cell array just involves making a numpy object array:

>>>
>>> obj_arr = np.zeros((2,), dtype=np.object)
>>> obj_arr[0] = 1
>>> obj_arr[1] = 'a string'
>>> obj_arr
array([1, 'a string'], dtype=object)
>>> sio.savemat('np_cells.mat', {'obj_arr':obj_arr})
octave:16> load np_cells.mat
octave:17> obj_arr
obj_arr =
{
  [1,1] = 1
  [2,1] = a string
}
"""

vals = test_tensor.train_vals
coordinates = test_tensor.train_entries

ndim = len(test_tensor.dims)
temp = np.empty((ndim,), dtype=np.object)

# Note that matlab is 1-indexed
for dim, _ in enumerate(test_tensor.dims):
    obj_arr = np.asarray([[coord[dim]+1] for coord in coordinates], dtype=np.double)
    temp[dim] = obj_arr
#
# idx = np.array(idx, dtype=np.object)
# temp = np.empty(idx.shape, dtype=np.object)
#
# for dim, nrows in enumerate(test_tensor.dims):
#     temp[dim, :, :] = idx[dim, :, :]

# idx = np.asarray([np.asarray([[1],[2]]), np.asarray([[1],[2],[3]])])
#
# temp = np.empty(idx.shape, dtype=np.object)
# temp[0] = np.asarray([[1],[2]])
# temp[1] = np.asarray([[1],[2],[3]])
#

# xi = np.asarray([val+1 for val in vals])
if datatype == "binary":
    xi  = np.asarray([0 if val == -1 else 1 for val in vals], dtype=np.double)
elif datatype == "count":
    xi = np.asarray(vals, dtype=np.double)

xi = np.expand_dims(xi, axis=1)
data = {"id" : temp, "xi" : xi}

if datatype == "count":
    sio.savemat('./Matlab/count_synthetic_tensor.mat', data)
else:
    sio.savemat('./Matlab/binary_synthetic_tensor.mat', data)


