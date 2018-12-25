import argparse
import sys
import os

import scipy.io as sio

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


from Tensor.real_tensor import RV_tensor
from Tensor.binary_tensor import binary_tensor
from Tensor.count_tensor import count_tensor


# Create a text file such that


parser = argparse.ArgumentParser(description="Transforming from matlab .mat file to text file")
parser.add_argument("-f", "--file", type=str, help=".mat file", required=True)
parser.add_argument("-o", "--output", type=str, help="output text file", required=True)
# parser.add_argument("-d", "--datatype", type=str, help="tensor entry data type", choices=["binary", "count", "real"])


args = parser.parse_args()

inputf = args.file
output = args.output

assert(".mat" in inputf)


tensor = sio.loadmat(inputf)
ids = tensor["id"]
xi  = tensor["xi"]

num_dimensions = ids.shape[1]
num_entries    = xi.shape[0]

size_per_dimension = [max(ids[0][j])[0] for j in range(num_dimensions)]
count = 0

with open(output, "w") as f:
	f.write(",".join([str(x) for x in size_per_dimension]))
	f.write("\n")
	for i in range(num_entries):
		val = xi[i][0]
		coordinates = [ids[0][j][i,0] for j in range(num_dimensions)]
		out = ",".join([str(x) for x in coordinates]) + "," + str(val)
		f.write(out)
		f.write("\n")
		# if count == 10:
		# 	break
		


	