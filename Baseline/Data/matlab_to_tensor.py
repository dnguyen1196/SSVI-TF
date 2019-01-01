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
parser.add_argument("-d", "--datatype", type=str, help="tensor entry data type", choices=["binary", "count", "real"])


args = parser.parse_args()

inputf = args.file
output = args.output
datatype = args.datatype

assert(".mat" in inputf)


tensor = sio.loadmat(inputf)
ids = tensor["id"]
xi  = tensor["xi"]

num_dimensions = ids.shape[1]
num_entries    = xi.shape[0]

size_per_dimension = [max(ids[0][j])[0] for j in range(num_dimensions)]
count = 0

# fill in the blanks for the binary data?

with open(output, "w") as f:
	f.write(",".join([str(x) for x in size_per_dimension]))
	f.write("\n")
	for i in range(num_entries):
		val = xi[i][0]
		# Remember that matlab is indexed-1
		coordinates = [ids[0][j][i,0]-1 for j in range(num_dimensions)]
		# print(coordinates)
		out = ",".join([str(x) for x in coordinates]) + "," + str(val)
		f.write(out)
		f.write("\n")
		
		relation_dim = 2
		if datatype == "binary": # If data is binary, augment [id1, id2, not id3] -> 0
			max_dim3 = size_per_dimension[relation_dim]
			for k in range(max_dim3):
				if k != coordinates[relation_dim]:
					neg_coordinates = coordinates[:-1] + [k]
					out = ",".join([str(x) for x in neg_coordinates]) + "," + str(-1)
					f.write(out)
					f.write("\n")


		# count += 1
		# if count == 10:
		# 	break
		


	