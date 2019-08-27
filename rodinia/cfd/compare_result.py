import numpy as np
import math
density = np.loadtxt("density").flatten()
ref_density = np.loadtxt("ref_result/density").flatten()
momentum = np.loadtxt("momentum").flatten()
ref_momentum =  np.loadtxt("ref_result/momentum").flatten()
density_energy = np.loadtxt("density_energy").flatten()
ref_density_energy = np.loadtxt("ref_result/density_energy").flatten()
rmse_density = math.sqrt(sum((density- ref_density)*(density- ref_density))/ float(len(density)))
rmse_momentum = math.sqrt(sum((momentum- ref_momentum)*(momentum- ref_momentum))/ float(len(momentum)))
rmse_density_energy = math.sqrt(sum((density_energy - ref_density_energy)*(density_energy - ref_density_energy))/ float(len(density_energy)))

all_in = np.concatenate((density, momentum, density_energy)).flatten()
all_ref = np.concatenate((ref_density, ref_momentum, ref_density_energy)).flatten()
rmse_all = math.sqrt(sum((all_in- all_ref)*(all_in- all_ref))/ float(len(all_in)))

print(' density len %d rmse %E' %(len(density), rmse_density) )

print(" momentum len %d rmse %E" %(len(momentum), rmse_momentum) )


print(" density_energy len %d rmse %E" %(len(density_energy), rmse_density_energy ) )


print(" all len %d rmse %E" %(len(all_in), rmse_all) )
