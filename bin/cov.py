import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

project_path = Path.cwd().parent

sys.path.append(f'{project_path.parent}/common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path.parent}/common_data/common_config')
import ISTF_fid_params as ISTFfid
import mpl_cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

# ! settings
survey_area_ISTF = 15_000  # deg^2
deg2_in_sphere = 41252.96  # deg^2 in a spere

zbins = 10
nbl = 32
sigma_eps = 0.30
EP_or_ED = 'EP'
GL_or_LG = 'GL'
triu_tril = 'triu'
row_col_major = 'row-major'
probe_ordering = [['L', 'L'], [GL_or_LG[0], GL_or_LG[1]], ['G', 'G']]
block_index = 'ell'
n_probes = 2
survey = 'SKA_eNLA'
# ! end settings


if survey.startswith('SKA'):
    fsky = 0.7
    n_gal = 8.7
elif survey.startswith('Euclid'):
    fsky = survey_area_ISTF / deg2_in_sphere
    n_gal = 30
else:
    raise ValueError('survey name must start with "SKA" or "Euclid"')

zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

ind = mm.build_full_ind(triu_tril, row_col_major, zbins)
ind_auto = ind[:zpairs_auto, :]

cl_LL_3d = np.load(f'{project_path}/data/{survey}/CLL.npy')
cl_LG_3d = np.load(f'{project_path}/data/{survey}/CLG.npy')
cl_GL_3d = np.load(f'{project_path}/data/{survey}/CGL.npy')
cl_GG_3d = np.load(f'{project_path}/data/{survey}/CGG.npy')
print(cl_LL_3d.shape, cl_LG_3d.shape, cl_GG_3d.shape)

if survey != 'SKA_TATT':
    # in these 2 cases the cls are already in the correct shape
    cl_LL_3d = cl_LL_3d.transpose(2, 0, 1)
    cl_LG_3d = cl_LG_3d.transpose(2, 0, 1)
    cl_GL_3d = cl_GL_3d.transpose(2, 0, 1)
    cl_GG_3d = cl_GG_3d.transpose(2, 0, 1)

assert cl_GL_3d.shape == cl_LG_3d.shape == cl_LL_3d.shape == cl_GG_3d.shape == (nbl, zbins, zbins)

ell_values = np.load(f'{project_path}/data/{survey}/ell.npy')
delta_ell = np.load(f'{project_path}/data/{survey}/delta_ell.npy')

cl_3x2pt_5d = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
cl_3x2pt_5d[0, 0, :, :, :] = cl_LL_3d
cl_3x2pt_5d[0, 1, :, :, :] = cl_LG_3d
cl_3x2pt_5d[1, 0, :, :, :] = cl_GL_3d
cl_3x2pt_5d[1, 1, :, :, :] = cl_GG_3d

noise_3x2pt_4d = mm.build_noise(zbins, n_probes, sigma_eps2=sigma_eps ** 2, ng=n_gal, EP_or_ED=EP_or_ED)

# create a fake axis for ell, to have the same shape as cl_3x2pt_5d
noise_3x2pt_5d = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
for probe_A in (0, 1):
    for probe_B in (0, 1):
        for ell_idx in range(nbl):
            noise_3x2pt_5d[probe_A, probe_B, ell_idx, :, :] = noise_3x2pt_4d[probe_A, probe_B, ...]

cl_LL_5d = cl_LL_3d[np.newaxis, np.newaxis, ...]
cl_GG_5d = cl_GG_3d[np.newaxis, np.newaxis, ...]
noise_LL_5d = noise_3x2pt_5d[0, 0, ...][np.newaxis, np.newaxis, ...]
noise_GG_5d = noise_3x2pt_5d[1, 1, ...][np.newaxis, np.newaxis, ...]

cov_GO_WL_6D = mm.covariance_einsum(cl_LL_5d, noise_LL_5d, fsky, ell_values, delta_ell)[0, 0, 0, 0, ...]
cov_GO_GC_6D = mm.covariance_einsum(cl_GG_5d, noise_GG_5d, fsky, ell_values, delta_ell)[0, 0, 0, 0, ...]

# ! reshape
cov_3x2pt_GO_10D_arr = mm.covariance_einsum(cl_3x2pt_5d, noise_3x2pt_5d, fsky, ell_values, delta_ell)
cov_3x2pt_dict_10D = mm.cov_10D_array_to_dict(cov_3x2pt_GO_10D_arr)

cov_3x2pt_GO_4D = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_dict_10D, probe_ordering, nbl, zbins, ind.copy(), GL_or_LG)
cov_GO_WL_4D = mm.cov_6D_to_4D(cov_GO_WL_6D, nbl, zpairs_auto, ind_auto)
cov_GO_GC_4D = mm.cov_6D_to_4D(cov_GO_GC_6D, nbl, zpairs_auto, ind_auto)

cov_GO_WL_2D = mm.cov_4D_to_2D(cov_GO_WL_4D, block_index=block_index)
cov_GO_GC_2D = mm.cov_4D_to_2D(cov_GO_GC_4D, block_index=block_index)
cov_3x2pt_GO_2D = mm.cov_4D_to_2D(cov_3x2pt_GO_4D, block_index=block_index)

mm.matshow(cov_GO_WL_2D, log=True, abs_val=False, title='cov_WL')
mm.matshow(cov_GO_GC_2D, log=True, abs_val=False, title='cov_GC')
mm.matshow(cov_3x2pt_GO_2D, log=True, abs_val=False, title='cov_3x2pt')

np.savez_compressed(f'../output/{survey}/cov_3x2pt_GO_10D_arr.npz', cov_3x2pt_GO_10D_arr)
np.savez_compressed(f'../output/{survey}/cov_3x2pt_GO_4D.npz', cov_3x2pt_GO_4D)
np.savez_compressed(f'../output/{survey}/cov_3x2pt_GO_2D.npz', cov_3x2pt_GO_2D)
np.savez_compressed(f'../output/{survey}/cov_WL_GO_2D.npz', cov_GO_WL_2D)
np.savez_compressed(f'../output/{survey}/cov_GC_GO_2D.npz', cov_GO_GC_2D)

print('done')
