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
sigma_eps = 0.3
EP_or_ED = 'EP'
GL_or_LG = 'GL'
triu_tril = 'triu'
row_col_major = 'row-major'
probe_ordering = [['L', 'L'], [GL_or_LG[0], GL_or_LG[1]], ['G', 'G']]
block_index = 'ell'
n_probes = 2
survey = 'Euclid'
# ! end settings


if survey == 'SKA' or survey == 'SKA_withbeta':
    fsky = 0.7
    n_gal = 8.7
elif survey == 'Euclid':
    fsky = survey_area_ISTF / deg2_in_sphere
    n_gal = 30
else:
    raise ValueError('survey must be either "SKA" or "SKA_withbeta" or "Euclid"')

ind = mm.build_full_ind(triu_tril, row_col_major, zbins)

cl_LL_3d = np.load(f'{project_path}/data/Euclid/CLL.npy').transpose(2, 0, 1)
cl_GG_3d = np.load(f'{project_path}/data/Euclid/CGG.npy').transpose(2, 0, 1)
cl_LG_3d = np.load(f'{project_path}/data/Euclid/CLG.npy').transpose(2, 0, 1)
cl_GL_3d = cl_LG_3d.transpose(0, 2, 1)
ell_values = np.load(f'{project_path}/data/Euclid/ell.npy')

cl_LL_3d_v2 = np.load(f'{project_path}/data/OWL/delta_ell.npy')
cl_LL_3d_v3 = np.load(f'{project_path}/data/Euclid_v2/delta_ell.npy')
assert np.array_equal(cl_LL_3d_v2, cl_LL_3d_v3)

ell_idx = 3
mm.matshow(cl_LL_3d[ell_idx, :, :], log=True, title='cl_GL_3d')
mm.matshow(cl_LL_3d_v2[ell_idx, :, :], log=True, title='cl_GL_3d_v2')




print('done')
