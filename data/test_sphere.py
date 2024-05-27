import numpy as np
import h5py
from tqdm import tqdm

def density_profile(x, y, z, lamb, size):
    rho = lamb * np.exp(
        - lamb * np.sqrt(
            (x-size)**2 + 
            (y-size)**2 +
            (z-size)**2
            )
    )
    return rho

def main(path, size):
    with h5py.File(path+"/sphere.hdf5", 'w') as sphere:
        sphere.create_dataset(
            'dens',
            shape=(size, size, size),
            chunks=(size / 4, size / 4, size / 4)
        )
        for i in tqdm(range(size)):
            for j in tqdm(range(size)):
                for k in (range(size)):
                    sphere['dens'][i][j][k] = density_profile(x=i, y=j, z=k, lamb=0.5, size=(size / 2))
    return None

if __name__ == "__main__":
    main(
        path='/home/tboes/Dokumente/DATA/prp_files',
        size=128
    )