from modules import laplacian_fixed
from modules.data import Data as _Data


class Data(_Data):
    def _solve(self, grid_resolution, mpoints, fpoints):
        return laplacian_fixed.sliceToSlice3DLaplacian(grid_resolution, mpoints, fpoints)
