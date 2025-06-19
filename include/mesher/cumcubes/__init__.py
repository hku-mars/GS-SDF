# Copyright (c) Zhihao Liang. All rights reserved.
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, Callable

import torch
import numpy as np

from . import src as _C
from .utils import Timer, scale_to_bound


def marching_cubes(
    density_grid: Union[torch.Tensor, np.ndarray],
    thresh: float,
    scale: Optional[Union[float, Sequence]]=None,
    verbose: bool=False,
    cpu: bool=False
) -> Tuple[torch.Tensor]:
    """python wrapper of marching cubes

    Args:
        density_grid (Union[torch.Tensor, np.ndarray]):
            input density grid to realize marching cube
        thresh (float): thresh of marching cubes
        scale (Optional[Union[float, Sequence]], optional):
            the scale of density grid. Defaults to None.
        verbose (bool, optional):
            print verbose informations or not. Defaults to False.
        cpu (bool, optional):
            cpu mode(wrapper of mcubes). Defaults to False.

    Returns:
        Tuple[torch.Tensor]: vertices and faces
    """
    # get the bound according to the given scale
    lower: List[float]
    upper: List[float]
    # process scale as the bounding box
    if scale is None:
        lower = [0.0, 0.0, 0.0]
        upper = [density_grid.shape[0], density_grid.shape[1], density_grid.shape[2]]
    else:
        lower, upper = scale_to_bound(scale)

    if cpu or not torch.cuda.is_available():
        try:
            import mcubes
        except:
            raise ImportError("the cpu mode cumcubes is the wrapper of `mcubes`, please install the mcubes")
        
        density_grid = density_grid.detach().cpu().numpy()
        vertices, faces = mcubes.marching_cubes(density_grid, thresh)
        offset = np.array(lower)
        scale = (np.array(upper) - np.array(lower)) / np.array(density_grid.shape)
        vertices = vertices / scale + offset

        vertices = torch.tensor(vertices)
        faces = torch.tensor(faces.astype(np.int64))
    else:
        # process density_grid
        if isinstance(density_grid, np.ndarray): density_grid = torch.tensor(density_grid)
        density_grid = density_grid.cuda()
        density_grid = density_grid.to(torch.float32)

        if (density_grid.shape[0] < 2 or density_grid.shape[1] < 2 or density_grid.shape[2] < 2):
            raise ValueError()

        vertices, faces = _C.marching_cubes(density_grid, thresh, lower, upper)
    
    if verbose:
        print(f"#vertices={vertices.shape[0]}")
        print(f"#triangles={faces.shape[0]}")

    return vertices, faces


def marching_cubes_func(
    scale: Optional[Union[float, Sequence]]=None,
    num_x: int=100,
    num_y: int=100,
    num_z: int=100,
    func: Callable=lambda x, y, z: x**2 + y**2 + z**2,
    thresh: float=16,
    verbose: bool=False,
    cpu: bool=False
) -> Tuple[torch.Tensor]:
    """python wrapper of marching cubes via the given function

    Args:
        scale (Optional[Union[float, Sequence]], optional):
            the scale of density grid. Defaults to None.. Defaults to None.
        num_x (int, optional): the sampling scale of x-axis. Defaults to 100.
        num_y (int, optional): the sampling scale of y-axis. Defaults to 100.
        num_z (int, optional): the sampling scale of z-axis. Defaults to 100.
        func (Callable, optional): the funcion for marching cudes. Defaults to lambdax.
        thresh (float, optional): thresh of marching cubes. Defaults to 16.
        verbose (bool, optional):
            print verbose informations or not. Defaults to False.
        cpu (bool, optional):
            cpu mode(wrapper of mcubes). Defaults to False.

    Returns:
        Tuple[torch.Tensor]: vertices and faces
    """
    # convert the scale to the bound
    lower, upper = scale_to_bound(scale)

    if cpu or not torch.cuda.is_available():
        try:
            import mcubes
        except:
            raise ImportError("the cpu mode cumcubes is the wrapper of `mcubes`, please install the mcubes")
        
        vertices, faces = mcubes.marching_cubes_func(
            tuple(lower),
            tuple(upper),
            num_x,
            num_y,
            num_z,
            func,
            thresh)

        vertices = torch.tensor(vertices)
        faces = torch.tensor(faces.astype(np.int64))
    else:
        # get the sample grid
        x = torch.linspace(lower[0], upper[0], num_x)
        y = torch.linspace(lower[0], upper[0], num_y)
        z = torch.linspace(lower[0], upper[0], num_z)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        sample_points = torch.stack((X, Y, Z), dim=-1).reshape(num_x, num_y, num_z, 3)

        vertices, faces = _C.marching_cubes_func(sample_points, thresh, lower, upper, func)
    
    if verbose:
        print(f"#vertices={vertices.shape[0]}\n")
        print(f"#triangles={faces.shape[0]}\n")

    return vertices, faces


def save_mesh(
    vertices: Union[torch.Tensor, np.ndarray],
    faces: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]]=None,
    filename: Union[str, Path]="temp.ply",
    verbose: bool=False
) -> None:
    """save mesh into the given filename

    Args:
        vertices (Union[torch.Tensor, np.ndarray]): vertices of the mesh to save
        faces (Union[torch.Tensor, np.ndarray]): faces of the mesh to save
        colors (Optional[Union[torch.Tensor, np.ndarray]], optional):
            vertices of the mesh to save. Defaults to None.
        filename (Union[str, Path], optional):
            the save path. Defaults to "temp.ply".
        verbose (bool, optional):
            print verbose mention or not. Defaults to False.
    """

    if isinstance(filename, Path):
        filename = str(filename)

    if isinstance(vertices, np.ndarray): vertices = torch.tensor(vertices)
    if isinstance(faces, np.ndarray): faces = torch.tensor(faces)

    # process colors
    if colors is None:
        colors = torch.ones_like(vertices) * 127
    elif isinstance(colors, np.ndarray):
        colors = torch.tensor(colors)
    colors = colors.to(torch.uint8)

    if filename.endswith(".ply"):
        _C.save_mesh_as_ply(filename, vertices, faces, colors)
    else:
        raise NotImplementedError()
    
    if verbose:
        print(f"save as {filename} successfully!")


__all__ = ["Timer", "marching_cubes", "save_mesh"]
