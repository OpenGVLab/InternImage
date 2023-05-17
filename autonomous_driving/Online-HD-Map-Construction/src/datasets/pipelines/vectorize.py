import numpy as np
from mmdet.datasets.builder import PIPELINES
from shapely.geometry import LineString
from numpy.typing import NDArray
from typing import List, Tuple, Union, Dict

@PIPELINES.register_module(force=True)
class VectorizeMap(object):
    """Generate vectoized map and put into `semantic_mask` key.
    Concretely, shapely geometry objects are converted into sample points (ndarray).
    We use args `sample_num`, `sample_dist`, `simplify` to specify sampling method.

    Args:
        roi_size (tuple or list): bev range .
        normalize (bool): whether to normalize points to range (0, 1).
        coords_dim (int): dimension of point coordinates.
        simplify (bool): whether to use simpily function. If true, `sample_num` \
            and `sample_dist` will be ignored.
        sample_num (int): number of points to interpolate from a polyline. Set to -1 to ignore.
        sample_dist (float): interpolate distance. Set to -1 to ignore.
    """

    def __init__(self, 
                 roi_size: Union[Tuple, List], 
                 normalize: bool,
                 coords_dim: int,
                 simplify: bool=False, 
                 sample_num: int=-1, 
                 sample_dist: float=-1, 
        ):
        self.coords_dim = coords_dim
        self.sample_num = sample_num
        self.sample_dist = sample_dist
        self.roi_size = np.array(roi_size)
        self.normalize = normalize
        self.simplify = simplify
        self.sample_fn = None

        if sample_dist > 0:
            assert sample_num < 0 and not simplify
            self.sample_fn = self.interp_fixed_dist
        if sample_num > 0:
            assert sample_dist < 0 and not simplify
            self.sample_fn = self.interp_fixed_num

    def interp_fixed_num(self, line: LineString) -> NDArray:
        ''' Interpolate a line to fixed number of points.
        
        Args:
            line (LineString): line
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''

        distances = np.linspace(0, line.length, self.sample_num)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
            for distance in distances]).squeeze()

        return sampled_points

    def interp_fixed_dist(self, line: LineString) -> NDArray:
        ''' Interpolate a line at fixed interval.
        
        Args:
            line (LineString): line
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''

        distances = list(np.arange(self.sample_dist, line.length, self.sample_dist))
        # make sure to sample at least two points when sample_dist > line.length
        distances = [0,] + distances + [line.length,] 
        
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points
    
    def get_vectorized_lines(self, map_geoms: Dict) -> Dict:
        ''' Vectorize map elements. Iterate over the input dict and apply the 
        specified sample funcion.
        
        Args:
            line (LineString): line
        
        Returns:
            vectors (array): dict of vectorized map elements.
        '''

        vectors = {}
        for label, geom_list in map_geoms.items():
            vectors[label] = []
            for geom in geom_list:
                if geom.geom_type == 'LineString':
                    geom = LineString(np.array(geom.coords)[:, :self.coords_dim])
                    if self.simplify:
                        line = geom.simplify(0.2, preserve_topology=True)
                        line = np.array(line.coords)
                    elif self.sample_fn:
                        line = self.sample_fn(geom)
                    else:
                        line = np.array(line.coords)

                    if self.normalize:
                        line = self.normalize_line(line)
                    vectors[label].append(line)

                elif geom.geom_type == 'Polygon':
                    # polygon objects will not be vectorized
                    continue
                
                else:
                    raise ValueError('map geoms must be either LineString or Polygon!')
        return vectors
    
    def normalize_line(self, line: NDArray) -> NDArray:
        ''' Convert points to range (0, 1).
        
        Args:
            line (LineString): line
        
        Returns:
            normalized (array): normalized points.
        '''

        origin = -np.array([self.roi_size[0]/2, self.roi_size[1]/2])

        line[:, :2] = line[:, :2] - origin

        # transform from range [0, 1] to (0, 1)
        eps = 2
        line[:, :2] = line[:, :2] / (self.roi_size + eps)

        return line
    
    def __call__(self, input_dict):
        map_geoms = input_dict['map_geoms']

        input_dict['vectors'] = self.get_vectorized_lines(map_geoms)
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(simplify={self.simplify}, '
        repr_str += f'sample_num={self.sample_num}), '
        repr_str += f'sample_dist={self.sample_dist}), ' 
        repr_str += f'roi_size={self.roi_size})'
        repr_str += f'normalize={self.normalize})'
        repr_str += f'coords_dim={self.coords_dim})'

        return repr_str