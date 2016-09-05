from __future__ import print_function

from collections import namedtuple
from ctypes import (c_int, c_float, c_double, c_void_p, c_bool,
                    POINTER, CFUNCTYPE, cast)

import numpy as np
import numpy.ctypeslib as npc

from .vl_ctypes import (LIB, CustomStructure, Enum,
                        vl_type, vl_size, np_to_c_types, c_to_vl_types)
from .utils import is_integer


class VectorComparisonType(Enum):
    DISTANCE_L1 = 0
    DISTANCE_L2 = 1
    DISTANCE_CHI2 = 2
    DISTANCE_HELLINGER = 3
    DISTANCE_JS = 4
    DISTANCE_MAHALANOBIS = 5
    KERNEL_L1 = 6
    KERNEL_L2 = 7
    KERNEL_CHI2 = 8
    KERNEL_HELLINGER = 9
    KERNEL_JS = 10

FloatVectorComparisonFunction = POINTER(CFUNCTYPE(
    c_float, vl_size, POINTER(c_float), POINTER(c_float)))
DoubleVectorComparisonFunction = POINTER(CFUNCTYPE(
    c_double, vl_size, POINTER(c_double), POINTER(c_double)))


class KMeansAlgorithm(Enum):
    LLOYD = 0
    ELKAN = 1
    ANN = 2


class KMeansInitialization(Enum):
    RANDOM = 0
    PLUS_PLUS = 1


class VlKMeans(CustomStructure):
    _fields_ = [
        ('dataType', vl_type),
        ('dimension', vl_size),
        ('numCenters', vl_size),
        ('numTrees', vl_size),
        ('maxNumComparisons', vl_size),

        ('initialization', KMeansInitialization),
        ('algorithm', KMeansAlgorithm),
        ('distance', VectorComparisonType),
        ('maxNumIterations', vl_size),
        ('minEnergyVariation', c_double),
        ('numRepetitions', vl_size),
        ('verbosity', c_int),

        ('centers', c_void_p),
        ('centerDistances', c_void_p),

        ('energy', c_double),
        ('floatVectorComparisonFn', FloatVectorComparisonFunction),
        ('doubleVectorComparisonFn', DoubleVectorComparisonFunction),
    ]
VlKMeans_p = POINTER(VlKMeans)

################################################################################
### functions in the SO

# create and destroy
vl_kmeans_new = LIB['vl_kmeans_new']
vl_kmeans_new.restype = VlKMeans_p
vl_kmeans_new.argtypes = [vl_type, VectorComparisonType]

vl_kmeans_new_copy = LIB['vl_kmeans_new_copy']
vl_kmeans_new_copy.restype = VlKMeans_p
vl_kmeans_new_copy.argtypes = [VlKMeans_p]

vl_kmeans_delete = LIB['vl_kmeans_delete']
vl_kmeans_delete.restype = None
vl_kmeans_delete.argtypes = [VlKMeans_p]

# basic data processing
vl_kmeans_reset = LIB['vl_kmeans_reset']
vl_kmeans_reset.restype = None
vl_kmeans_reset.argtypes = [VlKMeans_p]

vl_kmeans_cluster = LIB['vl_kmeans_cluster']
vl_kmeans_cluster.restype = c_double
vl_kmeans_cluster.argtypes = [VlKMeans_p, c_void_p, vl_size, vl_size, vl_size]

vl_kmeans_quantize = LIB['vl_kmeans_quantize']
vl_kmeans_quantize.restype = None
vl_kmeans_quantize.argtypes = [
    VlKMeans_p, npc.ndpointer(dtype=np.uint32), c_void_p, c_void_p, vl_size]

vl_kmeans_quantize_ann = LIB['vl_kmeans_quantize_ann']
vl_kmeans_quantize_ann.restype = None
vl_kmeans_quantize_ann.argtypes = [
    VlKMeans_p, npc.ndpointer(dtype=np.uint32), c_void_p, c_void_p, vl_size,
    c_bool]

# advanced data processing
vl_kmeans_set_centers = LIB['vl_kmeans_set_centers']
vl_kmeans_set_centers.restype = None
vl_kmeans_set_centers.argtypes = [VlKMeans_p, c_void_p, vl_size, vl_size]

vl_kmeans_init_centers_with_rand_data = \
    LIB['vl_kmeans_init_centers_with_rand_data']
vl_kmeans_init_centers_with_rand_data.restype = None
vl_kmeans_init_centers_with_rand_data.argtypes = [
    VlKMeans_p, c_void_p, vl_size, vl_size, vl_size]

vl_kmeans_init_centers_plus_plus = LIB['vl_kmeans_init_centers_plus_plus']
vl_kmeans_init_centers_plus_plus.restype = None
vl_kmeans_init_centers_plus_plus.argtypes = [
    VlKMeans_p, c_void_p, vl_size, vl_size, vl_size]

vl_kmeans_refine_centers = LIB['vl_kmeans_refine_centers']
vl_kmeans_refine_centers.restype = c_double
vl_kmeans_refine_centers.argtypes = [VlKMeans_p, c_void_p, vl_size]


def _check_integer(x, name, lower=None, upper=None):
    if not is_integer(x):
        raise TypeError("{} must be an integer".format(name))
    if lower is not None and x < lower:
        raise ValueError("{} must be at least {}".format(name, lower))
    if upper is not None and x > upper:
        raise ValueError("{} must be no more than {}".format(name, upper))


def vl_kmeans(data, num_centers,
              algorithm='lloyd', initialization='plus_plus', distance='l2',
              max_iter=100, num_rep=1, verbosity=0,
              quantize=False, ret_energy=False):
    data = np.asarray(data)
    c_dtype = np_to_c_types.get(data.dtype, None)
    if c_dtype not in [c_float, c_double]:
        raise TypeError("data should be float32 or float64")
    vl_dtype = c_to_vl_types[c_dtype]

    if data.ndim != 2:
        raise TypeError("data must be num_data x dim")
    num_data, dim = data.shape
    if dim == 0:
        raise ValueError("data dimension is zero")

    _check_integer(num_centers, "num_centers", 0, num_data)
    _check_integer(num_rep, "num_rep", 1)
    _check_integer(verbosity, "verbosity", 0)
    _check_integer(max_iter, "max_iter", 0)

    algorithm = KMeansAlgorithm._members[algorithm.upper()]
    initialization = KMeansInitialization._members[initialization.upper()]
    distance = VectorComparisonType._members['DISTANCE_' + distance.upper()]

    kmeans_p = vl_kmeans_new(vl_dtype, distance)

    kmeans = kmeans_p[0]
    try:
        kmeans.verbosity = verbosity
        kmeans.numRepetitions = num_rep
        kmeans.algorithm = algorithm
        kmeans.initialization = initialization
        kmeans.maxNumIterations = max_iter

        if verbosity:
            pr = lambda *a, **k: print('kmeans:', *a, **k)
            pr("Initialization = {}".format(kmeans.initialization.name))
            pr("Algorithm = {}".format(kmeans.algorithm.name))
            pr("MaxNumIterations = {}".format(kmeans.maxNumIterations))
            pr("NumReptitions = {}".format(kmeans.numRepetitions))
            pr("data type = {}".format(kmeans.dataType.name))
            pr("distance = {}".format(kmeans.distance.name))
            pr("data dimension = {}".format(dim))
            pr("num. data points = {}".format(num_data))
            pr("num. centers = {}".format(num_centers))
            print()

        data_p = data.ctypes.data_as(c_void_p)
        energy = vl_kmeans_cluster(kmeans_p, data_p, dim, num_data, num_centers)

        # copy out the centers
        centers_p = cast(kmeans.centers, POINTER(c_dtype))
        centers = np.ctypeslib.as_array(centers_p, (num_centers, dim)).copy()

        ret = [centers]
        ret_fields = ['centers']

        if quantize:
            assignments = np.empty(num_data, dtype=np.uint32)
            vl_kmeans_quantize(kmeans_p, assignments, None, data_p, num_data)

            ret.append(assignments)
            ret_fields.append('assignments')

        if ret_energy:
            ret.append(energy)
            ret_fields.append('energy')

        if not quantize and not ret_energy:
            return centers
        return namedtuple('KMeansRetVal', ret_fields)(*ret)

    finally:
        vl_kmeans_delete(kmeans_p)



class KMeans(object):
    def __init__(self, num_centers, dtype=np.float32,
        algorithm='lloyd', initialization='plus_plus', distance='l2',
        max_iter=100, num_rep=1, verbosity=0):

        _check_integer(num_rep, "num_rep", 1)
        _check_integer(verbosity, "verbosity", 0)
        _check_integer(max_iter, "max_iter", 0)

        algorithm = KMeansAlgorithm._members[algorithm.upper()]
        initialization = KMeansInitialization._members[initialization.upper()]
        distance = VectorComparisonType._members['DISTANCE_' + distance.upper()]

        dtype = np.dtype(dtype)
        c_dtype = np_to_c_types.get(dtype, None)
        if c_dtype not in [c_float, c_double]:
            raise TypeError("data should be float32 or float64")
        self.c_dtype = c_dtype
        self.vl_dtype = c_to_vl_types[c_dtype]

        self.num_centers = num_centers
        self.kmeans_p = vl_kmeans_new(self.vl_dtype, distance)
        self.kmeans = self.kmeans_p[0]
        self.kmeans.verbosity = verbosity
        self.kmeans.numRepetitions = num_rep
        self.kmeans.algorithm = algorithm
        self.kmeans.initialization = initialization
        self.kmeans.maxNumIterations = max_iter

    def _check_data(self, data):
        data = np.asarray(data)
        c_dtype = np_to_c_types.get(data.dtype, None)
        if c_dtype != self.c_dtype:
            raise TypeError("different dtype")
        if data.ndim != 2:
            raise TypeError("data must be num_data x dim")
        num_data, dim = data.shape
        if dim == 0:
            raise ValueError("data dimension is zero")
        return data

    def fit(self, data):
        data = self._check_data(data)
        num_data, dim = data.shape

        data_p = data.ctypes.data_as(c_void_p)
        self.energy = vl_kmeans_cluster(self.kmeans_p, data_p,
            data.shape[1], data.shape[0], self.num_centers)

        # copy out the centers
        centers_p = cast(self.kmeans.centers, POINTER(self.c_dtype))
        self.centers = np.ctypeslib.as_array(centers_p,
            (self.num_centers, self.kmeans.dimension)).copy()

        return self

    def transform(self, data):
        data = self._check_data(data)
        assignments = np.empty(data.shape[0], dtype=np.uint32)
        data_p = data.ctypes.data_as(c_void_p)
        vl_kmeans_quantize(self.kmeans_p, assignments,
            None, data_p, data.shape[0])
        return assignments

    def fit_transform(self, data):
        return self.fit(data).transform(data)
