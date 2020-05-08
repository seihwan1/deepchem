#"""
#Scores protein-ligand poses using DeepChem.
#"""
#from deepchem.feat import RdkitGridFeaturizer
#
#__author__ = "Bharath Ramsundar"
#__copyright__ = "Copyright 2016, Stanford University"
#__license__ = "MIT"
#
#import numpy as np
#import os
#import tempfile
#from deepchem.data import NumpyDataset
#from subprocess import call
#
import numpy as np


def pairwise_distances(coords1, coords2):
  """Returns matrix of pairwise Euclidean distances.

  Parameters
  ----------
  coords1: jax.np.ndarray
    Of shape `(N, 3)`
  coords2: jax.np.ndarray
    Of shape `(M, 3)`

  Returns
  -------
  A `(N,M)` array with pairwise distances.
  """
  return np.sum((coords1[None,:] - coords2[:, None])**2, -1)**0.5

def cutoff_filter(d, x, cutoff=8.0):
  """Applies a cutoff filter on pairwise distances

  Parameters
  ----------
  d: np.ndarray
    Pairwise distances matrix. Of shape `(N, M)` 
  x: np.ndarray
    Matrix of shape `(N, M)` 
  cutoff: float, optional (default 8)
    Cutoff for selection in Angstroms

  Returns
  -------
  A `(N,M)` array with values where distance is too large thresholded
  to 0.
  """
  return np.where(d < cutoff, x, np.zeros_like(x))

def vina_nonlinearity(c, w, Nrot):
  """Computes non-linearity used in Vina.

  Parameters
  ----------
  c: np.ndarray 
    Of shape `(N, M)` 
  w: float
    Weighting term
  Nrot: int
    Number of rotatable bonds in this molecule

  Returns
  -------
  A `(N, M)` array with activations under a nonlinearity.
  """
  out_tensor = c / (1 + w * Nrot)
  return out_tensor

def vina_repulsion(d):
  """Computes Autodock Vina's repulsion interaction term.

  Parameters
  ----------
  d: jax.np.ndarray
    Of shape `(N, M)`.

  Returns
  -------
  A `(N, M)` array with repulsion terms.
  """
  return np.where(d >= 0, d**2, np.zeros_like(d))

def vina_hydrophobic(d):
  """Computes Autodock Vina's hydrophobic interaction term.

  Parameters
  ----------
  d: jax.np.ndarray
    Of shape `(N, M)`.

  Returns
  -------
  A `(N, M)` array of hydrophoboic interactions in a piecewise linear
  curve.
  """
  out_tensor = np.where(d < 0.5, np.ones_like(d),
                        np.where(d < 1.5, 1.5 - d, np.zeros_like(d)))
  return out_tensor


def vina_hbond(d):
  """Computes Autodock Vina's hydrogen bond interaction term.

  Parameters
  ----------
  d: jax.np.ndarray
    Of shape `(N, M)`.

  Returns
  -------
  A `(N, M)` array of hydrophoboic interactions in a piecewise linear
  curve.
  """
  out_tensor = np.where(
      d < -0.7, np.ones_like(d),
      np.where(d < 0, (1.0 / 0.7) * (0 - d), np.zeros_like(d)))
  return out_tensor

def gaussian_first(d):
  """Computes Autodock Vina's first Gaussian interaction term.

  Parameters
  ----------
  d: jax.np.ndarray
    Of shape `(N, M)`.

  Returns
  -------
  A `(N, M)` array of gaussian interaction terms.
  """
  out_tensor = np.exp(-(d / 0.5)**2)
  return out_tensor

def gaussian_second(d):
  """Computes Autodock Vina's second Gaussian interaction term.

  Parameters
  ----------
  d: jax.np.ndarray
    Of shape `(N, M)`.

  Returns
  -------
  A `(N, M)` array of gaussian interaction terms.
  """
  out_tensor = np.exp(-((d - 3) / 2)**2)
  return out_tensor


class VinaFreeEnergy(object):
  """Uses Numpy to compute the VinaFreeEnergy
  """

  def __init__(self,
               N_atoms,
               #M_nbrs,
               ndim,
               nbr_cutoff,
               start,
               stop,
               stddev=.3,
               Nrot=1,
               **kwargs):
    super(VinaFreeEnergy, self).__init__(**kwargs)
    self.stddev = stddev
    # Number of rotatable bonds
    # TODO(rbharath): Vina actually sets this per-molecule. See if makes
    # a difference.
    self.Nrot = Nrot
    self.N_atoms = N_atoms
    #self.M_nbrs = M_nbrs
    self.ndim = ndim
    self.nbr_cutoff = nbr_cutoff
    self.start = start
    self.stop = stop

  #def build(self, input_shape):
  #  self.weighted_combo = WeightedLinearCombo()
  #  self.w = tf.Variable(tf.random.normal((1,), stddev=self.stddev))
  #  self.built = True

def weighted_linear_sum(w, x):
  """Computes weighted linear sum.

  Parameters
  ----------
  w: jax.np.ndarray
    Of shape `(N,)`
  x: jax.np.ndarray
    Of shape `(N,)`
  """
  return np.sum(np.dot(w, x))

def vina_energy_term(coords1, coords2, weights, wrot):
  """
  Parameters
  ----------
  coords1: jax.np.ndarray 
    Molecular coordinates of shape `(N, 3)`
  coords2: jax.np.ndarray 
    Molecular coordinates of shape `(M, 3)`
  weights: jax.np.ndarray
    Of shape `(5,)`
  wrot: float
    The scaling factor for nonlinearity

  Returns
  -------
  Scalar of energy
  """
  #X = inputs[0]
  #Z = inputs[1]

  ## TODO(rbharath): This layer shouldn't be neighbor-listing. Make
  ## neighbors lists an argument instead of a part of this layer.
  #nbr_list = NeighborList(self.N_atoms, self.M_nbrs, self.ndim,
  #                        self.nbr_cutoff, self.start, self.stop)(X)

  #dists = InteratomicL2Distances(self.N_atoms, self.M_nbrs,
  #                               self.ndim)([X, nbr_list])

  ## Shape (N, M)

  # TODO(rbharath): The autodock vina source computes surface distances which take into account the van der Waals radius of each atom type.
  dists = pairwise_distances(coords1, coords2)
  repulsion = vina_repulsion(dists)
  hydrophobic = vina_hydrophobic(dists)
  hbond = vina_hbond(dists)
  gauss_1 = vina_gaussian_first(dists)
  gauss_2 = vina_gaussian_second(dists)

  # Shape (N, M)
  interactions = weighted_linear_sum(
      weights, np.array([repulsion, hydrophobic, hbond, gauss_1, gauss_2]))

  # Shape (N, M)
  thresholded = cutoff_filter(dists, interactions)

  free_energies = vina_nonlinearity(thresholded, wrot)
  #return tf.reduce_sum(free_energies)
  return np.sum(free_energies)
