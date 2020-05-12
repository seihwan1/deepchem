"""
Docks Molecular Complexes 
"""
import logging
import numpy as np
import os
import tempfile
from subprocess import call

logger = logging.getLogger(__name__)


class Docker(object):
  """A generic molecular docking class

  This class provides a docking engine which uses provided models for
  featurization, pose generation, and scoring. Most pieces of dockign
  software are command line tools that are invoked from the shell. The
  goal of this class is to provide a python clean API for invoking
  molecular docking programmatically.
  """

  def __init__(self, scoring_model, featurizer, pose_generator, exhaustiveness=10, detect_pockets=False):
    """Builds model.

    Parameters
    ----------
    scoring_model: `Model`
      Should make predictions on molecular complex.
    featurizer: `ComplexFeaturizer`
      Featurizer associated with `scoring_model`
    pose_generator: `PoseGenerator`
      The pose generator to use for this model
    """
    self.base_dir = tempfile.mkdtemp()
    self.pose_generator = pose_generator

  def dock(self,
           molecular_complex,
           centroid=None,
           box_dims=None,
           dry_run=False):
    """Docks using Vina and RF.

    Parameters
    ----------
    molecular_complex: Object
      Some representation of a molecular complex.
    """
    for posed_compled in self.pose_generator.generate_poses(molecular_complex):
      # TODO: How to handle the failure here?
      features, _ = self.featurizer.featurize_complexes([molecular_complex])
      dataset = NumpyDataset(X=features)
      score = self.model.predict(dataset)
      yield (score, posed_complex)
