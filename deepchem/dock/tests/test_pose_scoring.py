"""
Tests for Pose Scoring
"""
__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import sys
import logging
import unittest
import tempfile
import os
import shutil
import numpy as np
import deepchem as dc
from subprocess import call
from deepchem.utils import download_url
from deepchem.utils import get_data_dir
from deepchem.dock.pose_scoring import cutoff_filter

logger = logging.getLogger(__name__)


class TestPoseScoring(unittest.TestCase):
  """
  Does sanity checks on pose generation.
  """

  def setUp(self):
    """Downloads dataset."""
    download_url(
        "http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/core_grid.json"
    )
    json_fname = os.path.join(get_data_dir(), 'core_grid.json')
    self.core_dataset = dc.data.NumpyDataset.from_json(json_fname)

  def test_cutoff(self):
    N = 10
    M = 5
    d = np.ones((N, M))
    x = np.random.rand(N, M)
    cutoff_dist = 0.5
    x_thres = cutoff_filter(d, x, cutoff=cutoff_dist)
    ###################################
    print("x_thres")
    print(x_thres)
    ###################################
    assert (x_thres == np.zeros((N, M))).all()

  #def test_pose_scorer_init(self):
  #  """Tests that pose-score works."""
  #  sklearn_model = RandomForestRegressor(n_estimators=10)
  #  model = dc.models.SklearnModel(sklearn_model)
  #  logger.info("About to fit model on core set")
  #  model.fit(self.core_dataset)

  #  pose_scorer = dc.dock.GridPoseScorer(model, feat="grid")

  #def test_pose_scorer_score(self):
  #  """Tests that scores are generated"""
  #  current_dir = os.path.dirname(os.path.realpath(__file__))
  #  protein_file = os.path.join(current_dir, "1jld_protein.pdb")
  #  ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

  #  sklearn_model = RandomForestRegressor(n_estimators=10)
  #  model = dc.models.SklearnModel(sklearn_model)
  #  logger.info("About to fit model on core set")
  #  model.fit(self.core_dataset)

  #  pose_scorer = dc.dock.GridPoseScorer(model, feat="grid")
  #  score = pose_scorer.score(protein_file, ligand_file)
  #  assert score.shape == (1,)
