"""
Tests for Pose Generation
"""
import os
import sys
import unittest
import logging
import numpy as np
import deepchem as dc
from nose.plugins.attrib import attr
from deepchem.dock.binding_pocket import ConvexHullPocketFinder


class TestPoseGeneration(unittest.TestCase):
  """
  Does sanity checks on pose generation.
  """

  @attr("slow")
  def test_vina_initialization(self):
    """Test that VinaPoseGenerator can be initialized."""
    vpg = dc.dock.VinaPoseGenerator()

  @attr("slow")
  def test_pocket_vina_initialization(self):
    """Test that VinaPoseGenerator can be initialized."""
    pocket_finder = ConvexHullPocketFinder()
    vpg = dc.dock.VinaPoseGenerator(pocket_finder=pocket_finder)

  @attr("slow")
  def test_vina_poses(self):
    """Test that VinaPoseGenerator creates pose files.

    This test takes some time to run, about a minute and a half on
    development laptop.
    """
    # Let's turn on logging since this test will run for a while
    logging.basicConfig(level=logging.INFO)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    vpg = dc.dock.VinaPoseGenerator(pocket_finder=None)
    poses = vpg.generate_poses(
        (protein_file, ligand_file),
        exhaustiveness=1,
        num_modes=1,
        out_dir="/tmp")

    assert len(poses) == 1
    protein, ligand = poses[0]
    from rdkit import Chem
    assert isinstance(protein, Chem.Mol)
    assert isinstance(ligand, Chem.Mol)

  @attr("slow")
  def test_vina_pose_specified_centroid(self):
    """Test that VinaPoseGenerator creates pose files with specified centroid/box dims.

    This test takes some time to run, about a minute and a half on
    development laptop.
    """
    # Let's turn on logging since this test will run for a while
    logging.basicConfig(level=logging.INFO)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    centroid = np.array([56.21891368, 25.95862964, 3.58950065])
    box_dims = np.array([51.354, 51.243, 55.608])
    vpg = dc.dock.VinaPoseGenerator(pocket_finder=None)
    poses = vpg.generate_poses(
        (protein_file, ligand_file),
        centroid=centroid,
        box_dims=box_dims,
        exhaustiveness=1,
        num_modes=1,
        out_dir="/tmp")

    assert len(poses) == 1
    protein, ligand = poses[0]
    from rdkit import Chem
    assert isinstance(protein, Chem.Mol)
    assert isinstance(ligand, Chem.Mol)

  @attr('slow')
  def test_pocket_vina_poses(self):
    """Test that VinaPoseGenerator creates pose files.

    This test is quite slow and takes about 5 minutes to run on a
    development laptop.
    """
    # Let's turn on logging since this test will run for a while
    logging.basicConfig(level=logging.INFO)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    # Note this may download autodock Vina...
    convex_finder = dc.dock.ConvexHullPocketFinder()
    vpg = dc.dock.VinaPoseGenerator(pocket_finder=convex_finder)
    poses = vpg.generate_poses(
        (protein_file, ligand_file),
        exhaustiveness=1,
        num_modes=1,
        num_pockets=2,
        out_dir="/tmp")

    assert len(poses) == 2
    from rdkit import Chem
    for pose in poses:
      protein, ligand = pose
      assert isinstance(protein, Chem.Mol)
      assert isinstance(ligand, Chem.Mol)
