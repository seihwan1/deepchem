"""
Tests for binding pocket detection.
"""
import sys
import logging
import unittest
import os
import numpy as np
import deepchem as dc
from nose.tools import nottest
from deepchem.utils import rdkit_util
from deepchem.utils import coordinate_box_utils as box_utils

logger = logging.getLogger(__name__)


class TestBindingPocket(unittest.TestCase):
  """
  Does sanity checks on binding pocket generation.
  """

  def test_convex_init(self):
    """Tests that ConvexHullPocketFinder can be initialized."""
    finder = dc.dock.ConvexHullPocketFinder()

  def test_get_face_boxes_for_protein(self):
    """Tests that binding pockets are detected."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")
    coords = rdkit_util.load_molecule(protein_file)[0]

    boxes = box_utils.get_face_boxes(coords)
    assert isinstance(boxes, list)
    # Pocket is of form ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    for pocket in boxes:
      assert isinstance(pocket, box_utils.CoordinateBox)

  def test_boxes_to_atoms(self):
    """Test that mapping of protein atoms to boxes is meaningful."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")
    coords = rdkit_util.load_molecule(protein_file)[0]
    boxes = box_utils.get_face_boxes(coords)

    mapping = dc.dock.binding_pocket.boxes_to_atoms(coords, boxes)
    assert isinstance(mapping, dict)
    for box, box_atoms in mapping.items():
      (x_min, x_max), (y_min, y_max), (z_min, z_max) = box.x_range, box.y_range, box.z_range
      for atom_ind in box_atoms:
        atom = coords[atom_ind]
        assert x_min <= atom[0] and atom[0] <= x_max
        assert y_min <= atom[1] and atom[1] <= y_max
        assert z_min <= atom[2] and atom[2] <= z_max

  def test_convex_find_pockets(self):
    """Test that some pockets are filtered out."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    import mdtraj as md
    protein = md.load(protein_file)

    finder = dc.dock.ConvexHullPocketFinder()
    all_pockets = finder.find_all_pockets(protein_file)
    pockets = finder.find_pockets(
        protein_file, ligand_file)
    # Test that every atom in pocket maps exists
    n_protein_atoms = protein.xyz.shape[1]
    for pocket in pockets:
      assert isinstance(pocket, box_utils.CoordinateBox)

    assert len(pockets) < len(all_pockets)

  def test_extract_active_site(self):
    """Test that computed pockets have strong overlap with true binding pocket."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    active_site_box, active_site_coords = (
        dc.dock.binding_pocket.extract_active_site(protein_file, ligand_file))
    finder = dc.dock.ConvexHullPocketFinder()
    pockets = finder.find_pockets(protein_file, ligand_file)

    assert len(pockets) > 0
