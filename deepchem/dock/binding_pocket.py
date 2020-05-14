"""
Computes putative binding pockets on protein.
"""
import os
import logging
import tempfile
import numpy as np
from subprocess import call
from deepchem.feat.fingerprints import CircularFingerprint
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils import rdkit_util
from deepchem.utils import coordinate_box_utils as box_utils

logger = logging.getLogger(__name__)


def extract_active_site(protein_file, ligand_file, cutoff=4):
  """Extracts a box for the active site.

  Params
  ------
  protein_file: str
    Location of protein PDB
  ligand_file: str
    Location of ligand input file
  cutoff: int, optional
    The distance in angstroms from the protein pocket to
    consider for featurization.
  """
  protein = rdkit_util.load_molecule(
      protein_file, add_hydrogens=False)[0]
  ligand = rdkit_util.load_molecule(
      ligand_file, add_hydrogens=True, calc_charges=True)[0]
  contact_atoms = get_contact_atom_indices([protein, ligand], cutoff=cutoff)
  n_contact_atoms = len(contact_atoms)
  protein_coords = protein[0]
  pocket_coords = protein_coords[contact_atoms]

  x_min = int(np.floor(np.amin(pocket_coords[:, 0])))
  x_max = int(np.ceil(np.amax(pocket_coords[:, 0])))
  y_min = int(np.floor(np.amin(pocket_coords[:, 1])))
  y_max = int(np.ceil(np.amax(pocket_coords[:, 1])))
  z_min = int(np.floor(np.amin(pocket_coords[:, 2])))
  z_max = int(np.ceil(np.amax(pocket_coords[:, 2])))
  box = CoordinateBox((x_min, x_max), (y_min, y_max), (z_min, z_max))
  return (box, pocket_atoms, pocket_coords)

def boxes_to_atoms(coords, boxes):
  """Maps each box to a list of atoms in that box.

  Parameters
  ----------
  coords: np.ndarray
    Of shape `(N, 3)
  boxes: list
    list of `CoordinateBox` objects.

  Returns
  -------
  dictionary mapping `CoordinateBox` objects to lists of atom coordinates
  """
  mapping = {}
  for box_ind, box in enumerate(boxes):
    box_atoms = []
    logger.info("Handing box %d/%d" % (box_ind, len(boxes)))
    for atom_ind in range(len(atom_coords)):
      atom = atom_coords[atom_ind]
      if atom in box:
        box_atoms.append(atom_ind)
    mapping[box] = box_atoms
  return mapping

class BindingPocketFinder(object):
  """Abstract superclass for binding pocket detectors

  Many times when working with a new protein or other macromolecule,
  it's not clear what zones of the macromolecule may be good targets
  for potential ligands or other molecules to interact with. This
  abstract class provides a template for child classes that
  algorithmically locate potential binding pockets that are good
  potential interaction sites.

  Note that potential interactions sites can be found by many
  different methods, and that this abstract class doesn't specify the
  technique to be used.
  """

  def find_pockets(self, molecule):
    """Finds potential binding pockets in proteins.

    Parameters
    ----------
    molecule: object
      Some representation of a molecule.
    """
    raise NotImplementedError


class ConvexHullPocketFinder(BindingPocketFinder):
  """Implementation that uses convex hull of protein to find pockets.

  Based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4112621/pdf/1472-6807-14-18.pdf
  """

  def __init__(self, scoring_model=None, pocket_featurizer=None, ligand_featurizer=None, pad=5):
    """Initialize the pocket finder.

    Parameters
    ----------
    scoring_model: `dc.models.Model`, optional
      If specified, use this model to prune pockets.
    pocket_featurizer: `dc.feat.BindingPocketFeaturizer`, optional
      If `scoring_model` is specified, `pocket_featurizer` must be as
      well. This featurizer is used to featurize binding pockets.
    ligand_featurizer: `dc.feat.MolecularFeaturizer`, optional
      Featurizer used to featurize the ligand.
    pad: float, optional
      The number of angstroms to pad around a binding pocket's atoms
      to get a binding pocket box.
    """
    self.scoring_model = scoring_model
    self.pocket_featurizer = pocket_featurizer
    self.ligand_featurizer = ligand_featurizer
    self.pad = pad

  def find_all_pockets(self, protein_file):
    """Find list of binding pockets on protein.
    
    Parameters
    ----------
    protein_file: str
      Protein to load in.
    """
    protein = rdkit_util.load_molecule(protein_file)[0]
    return get_all_boxes(protein, self.pad)

  def find_pockets(self, protein_file, ligand_file):
    """Find list of suitable binding pockets on protein.


    TODO: What is a pocket? Maybe this should be a class so it's a
    clean API. Returning a tuple feels kludgey.

    Params
    ------
    protein_file: str
      Location of the PDB file to load
    ligand_file: str
      Location of the ligand file to load

    Returns
    -------
    List of pockets. Each pocket is a `CoordinateBox`
    """
    protein_coords = rdkit_util.load_molecule(
        protein_file, add_hydrogens=False, calc_charges=False)[0]
    ligand_coords = rdkit_util.load_molecule(
        ligand_file, add_hydrogens=False, calc_charges=False)[0]
    boxes = get_all_boxes(protein_coords, self.pad)
    boxes = merge_overlapping_boxes(boxes)
    mapping = boxes_to_atoms(protein_coords, boxes)
    pocket_coords = []
    for box in boxes:
      atoms = mapping[pocket]
      coords = protein_coords[atoms]
      pocket_coords.append(coords)
    return boxes, pocket_coords


#class RFConvexHullPocketFinder(BindingPocketFinder):
#  """Uses pre-trained RF model + ConvexHulPocketFinder to select pockets."""
#
#  def __init__(self, pad=5):
#    self.pad = pad
#    self.convex_finder = ConvexHullPocketFinder(pad)
#
#    # Load binding pocket model
#    # Fit model on dataset
#    self.model = SklearnModel(model_dir=self.model_dir)
#    self.model.reload()
#
#    # Create featurizers
#    self.pocket_featurizer = BindingPocketFeaturizer()
#    self.ligand_featurizer = CircularFingerprint(size=1024)
#
#  def find_pockets(self, protein_file, ligand_file):
#    """Compute features for a given complex
#
#    TODO(rbharath): This has a log of code overlap with
#    compute_binding_pocket_features in
#    examples/binding_pockets/binding_pocket_datasets.py. Find way to refactor
#    to avoid code duplication.
#    """
#    # if not ligand_file.endswith(".sdf"):
#    #   raise ValueError("Only .sdf ligand files can be featurized.")
#    # ligand_basename = os.path.basename(ligand_file).split(".")[0]
#    # ligand_mol2 = os.path.join(
#    #     self.base_dir, ligand_basename + ".mol2")
#    #
#    # # Write mol2 file for ligand
#    # obConversion = ob.OBConversion()
#    # conv_out = obConversion.SetInAndOutFormats(str("sdf"), str("mol2"))
#    # ob_mol = ob.OBMol()
#    # obConversion.ReadFile(ob_mol, str(ligand_file))
#    # obConversion.WriteFile(ob_mol, str(ligand_mol2))
#    #
#    # # Featurize ligand
#    # mol = Chem.MolFromMol2File(str(ligand_mol2), removeHs=False)
#    # if mol is None:
#    #   return None, None
#    # # Default for CircularFingerprint
#    # n_ligand_features = 1024
#    # ligand_features = self.ligand_featurizer.featurize([mol])
#    #
#    # # Featurize pocket
#    # pockets, pocket_atoms_map, pocket_coords = self.convex_finder.find_pockets(
#    #     protein_file, ligand_file)
#    # n_pockets = len(pockets)
#    # n_pocket_features = BindingPocketFeaturizer.n_features
#    #
#    # features = np.zeros((n_pockets, n_pocket_features+n_ligand_features))
#    # pocket_features = self.pocket_featurizer.featurize(
#    #     protein_file, pockets, pocket_atoms_map, pocket_coords)
#    # # Note broadcast operation
#    # features[:, :n_pocket_features] = pocket_features
#    # features[:, n_pocket_features:] = ligand_features
#    # dataset = NumpyDataset(X=features)
#    # pocket_preds = self.model.predict(dataset)
#    # pocket_pred_proba = np.squeeze(self.model.predict_proba(dataset))
#    #
#    # # Find pockets which are active
#    # active_pockets = []
#    # active_pocket_atoms_map = {}
#    # active_pocket_coords = []
#    # for pocket_ind in range(len(pockets)):
#    #   #################################################### DEBUG
#    #   # TODO(rbharath): For now, using a weak cutoff. Fix later.
#    #   #if pocket_preds[pocket_ind] == 1:
#    #   if pocket_pred_proba[pocket_ind][1] > .15:
#    #   #################################################### DEBUG
#    #     pocket = pockets[pocket_ind]
#    #     active_pockets.append(pocket)
#    #     active_pocket_atoms_map[pocket] = pocket_atoms_map[pocket]
#    #     active_pocket_coords.append(pocket_coords[pocket_ind])
#    # return active_pockets, active_pocket_atoms_map, active_pocket_coords
#    # # TODO(LESWING)
#    raise ValueError("Karl Implement")
