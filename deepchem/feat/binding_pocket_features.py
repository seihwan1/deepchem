"""
Featurizes proposed binding pockets.
"""
import numpy as np
import logging
from deepchem.feat import Featurizer

logger = logging.getLogger(__name__)

class BindingPocketFeaturizer(Featurizer):
  """Featurizes binding pockets with information about chemical
  environments.

  In many applications, it's desirable to look at binding pockets on
  macromolecules which may be good targets for potential ligands or
  other molecules to interact with. A `BindingPocketFeaturizer`
  expects to be given a macromolecule, and a list of pockets to
  featurize on that macromolecule. These pockets should be of the form
  produced by a `dc.dock.BindingPocketFinder`, that is as a list of
  `dc.utils.CoordinateBox` objects.

  The base featurization in this class's featurization is currently
  very simple and counts the number of residues of each type present
  in the pocket. It's likely that you'll want to overwrite this
  implementation for more sophisticated downstream usecases.
  """

  residues = [
      "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
      "LEU", "LYS", "MET", "PHE", "PRO", "PYL", "SER", "SEC", "THR", "TRP",
      "TYR", "VAL", "ASX", "GLX"
  ]

  n_features = len(residues)

  def featurize(self,
                protein_file,
                pockets,
                pocket_coords):
    """
    Calculate atomic coodinates.

    Params
    ------
    protein_file: str
      Location of PDB file. Will be loaded by MDTraj
    pockets: list[CoordinateBox]
      List of `dc.utils.CoordinateBox` objects.
    pocket_coords: list
      Coordinates of atoms. Must be of same length as `pockets`.

    Raises
    ------
    `ValueError` if `len(pockets) == len
    """
    if len(
    import mdtraj
    protein = mdtraj.load(protein_file)
    n_pockets = len(pockets)
    n_residues = len(BindingPocketFeaturizer.residues)
    res_map = dict(zip(BindingPocketFeaturizer.residues, range(n_residues)))
    all_features = np.zeros((n_pockets, n_residues))
    for pocket_num, (pocket, coords) in enumerate(zip(pockets, pocket_coords)):
      pocket_atoms = pocket_atoms_map[pocket]
      for ind, atom in enumerate(pocket_atoms):
        atom_name = str(protein.top.atom(atom))
        # atom_name is of format RESX-ATOMTYPE
        # where X is a 1 to 4 digit number
        residue = atom_name[:3]
        if residue not in res_map:
          logger.info("Warning: Non-standard residue in PDB file")
          continue
        atomtype = atom_name.split("-")[1]
        all_features[pocket_num, res_map[residue]] += 1
    return all_features
