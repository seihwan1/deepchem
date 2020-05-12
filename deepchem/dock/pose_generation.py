"""
Generates protein-ligand docked poses using Autodock Vina.
"""
import platform
import deepchem
from deepchem.utils import mol_xyz_util
import logging
import numpy as np
import os
import tempfile
from subprocess import call
from deepchem.utils.rdkit_util import add_hydrogens_to_mol
from subprocess import check_output
from deepchem.utils import rdkit_util
from deepchem.utils import vina_utils

logger = logging.getLogger(__name__)


class PoseGenerator(object):
  """A Pose Generator computes low energy conformations for molecular complexes.

  Many questions in structural biophysics reduce to that of computing
  the binding free energy of molecular complexes. A key step towards
  computing the binding free energy of two complexes is to find low
  energy "poses", that is energetically favorable conformations of
  molecules with respect to each other. One application of this
  technique is to find low energy poses for protein-ligand
  interactions.
  """

  def generate_poses(self, molecular_complex):
    """Generates a list of low energy poses for molecular complex

    Parameters
    ----------
    molecular_complexes: list
      A representation of a molecular complex.

    Returns
    -------
    A list of molecular complexes in a low energy state
    """
    raise NotImplementedError


def _write_vina_conf(protein_filename,
                     ligand_filename,
                     centroid,
                     box_dims,
                     conf_filename,
                     exhaustiveness=None):
  """Writes Vina configuration file to disk.

  Parameters
  ----------
  protein_filename: str
    Filename for protein 
  ligand_filename: str
    Filename for the ligand
  centroid: np.ndarray
    Of shape `(3,)` holding centroid of system
  box_dims: np.ndarray
    Of shape `(3,)` holding the size of the box to dock
  conf_filename: str
    Filename to write Autodock Vina configuration to.
  exhaustiveness: int, optional
    The exhaustiveness of the search to be performed by Vina
  """
  with open(conf_filename, "w") as f:
    f.write("receptor = %s\n" % protein_filename)
    f.write("ligand = %s\n\n" % ligand_filename)

    f.write("center_x = %f\n" % centroid[0])
    f.write("center_y = %f\n" % centroid[1])
    f.write("center_z = %f\n\n" % centroid[2])

    f.write("size_x = %f\n" % box_dims[0])
    f.write("size_y = %f\n" % box_dims[1])
    f.write("size_z = %f\n\n" % box_dims[2])

    if exhaustiveness is not None:
      f.write("exhaustiveness = %d\n" % exhaustiveness)


class VinaPoseGenerator(PoseGenerator):
  """Uses Autodock Vina to generate binding poses.

  This class uses Autodock Vina to make make predictions of
  binding poses. It downloads the Autodock Vina executable for
  your system to your specified DEEPCHEM_DATA_DIR (remember this
  is an environment variable you set) and invokes the executable
  to perform pose generation for you.
  """

  def __init__(self, exhaustiveness=10, sixty_four_bits=True, pocket_finder=None):
    """Initializes Vina Pose Generator

    Params
    ------
    exhaustiveness: int, optional
      Tells Autodock Vina how exhaustive it should be with pose
      generation.
    sixty_four_bits: bool, optional
      Specifies whether this is a 64-bit machine. Needed to download
      the correct executable. 
    pocket_finder: object, optional
      If specified should be an instance of `dc.dock.BindingPocketFinder`.
    """
    data_dir = deepchem.utils.get_data_dir()
    if platform.system() == 'Linux':
      url = "http://vina.scripps.edu/download/autodock_vina_1_1_2_linux_x86.tgz"
      filename = "autodock_vina_1_1_2_linux_x86.tgz" 
      dirname = "autodock_vina_1_1_2_linux_x86"
    elif platform.system() == 'Darwin':
      if sixty_four_bits:
        url = "http://vina.scripps.edu/download/autodock_vina_1_1_2_mac_64bit.tar.gz"
        filename = "autodock_vina_1_1_2_mac_64bit.tar.gz"
        dirname = "autodock_vina_1_1_2_mac_catalina_64bit"
      else:
        url = "http://vina.scripps.edu/download/autodock_vina_1_1_2_mac.tgz"
        filename = "autodock_vina_1_1_2_mac.tgz"
        dirname = "autodock_vina_1_1_2_mac"
    else:
      raise ValueError("This class can only run on Linux or Mac. If you are on Windows, please try using a cloud platform to run this code instead.")
    self.vina_dir = os.path.join(data_dir, dirname)
    self.exhaustiveness = exhaustiveness
    self.pocket_finder = pocket_finder 
    if not os.path.exists(self.vina_dir):
      logger.info("Vina not available. Downloading")
      wget_cmd = "wget -nv -c -T 15 %s" % url 
      #retcode = call(wget_cmd.split(), shell=True)
      check_output(wget_cmd.split())
      logger.info("Downloaded Vina. Extracting")
      untar_cmd = "tar -xzvf %s" % filename
      check_output(untar_cmd.split())
      logger.info("Moving to final location")
      mv_cmd = "mv %s %s" % (dirname, data_dir)
      check_output(mv_cmd.split())
      logger.info("Cleanup: removing downloaded vina tar.gz")
      rm_cmd = "rm %s" % filename
      call(rm_cmd.split())
    self.vina_cmd = os.path.join(self.vina_dir, "bin/vina")

  def generate_poses(self,
                     molecular_complex,
                     centroid=None,
                     box_dims=None,
                     dry_run=False,
                     out_dir=None):
    """Generates the docked complex and outputs files for docked complex.

    Parameters
    ----------
    molecular_complexes: list
      A representation of a molecular complex.
    centroid: np.ndarray, optional
      The centroid to dock against. Is computed if not specified.
    box_dims: np.ndarray, optional
      Of shape `(3,)` holding the size of the box to dock. If not
      specified is set to size of molecular complex plus 5 angstroms.
    dry_run: bool, optional
      If True, don't generate poses, but do setup steps
    out_dir: str, optional
      If specified, write generated poses to this directory.

    Returns
    -------
    List of docked molecular complexes
    """
    if out_dir is None:
      out_dir = tempfile.mkdtemp()

    # Parse complex
    if len(molecular_complex) > 2:
      raise ValueError("Autodock Vina can only dock protein-ligand complexes and not more general molecular complexes.")

    (protein_file, ligand_file) = molecular_complex

    # Prepare protein 
    protein_name = os.path.basename(protein_file).split(".")[0]
    protein_hyd = os.path.join(out_dir, "%s_hyd.pdb" % protein_name)
    protein_pdbqt = os.path.join(out_dir, "%s.pdbqt" % protein_name)
    protein_mol = rdkit_util.load_molecule(
        protein_file, calc_charges=True, add_hydrogens=True)

    # Get protein centroid and range
    if centroid is not None and box_dims is not None:
      protein_centroid = centroid
    else:
      if self.pocket_finder is None:
        rdkit_util.write_molecule(protein_mol[1], protein_hyd, is_protein=True)
        rdkit_util.write_molecule(
            protein_mol[1], protein_pdbqt, is_protein=True)
        protein_centroid = mol_xyz_util.get_molecule_centroid(protein_mol[0])
        protein_range = mol_xyz_util.get_molecule_range(protein_mol[0])
        box_dims = protein_range + 5.0
      else:
        logger.info("About to find putative binding pockets")
        pockets, pocket_atoms_maps, pocket_coords = self.pocket_finder.find_pockets(
            protein_file, ligand_file)
        logger.info("Computing centroid and size of proposed pocket.")
        pocket_coord = pocket_coords[0]
        protein_centroid = np.mean(pocket_coord, axis=1)
        pocket = pockets[0]
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = pocket
        x_box = (x_max - x_min) / 2.
        y_box = (y_max - y_min) / 2.
        z_box = (z_max - z_min) / 2.
        box_dims = (x_box, y_box, z_box)

    # Prepare protein 
    ligand_name = os.path.basename(ligand_file).split(".")[0]
    ligand_pdbqt = os.path.join(out_dir, "%s.pdbqt" % ligand_name)

    ligand_mol = rdkit_util.load_molecule(
        ligand_file, calc_charges=True, add_hydrogens=True)
    rdkit_util.write_molecule(ligand_mol[1], ligand_pdbqt)

    # Write Vina conf file
    conf_file = os.path.join(out_dir, "conf.txt")
    _write_vina_conf(
        protein_pdbqt,
        ligand_pdbqt,
        protein_centroid,
        box_dims,
        conf_file,
        exhaustiveness=self.exhaustiveness)

    # Define locations of log and output files
    log_file = os.path.join(out_dir, "%s_log.txt" % ligand_name)
    out_pdbqt = os.path.join(out_dir, "%s_docked.pdbqt" % ligand_name)
    ########################################################
    print("out_pdbqt")
    print(out_pdbqt)
    ########################################################
    # TODO(rbharath): Let user specify the number of poses required.
    if not dry_run:
      logger.info("About to call Vina")
      call(
          "%s --config %s --log %s --out %s" % (self.vina_cmd, conf_file,
                                                log_file, out_pdbqt),
          shell=True)
    # TODO(rbharath): Convert the output pdbqt to a pdb file.
    ligands = vina_outputs.load_docked_ligands(out_pdbqt)
    complexes = [(protein_mol, ligand_mol) for ligand_mol in ligands]

    # Return docked files
    return docked_complex
