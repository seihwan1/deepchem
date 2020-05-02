import unittest

class TestFragmentUtil(unittest.TestCase):

  def setUp(self):
    # TODO test more formats for ligand
    current_dir = os.path.dirname(os.path.realpath(__file__))
    self.protein_file = os.path.join(current_dir,
                                     '../../feat/tests/3ws9_protein_fixer_rdkit.pdb')
    self.ligand_file = os.path.join(current_dir, '../../feat/tests/3ws9_ligand.sdf')

