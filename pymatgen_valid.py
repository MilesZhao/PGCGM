import os
import glob
import shutil
import tarfile
import random
import warnings
import argparse
warnings.filterwarnings("ignore")
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
import multiprocessing as mp

gen_cifs = os.listdir('ternary_gen_cifs/')
print(len(gen_cifs))

if os.path.exists('ternary_symm_cifs/'):
    os.system('rm -rf ternary_symm_cifs/')
os.system('mkdir ternary_symm_cifs/')

def read_tar(file):
    whichtar = file.split('.')[0]
    os.mkdir('ternary_symm_cifs/'+whichtar)
    with tarfile.open('ternary_gen_cifs/'+file, "r:gz") as tar:
        files = tar.getmembers()
        for cif in files:
            try:
                name = cif.name.split('/')[-1]
                sp = name.split('---')[0].replace('#','/')
                i = name.split('---')[1].replace('.cif','')
                f=tar.extractfile(cif)
                content=f.read().decode("utf-8")
                crystal = Structure.from_str(content,fmt="cif")
                formula = crystal.composition.reduced_formula
                sg_info = crystal.get_space_group_info(symprec=0.1)
                if sp == sg_info[0]:
                    crystal.to(fmt='cif',\
                     filename='ternary_symm_cifs/%s/%d-%s-%d-%s.cif'%\
                     (whichtar, len(crystal),formula,sg_info[1],i),symprec=0.1)
            except Exception as e:
                # print(e)
                pass

    #write valid crystals
    cifs = glob.glob('ternary_symm_cifs/%s/*.cif'%whichtar)
    archive = 'ternary_symm_cifs/' + file
    with tarfile.open(archive, "w:gz") as tar:
        for cif in cifs:
            tar.add(cif)  
    shutil.rmtree('ternary_symm_cifs/' + whichtar)


pool = mp.Pool(processes=46)
pool.map(read_tar, gen_cifs)
pool.close()

