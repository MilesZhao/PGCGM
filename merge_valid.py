import os
import tarfile
import argparse
import subprocess
import numpy as np
import pandas as pd
import multiprocessing as mp
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.core.sites import PeriodicSite
from collections import defaultdict,Counter

class StrucPost(Structure):
    def merge_same_element(self, prop = 0.6):
        mapping = defaultdict(list)
        for site in self.sites:
            mapping[site.species.reduced_formula].append(site)

        new_sites = []
        elements = list(mapping.keys())
        for symbol in elements:
            holes = mapping[symbol]
            if len(holes) == 1:
                new_sites.append(holes[0])
                continue
            frac_coords = np.array([site.frac_coords for site in holes])
            d = self.lattice.get_all_distances(frac_coords, frac_coords)
            np.fill_diagonal(d, 0)
            clusters = fcluster(linkage(squareform((d + d.T) / 2)), float(holes[0].specie.atomic_radius)*2*prop, "distance")
            
            for c in np.unique(clusters):
                inds = np.where(clusters == c)[0]
                species = holes[inds[0]].species
                coords = holes[inds[0]].frac_coords
                for n, i in enumerate(inds[1:]):
                    offset = holes[i].frac_coords - coords
                    coords = coords + ((offset - np.round(offset)) / (n + 2)).astype(coords.dtype)
                new_sites.append(PeriodicSite(species, coords, self.lattice))
        self._sites = new_sites

    def check_dist(self, prop=0.75):
        d = self.distance_matrix
        iu = np.triu_indices(len(d),1)
        d = d[iu]

        atom_radius = []
        for i in range(len(self)):
            for j in range(i+1, len(self)):
                atom_radius.append(self[i].specie.atomic_radius+self[j].specie.atomic_radius)
        atom_radius = np.array(atom_radius)*prop
        
        return np.all(d > atom_radius)

def fun(file):
    with tarfile.open('ternary_symm_cifs/'+file, "r:gz") as tar:
        files = tar.getmembers()
        for path in files:
            f=tar.extractfile(path)
            content=f.read().decode("utf-8")
            try:
                crystal = StrucPost.from_str(content,fmt="cif")
                crystal.merge_same_element(args.merge_ratio)
                n = len(crystal)
                if n > 100:
                    crystal.merge_same_element(args.further_merge_ratio)
                    n = len(crystal)

                formula = crystal.composition.reduced_formula
                sg_info = crystal.get_space_group_info(symprec=0.1)
                cif = path.name.split('/')[-1]
                cif = cif.split('-')
                cif[0] = str(n)
                cif[1] = formula
                sgid = cif[2]
                if cif[2] == str(sg_info[1]) and crystal.check_dist(args.dist_ratio):
                    cif = '-'.join(cif)
                    crystal.to(fmt='cif', filename='ternary_final_cifs/merged_cifs_%.2f_%.2f/'%(args.merge_ratio, args.dist_ratio)+cif, symprec=0.1)

                    pro = Composition(formula).anonymized_formula + '-' + str(sgid) + '-' + str(len(crystal))

                    f = open('ternary_final_cifs/mat_merged_%.2f_%.2f.csv'%(args.merge_ratio, args.dist_ratio), 'a')
                    f.write('%s\n'%cif.replace('.cif',''))
                    f.close()

            except Exception as e:
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crystal Structure Generative Adversial Network')
    parser.add_argument("--merge_ratio", type=float, default=0.5, help="lambda for merging atoms")
    parser.add_argument("--further_merge_ratio", type=float, default=1.5, help="lambda for further merging atoms")
    parser.add_argument("--dist_ratio", type=float, default=0.75, help="lambda for dist check")
    args = parser.parse_args()
    
    
    if os.path.exists('ternary_final_cifs/'):
        os.system('rm -rf ternary_final_cifs/')
    os.system('mkdir ternary_final_cifs/')

    if os.path.exists('ternary_final_cifs/merged_cifs_%.2f_%.2f'%(args.merge_ratio, args.dist_ratio)):
        os.system('rm -rf ternary_final_cifs/merged_cifs_%.2f_%.2f'%(args.merge_ratio, args.dist_ratio))
    os.system('mkdir ternary_final_cifs/merged_cifs_%.2f_%.2f'%(args.merge_ratio, args.dist_ratio))

    if os.path.isfile('ternary_final_cifs/mat_merged_%.2f_%.2f.csv'%(args.merge_ratio, args.dist_ratio)):
        os.system('rm ternary_final_cifs/mat_merged_%.2f_%.2f.csv'%(args.merge_ratio, args.dist_ratio))

    symm_cifs = os.listdir("ternary_symm_cifs/")
    

    pool = mp.Pool(processes=40)
    pool.map(fun, symm_cifs)
    pool.close()

