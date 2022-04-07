import os
import json
import glob
import pickle
import shutil
import tarfile
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from collections import Counter
from model import Generator
from simple_dist import sp_lookup
np.set_printoptions(precision=4,suppress=True)

short_LaAc = ['Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']
LaAc = ['Xe','Ac', 'Am', 'Bk', 'Ce', 'Cf', 'Cm', 'Dy', 'Er', 'Es', 'Eu', 'Fm', 'Gd', 'Ho', 'La', 'Lr', 'Lu', 'Md', 'Nd', 'No', 'Np', 'Pa', 'Pm', 'Pr', 'Pu', 'Sm', 'Tb', 'Th', 'Tm', 'U', 'Yb']

def fake_generator(n_samples, n_spacegroup, sp_prob, ele_num):
    label_sp = np.random.choice(n_spacegroup,n_samples,p=sp_prob)

    with open('data/elements_id_NoPo.json', 'r') as f:
        e_d = json.load(f)
        element_ids = list(e_d.values())

    label_elements = []
    for i in range(n_samples):
        fff = np.random.choice(element_ids,ele_num,replace=False)
        label_elements.append(fff)
    label_elements = np.array(label_elements)
    return label_sp,label_elements

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crystal Structure Generative Adversial Network')
    parser.add_argument("--element_crystal", type=int, default=3, help="number of elements in the crystal")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--n_samples", type=int, default=10000, help="number of crystal to generate")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--savedir', type=str, default=None)
    parser.add_argument('--matdir', type=str, default='virtual_mat')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    argdict = vars(args)
    print(argdict)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    #loading data
    # AUX_DATA, DATA = load_mat()
    with open('data/paras.pickle', 'rb') as handle:
        AUX_DATA = pickle.load(handle)
    n_spacegroup = len(AUX_DATA[-2])
    atom_embedding = np.load('data/elements_features.npy')
    atom_embedding = torch.Tensor(atom_embedding).to(device)
    box_abc_pat = torch.Tensor(AUX_DATA[0]).to(device)
    box_angle_pat = torch.Tensor(AUX_DATA[1]).to(device)
    cr = torch.Tensor(AUX_DATA[3]).to(device)
    lr = torch.Tensor([AUX_DATA[2]]).to(device)

    netG = Generator(atom_embedding.shape[1],args.latent_dim).to(device)
    checkpoint = torch.load('models/%s/%s'%(args.savedir, args.model))
    netG.load_state_dict(checkpoint['state_dict'])
    netG.eval()
    spinfo =  sp_lookup(device, AUX_DATA[-2])

    random_sp,random_ele = fake_generator(args.n_samples, n_spacegroup, AUX_DATA[-1], args.element_crystal)
    #prepare spacegroup label
    sp2id = AUX_DATA[-2]
    id2sp = {sp2id[k]:k for k in sp2id}
    print(id2sp)

    #prepare elemente labels
    with open('data/elements_id_NoPo.json', 'r') as f:
       e_d = json.load(f)
       re = {e_d[k]:k for k in e_d}
    

    if os.path.exists('%s/'%(args.matdir)):
        os.system('rm -rf %s/'%(args.matdir))
    os.system('mkdir %s/'%(args.matdir))

    count_sample = 0
    with torch.no_grad():
        for i in range(args.n_samples//args.batch_size+1):
            start_idx = i*args.batch_size
            end_idx = min((i+1)*args.batch_size, args.n_samples)

            sp_id = random_sp[start_idx:end_idx]
            sp_id = torch.Tensor(sp_id).type(torch.int64)

            ele = random_ele[start_idx:end_idx]
            ele = torch.Tensor(ele).to(device)
            ele_ids = ele.type(torch.int64)
            e = atom_embedding[ele_ids]

            z = torch.normal(0.0, 1.0, size=(end_idx-start_idx, args.latent_dim)).to(device)
            coords,box_abc = netG(spinfo.symm_op_collection[sp_id], torch.transpose(e,1,2), z)

            arr_coords0 = coords[:,0,:,:]*cr + cr
            arr_coords0 = arr_coords0.cpu().detach().numpy()

            arr_coords1 = coords[:,1,:,:]*cr + cr
            arr_coords1 = arr_coords1.cpu().detach().numpy()

            arr_coords2 = coords[:,2,:,:]*cr + cr
            arr_coords2 = arr_coords2.cpu().detach().numpy()

            arr_coords0 = np.round(arr_coords0, 2)
            arr_coords1 = np.round(arr_coords1, 2)
            arr_coords2 = np.round(arr_coords2, 2)
            
            box_abc = torch.exp(box_abc*lr + lr)
            box_abc = torch.einsum('bl,bls->bs', box_abc, box_abc_pat[sp_id])
            box_angles = box_angle_pat[sp_id]

            arr_lengths = box_abc.cpu().detach().numpy()
            arr_angles = box_angles.cpu().detach().numpy()
            arr_ele = ele.cpu().detach().numpy()
            arr_spid = sp_id.cpu().detach().numpy()
            
            for j in range(arr_coords0.shape[0]):
                for rot in range(3):
                    f = open('data/cif-template.txt', 'r')
                    template = f.read()
                    f.close()
                    if rot == 0:
                        coords = arr_coords0[j]
                    elif rot == 1:
                        coords = arr_coords1[j]
                    else:
                        coords = arr_coords2[j]

                    lengths = arr_lengths[j]
                    angles  = arr_angles[j]
                    elements = arr_ele[j]

                    template = template.replace('SYMMETRY-SG', id2sp[arr_spid[j]])
                    template = template.replace('LAL', str(lengths[0]))
                    template = template.replace('LBL', str(lengths[1]))
                    template = template.replace('LCL', str(lengths[2]))
                    template = template.replace('DEGREE1', str(angles[0]))
                    template = template.replace('DEGREE2', str(angles[1]))
                    template = template.replace('DEGREE3', str(angles[2]))
                    f = open('data/symmetry-equiv/%s.txt'%id2sp[arr_spid[j]].replace('/','#'), 'r')
                    sym_ops = f.read()
                    f.close()

                    template = template.replace('TRANSFORMATION\n', sym_ops)

                    for m in range(args.element_crystal):
                        row = ['',re[elements[m]],re[elements[m]]+str(m),\
                            str(coords[m][0]),str(coords[m][1]),str(coords[m][2]),'1']
                        row = '  '.join(row)+'\n'
                        template+=row

                    template += '\n'
                    f = open('%s/%s---%d_%d.cif'%\
                        (args.matdir,id2sp[arr_spid[j]].replace('/','#'), count_sample, rot),'w')
                    f.write(template)
                    f.close()
                if args.verbose and count_sample%1000==0:
                    print('space group: ', id2sp[arr_spid[j]])
                    print('lengths:',lengths)
                    print('angles:',angles)
                    print('coord:\n', arr_coords0[j], '\n---\n', arr_coords1[j], '\n---\n', arr_coords2[j])
                    print(' '*25)
                count_sample += 1

                if count_sample%1000==0:
                    cifs = glob.glob(args.matdir + '/*.cif')
                    archive = args.matdir + "/%d.tar.gz"%(count_sample//1000)
                    with tarfile.open(archive, "w:gz") as tar:
                        for file in cifs:
                            tar.add(file)

                    for f in cifs:
                        os.remove(f)






