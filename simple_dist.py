import torch
import numpy as np
from pymatgen.symmetry.groups import SpaceGroup

class sp_lookup:
    def __init__(self, device, sp_dict):
        self._affine_matrix_list = []
        self._affine_matrix_range = []
        self.device = device

        #mapping the heck
        d = {}
        for i in range(1,231):
            if i == 222:
                d['Pn-3n'] = i
                continue
            if i == 224:
                d['Pn-3m'] = i
                continue
            if i == 227:
                d['Fd-3m'] = i
                continue
            if i == 228:
                d['Fd-3c'] = i
                continue
            if i == 129:
                d['P4/nmm'] = i
                continue
            symbol = SpaceGroup.from_int_number(i).symbol
            if symbol.endswith("H"):
                symbol = symbol.replace("H", "")
            d[symbol] = i
        
        sp_list = []
        for symbol,i in sp_dict.items():
            sp_list.append((i, d[symbol], symbol))
        # exit(sp_list)
        for i,spid,_ in sp_list:
            symops = SpaceGroup.from_int_number(spid).symmetry_ops

            for op in symops:
                tmp = op.affine_matrix.astype(np.float32)
                
                if np.all(tmp == -1.0):
                    print(tmp)
                self._affine_matrix_list.append(tmp)
            
            self._affine_matrix_range.append((len(self._affine_matrix_list) - len(symops),\
             len(self._affine_matrix_list)))

    @property
    def affine_matrix_list(self):
        return torch.Tensor(self._affine_matrix_list).to(self.device)
    
    @property
    def affine_matrix_range(self):
        return torch.Tensor(self._affine_matrix_range).type(torch.int64).to(self.device)

    @property
    def symm_op_collection(self):
        arr = []
        for r0,r1 in self._affine_matrix_range:
            ops = np.array(self._affine_matrix_list)[r0:r1]
            zeros = np.zeros((192-len(ops), 4, 4))
            ops = np.concatenate((ops, zeros),0)
            arr.append(ops)
        arr = np.stack(arr, 0)

        return torch.Tensor(arr).type(torch.float32).to(self.device)

class DiffLattice(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.pi = torch.acos(torch.zeros(1)).type(torch.float64).to(self.device)[0]/90.0

    def forward(self, a,b,c,alpha,beta,gamma):
        a = a.type(torch.float64)
        b = b.type(torch.float64)
        c = c.type(torch.float64)
        alpha = alpha.type(torch.float64)
        beta = beta.type(torch.float64)
        gamma = gamma.type(torch.float64)

        alpha, beta, gamma = alpha*self.pi, beta*self.pi, gamma*self.pi
        cos_alpha, cos_beta, cos_gamma = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
        sin_alpha, sin_beta, sin_gamma = torch.sin(alpha), torch.sin(beta), torch.sin(gamma)

        val = (cos_alpha * cos_beta - cos_gamma).to(self.device) / (sin_alpha * sin_beta).to(self.device)
        one = torch.ones(val.shape[0]).type(torch.float64).to(self.device)
        mone = -one
        val = torch.max(torch.min(val, one), mone)
        gamma_star = torch.acos(val)
        
        zero = torch.zeros(val.shape[0]).type(torch.float64).to(self.device) 
        vector_a = torch.stack((a*sin_beta, zero, a*cos_beta), -1)
        vector_b = torch.stack(
            [
                -b*sin_alpha * torch.cos(gamma_star),
                b * sin_alpha * torch.sin(gamma_star),
                b * cos_alpha
            ], -1)
        vector_c = torch.stack([zero, zero, c], -1)
        matrix = torch.stack((vector_a, vector_b, vector_c), 1).type(torch.float32)

        return matrix

class DiffDistMatrix(torch.nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.epsilon = torch.tensor(1e-10).to(self.device)

    def forward(self, fcoords1, fcoords2, matrix):
        cart_coords = torch.einsum('sij,sjk->sik', fcoords1%1, matrix)
        cart_f1 = torch.einsum('sij,sjk->sik', fcoords1%1, matrix)
        cart_f2 = torch.einsum('sij,sjk->sik', fcoords2%1, matrix)
        cart_im = torch.einsum('ij,sjk->sik', self.images_view, matrix)

        I = fcoords1.shape[1]
        cart_f2 = cart_f2.unsqueeze_(2)
        cart_f2 = torch.repeat_interleave(cart_f2, repeats=I, dim=2)
        
        cart_f1 = cart_f1.unsqueeze_(1)
        cart_f1 = torch.repeat_interleave(cart_f1, repeats=I, dim=1)
        cart_f2 = cart_f2.sub_(cart_f1)
        cart_f2 = cart_f2.unsqueeze_(3) 
        cart_f2 = torch.repeat_interleave(cart_f2, repeats=27, dim=3)

        cart_im = cart_im.unsqueeze_(1)
        cart_im = torch.repeat_interleave(cart_im, repeats=I, dim=1)
        cart_im = cart_im.unsqueeze_(1)
        cart_im = torch.repeat_interleave(cart_im, repeats=I, dim=1)
        cart_f2 = cart_f2.add_(cart_im).pow_(2)
        cart_f2 = torch.sum(cart_f2, -1)
        dm,_ = torch.min(cart_f2, -1) 
        dm = dm + self.epsilon #numerical stability

        return torch.sqrt(dm),cart_coords

    @property
    def images_view(self):
        img = torch.tensor([
                       [-1., -1., -1.],
                       [-1., -1.,  0.],
                       [-1., -1.,  1.],
                       [-1.,  0., -1.],
                       [-1.,  0.,  0.],
                       [-1.,  0.,  1.],
                       [-1.,  1., -1.],
                       [-1.,  1.,  0.],
                       [-1.,  1.,  1.],
                       [ 0., -1., -1.],
                       [ 0., -1.,  0.],
                       [ 0., -1.,  1.],
                       [ 0.,  0., -1.],
                       [ 0.,  0.,  0.],
                       [ 0.,  0.,  1.],
                       [ 0.,  1., -1.],
                       [ 0.,  1.,  0.],
                       [ 0.,  1.,  1.],
                       [ 1., -1., -1.],
                       [ 1., -1.,  0.],
                       [ 1., -1.,  1.],
                       [ 1.,  0., -1.],
                       [ 1.,  0.,  0.],
                       [ 1.,  0.,  1.],
                       [ 1.,  1., -1.],
                       [ 1.,  1.,  0.],
                       [ 1.,  1.,  1.]
                       ]
                    )
        return img.to(self.device)

def calc_volume(matrix):
    cross = torch.cross(matrix[:,0,:], matrix[:,1,:])
    vol = torch.einsum('ij,ij->i',cross, matrix[:,2,:])
    return torch.abs(vol)

if __name__ == '__main__':
    import os,json
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    from monty.io import zopen
    from pymatgen.io.cif import CifParser,CifFile
    from pymatgen.util.coord import pbc_shortest_vectors
    import time
    import numpy as np

    device =  "cuda" if torch.cuda.is_available() else "cpu"

    with open('data/sg_freq.json', 'r') as fp:
        sg_freq = json.load(fp)
    idx = 0
    sg_ratio = {}
    sp2id = {}
    for k,v in sorted(sg_freq.items(), key=lambda x:x[1], reverse=True):
        if v >= 400:
            sg_ratio[k] = v
            sp2id[k] = idx
            idx += 1
    print(sp2id)
    spinfo = sp_lookup(device, sp2id)
    print(spinfo.symm_op_collection.shape)
    label_sp = np.random.choice(20,32)
    label_sp = torch.Tensor(label_sp).type(torch.int64).to(device)
    print(label_sp)
    print(spinfo.symm_op_collection[label_sp].shape)
    coords = torch.Tensor([[[0.25,0.25,0.75],[0.0,0.0,0.0],[0.12541,0.37459,0.12541]],[[0.0,0.0,0.0],[0.125,0.125,0.625],[0.114499,0.114499,0.385501]]]).to(device)
    ones = torch.ones((2,3,1)).to(device)
    print(coords.shape, ones.shape)
    coords = torch.cat((coords, ones), 2)
    print(coords)
    print(coords.shape)
    ops = spinfo.symm_op_collection[[15,4]]
    print(ops.shape)
    r = torch.einsum('bmij,bkj->bmki', ops, coords)[:,:,:,:3]
    r = r - torch.floor(r)
    print(r)
    print(r.shape)
    exit()
    cifs = os.listdir('/data/yong/mp_db/cifs/')
    cifs.sort()
    a,b,c,alpha,beta,gamma = [],[],[],[],[],[]
    coord0 = []
    for cif in cifs[2582:2583]:
        print(cif)
        with zopen('/data/yong/mp_db/cifs/%s'%cif, "rt") as f:
            contents = f.read()
            parser = CifParser.from_string(contents)
            struc = parser.get_structures(False)[0]
            lmatrix = list(struc.lattice.parameters)
            a.append(lmatrix[0])
            b.append(lmatrix[1])
            c.append(lmatrix[2])
            alpha.append(lmatrix[3])
            beta.append(lmatrix[4])
            gamma.append(lmatrix[5])
            print(struc)
            print(struc.cart_coords)
            coord0.append(struc.frac_coords)

    a = torch.tensor(a).type(torch.float64).to(device)
    b = torch.tensor(b).type(torch.float64).to(device)
    c = torch.tensor(c).type(torch.float64).to(device)
    alpha = torch.tensor(alpha).type(torch.float64).to(device)
    beta = torch.tensor(beta).type(torch.float64).to(device)
    gamma = torch.tensor(gamma).type(torch.float64).to(device)

    coord0 = torch.tensor(coord0).type(torch.float32).to(device)

    lattice = DiffLattice(device)
    dist_calc = DiffDistMatrix(device)
    matrix = lattice(a,b,c,alpha,beta,gamma)
    d,cart = dist_calc(coord0, coord0, matrix)
    # print(d)
    print(cart)
    exit()
    a = torch.tensor(a).type(torch.float64).to(device)
    b = torch.tensor(b).type(torch.float64).to(device)
    c = torch.tensor(c).type(torch.float64).to(device)
    alpha = torch.tensor(alpha).type(torch.float64).to(device)
    beta = torch.tensor(beta).type(torch.float64).to(device)
    gamma = torch.tensor(gamma).type(torch.float64).to(device)

    lattice = DiffLattice(device)
    matrix = lattice(a,b,c,alpha,beta,gamma)

    # print(matrix)
    print(matrix.shape)
    # volume = torch.abs(torch.dot(torch.cross(matrix[0][0], matrix[0][1]), matrix[0][2]))
    volume = calc_volume(matrix)
    print(volume)
