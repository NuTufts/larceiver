import os
import torch
import ROOT as rt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def uboonedetection_collate_fn(batch):
    batch = list(zip(*batch))
    print(batch)    
    return tuple(batch)


class ubooneDetection(torch.utils.data.Dataset):
    def __init__(self, root_file_path,
                 num_predictions=None,                 
                 planes=[2],
                 num_workers=1,
                 random_access=False,
                 return_masks=False):
        """
        Parameters:
          root_file_path: the location of a ROOT file to load data from
          num_predictions: if not None, then number of targets is padded or truncated to be a fixed size. Needed for DETR.
        """
        # the ROOT tree expected in the file
        #print("ubooneDetection")
        self._num_workers = num_workers
        if num_workers<=0:
            self._num_workers = 1
        self._num_predictions = num_predictions
        
        self.chains = {}
        for i in range(self._num_workers):
            self.chains[i] = rt.TChain("detr")        
            if type(root_file_path) is str:
                if not os.path.exists(root_file_path):
                    raise RuntimeError("Root file path does not exist: ",root_file_path)
                self.chains[i].Add( root_file_path )

        self.nentries = self.chains[0].GetEntries()
        self.random_access = random_access
        self.return_masks = return_masks

        self._current_entry  = [0 for i in range(self._num_workers)]
        self._current_nbytes = [0 for i in range(self._num_workers)]
        self._nloaded        = [0 for i in range(self._num_workers)]
        self.planes = planes
        

    def __getitem__(self, idx):
        #img, target = super(CocoDetection, self).__getitem__(idx)

        workerinfo = torch.utils.data.get_worker_info()
        workerid = 0
        if workerinfo is not None:
            workerid = workerinfo.id
            #print(workerinfo)
            #print("workerid: ",workerid," chain=",self.chains[workerid])

        chain = self.chains[workerid]
        current = self._current_entry[workerid]
        current_nbytes = 0
        nloaded = 0
        
        ok = False
        entry = 0
        while not ok:
            if not self.random_access:
                entry = current
            else:
                entry = np.random.randint(0,self.nentries)
        
            current_nbytes = chain.GetEntry(entry)
            nloaded += 1
            if current_nbytes==0:
                raise RuntimeError("Error reading entry %d"%(entry))

            img_v    = [ chain.image_v.at(p).tonumpy() for p in self.planes ]
            annote_v = [ chain.bbox_v.at(p).tonumpy()[:,:5] for p in self.planes ]

            # check the image is OK
            ok = True
            for p in range(len(img_v)):
                img = img_v[p]
                npix = (img>10.0).sum()
                if npix<20:
                    ok = False
                bbox = annote_v[p]
                nboxes = bbox.shape[0]
                if nboxes>5:
                    ok = False
            
            if not self.random_access:
                entry += 1
            
        if len(self.planes)>1:
            target = {'image_id':entry, 'annotations':annote_v }
            imgout = img_v
        else:
            target = {'image_id':entry, 'annotations':annote_v[0]}
            if self._num_predictions is not None:
                fixed_pred = np.zeros( (self._num_predictions,5) )
                nbbox = target['annotations'].shape[0]
                if nbbox<self._num_predictions:
                    fixed_pred[:nbbox,:] = target['annotations'][:,:]
                else:
                    fixed_pred[:,:] = target['annotations'][:self._num_predictions,:]
                target['annotations'] = fixed_pred
            imgout = img_v[0]
            
        self._nloaded[workerid] += nloaded
        self._current_entry[workerid] = entry
        return imgout, target

    def __len__(self):
        return self.nentries

    def print_status(self):
        for i in range(self._num_workers):
            print("worker: entry=%d nloaded=%d"%(self._current_entry[i],self._nloaded[i]))


if __name__ == "__main__":

    import time

    niter = 10
    num_workers = 0
    
    test = ubooneDetection( "test_detr2d.root", random_access=True, num_workers=num_workers, num_predictions=10 )
    loader = torch.utils.data.DataLoader(test,batch_size=64,
                                         num_workers=num_workers,
                                         persistent_workers=False)

    start = time.time()
    for iiter in range(niter):
        data = next(iter(loader))
        print(data[0].shape,data[1]['annotations'].shape)
    end = time.time()
    elapsed = end-start
    sec_per_iter = elapsed/float(niter)
    print("sec per iter: ",sec_per_iter)
    
    loader.dataset.print_status()
