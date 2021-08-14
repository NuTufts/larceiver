import os,sys
import torch
from detr.detr_args import get_detr_args_parser
from pathlib import Path

parser = get_detr_args_parser()
args = parser.parse_args()
if args.output_dir:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

#from larceiver import LArceiverDetection
from uboonedataset import ubooneDetection

# build model
from detr.detr import build
from detr.util.misc import NestedTensor,collate_fn


model,criterion,post = build(args)
print(model)

niter = 1
num_workers = 0
batch_size = 1
num_predictions = 16

# model
test = ubooneDetection( "test_detr2d.root", random_access=True,
                        num_workers=num_workers,
                        num_predictions=num_predictions,
                        num_channels=3 )
loader = torch.utils.data.DataLoader(test,batch_size=batch_size,
                                     num_workers=num_workers,
                                     collate_fn=collate_fn,
                                     persistent_workers=False)

samples, targets = next(iter(loader))
print("samples: ",type(samples))
print("samples.tensors: ",type(samples.tensors)," shape=",samples.tensors.shape)
print("samples.mask: ",type(samples.mask)," shape=",samples.mask.shape)
print("forward pass")
#out = model( samples )
#print("out shape: ",out)
