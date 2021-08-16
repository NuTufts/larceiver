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

# device
print("device: ",args.device)
device = torch.device(args.device)

model,criterion,post = build(args)
print(model)
model.to(device)
criterion.to(device)

niter = 1
num_workers = 0
batch_size = 1
num_predictions = 16

# model
test = ubooneDetection( "test_detr2d.root", random_access=True,
                        num_workers=num_workers,
                        num_predictions=None,
                        num_channels=3 )
loader = torch.utils.data.DataLoader(test,batch_size=batch_size,
                                     num_workers=num_workers,
                                     collate_fn=collate_fn,
                                     persistent_workers=False)

# TEST FORWARD PASS
samples, targets = next(iter(loader))
samples.tensors = samples.tensors.to(device)
samples.mask    = samples.mask.to(device)

print("samples: ",type(samples))
print("samples.tensors: ",type(samples.tensors)," shape=",samples.tensors.shape,samples.tensors.device)
print("samples.mask: ",type(samples.mask)," shape=",samples.mask.shape,samples.mask.device)
print("forward pass")
outputs = model( samples )
print("out shape: ",outputs)

## TEST LOSS
print("boxes: ",targets[0]['boxes'].dtype)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
loss_dict = criterion(outputs, targets)
print("loss dict: ",loss_dict)
sys.exit(0)

weight_dict = criterion.weight_dict
losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

# reduce losses over all GPUs for logging purposes
loss_dict_reduced = utils.reduce_dict(loss_dict)
loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                              for k, v in loss_dict_reduced.items()}
loss_dict_reduced_scaled = {k: v * weight_dict[k]
                        for k, v in loss_dict_reduced.items() if k in weight_dict}
losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        
loss_value = losses_reduced_scaled.item()
        
        
