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


model = build(args)
print(model)

sys.exit(0)

niter = 10
num_workers = 0
batch_size = 4
num_predictions = 16

# model
test = ubooneDetection( "test_detr2d.root", random_access=True,
                        num_workers=num_workers,
                        num_predictions=num_predictions,
                        num_channels=3 )
loader = torch.utils.data.DataLoader(test,batch_size=batch_size,
                                     num_workers=num_workers,
                                     persistent_workers=False)

data = next(iter(loader))

#turn image into sequence
seq = data[0].reshape( (batch_size, 512*512, 1) )
print("seq.shape=",seq.shape)
print("forward pass")
#out = model(seq)
#print("out shape: ",out.shape)
