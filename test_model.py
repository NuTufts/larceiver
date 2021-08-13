import torch
from larceiver import LArceiverDetection
from uboonedataset import ubooneDetection

niter = 10
num_workers = 0
batch_size = 4
num_predictions = 16

# model
model = LArceiverDetection( depth=4,
                            dim=1,
                            num_detection_queries=num_predictions,
                            queries_dim=4,
                            num_latents = 128,
                            latent_dim = 32,
                            cross_heads = 1,
                            latent_heads = 8,
                            cross_dim_head = 16,
                            latent_dim_head = 16,
                            weight_tie_layers = False,
                            self_per_cross_attn = 1 )

print(model)
print(model.query_embed.weight.shape)



test = ubooneDetection( "test_detr2d.root", random_access=True, num_workers=num_workers, num_predictions=num_predictions )
loader = torch.utils.data.DataLoader(test,batch_size=batch_size,
                                     num_workers=num_workers,
                                     persistent_workers=False)

data = next(iter(loader))

#turn image into sequence
seq = data[0].reshape( (batch_size, 512*512, 1) )
print("seq.shape=",seq.shape)
print("forward pass")
out = model(seq)
print("out shape: ",out.shape)
