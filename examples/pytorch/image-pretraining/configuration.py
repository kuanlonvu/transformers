import torch
from transformers import SwinConfig

if torch.cuda.is_available():
  device = 0
  torch_device = torch.device('cuda')
else:
  device = -1
  torch_device = torch.device('cpu')

print(f'Using device: {device}')


IMAGE_SIZE = 192
PATCH_SIZE = 4
EMBED_DIM = 128
DEPTHS = [2, 2, 18, 2]
NUM_HEADS = [4, 8, 16, 32]
WINDOW_SIZE = 6

config = SwinConfig(
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    depths=DEPTHS,
    num_heads=NUM_HEADS,
    window_size=WINDOW_SIZE,
)
config.save_pretrained("./")