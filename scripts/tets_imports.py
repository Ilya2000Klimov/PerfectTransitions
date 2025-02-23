import torch
import torchaudio
import fairseq
import yaml

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Fairseq version:", fairseq.__version__)
print("PyYAML version:", yaml.__version__)
