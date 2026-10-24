import os

# Precisa valer antes de o PyTorch inicializar a GPU. Sem isso, o cache do alocador fragmenta entre
# fotos (13 GB reservados para ~3 GB em uso) e, no WSL2, transborda para a RAM: o SAM passa de ~5 s
# para até ~2 min por foto. Com segmentos expansíveis a reserva fica estável em ~3,3 GB.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
