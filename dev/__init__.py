from .draft import RelEnc as RelationalEncoder, Encoder, RelationHead, RelationHead2, RelationHead3, RelationHeadLong, aggregate
from .from_paper import MultiCIFAR10, train_transform
from .datasets import MRCData,MRCSampler
from torch.utils.data import DataLoader
from torch import save as save_model
from torch.cuda import is_available as cuda_is_available
device = "cuda" if cuda_is_available() else "cpu"

def relation_head(i):
    return [RelationHead,RelationHead2,RelationHead3,RelationHeadLong,RelationHeadCone][i]
