import torch, numpy as np
import torchvision.transforms as v2
from torchvision.models import alexnet, AlexNet_Weights



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alex = alexnet(weights=AlexNet_Weights.DEFAULT)
alex.classifier = torch.nn.Sequential(
    *list(alex.classifier.children())[:-1]
)
# AlexNet in evaluation mode and move it to the device (GPU or CPU)
backbone = alex.to(device).eval()


transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
@torch.no_grad()
def extract_torch(crops, bs=64):
    out = []
    for i in range(0, len(crops), bs):
        batch = torch.stack([transform(c) for c in crops[i:i+bs]]).to(device)
        out.append(backbone(batch).cpu().numpy())
    return np.vstack(out)