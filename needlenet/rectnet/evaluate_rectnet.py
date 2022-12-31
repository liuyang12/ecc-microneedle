import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_rectnet(rectnet, dataloader, device):
    rectnet.eval()
    num_val_batches = len(dataloader)

    criterion = torch.nn.MSELoss()  # MSE
    val_loss = 0

    with tqdm(total=num_val_batches, desc='Evaluation round', unit='batch', leave=False) as pbar:
        for idx, (image, corner) in enumerate(dataloader):
            image  =  image.to(device=device, dtype=torch.float32) # [B,1,H,W]
            corner = corner.to(device=device, dtype=torch.float32) # [B,4*2]

            with torch.no_grad():
                corner_pred = rectnet(image)

                val_loss += criterion(corner_pred, corner)
    
    rectnet.train()

    return val_loss / num_val_batches