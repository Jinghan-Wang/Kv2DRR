import torch
from torch.utils.data import DataLoader
from datasets.paired_image_dataset import PairedNiiDataset
from models.residual_mapper import ResidualToneMapper
from losses.tone_losses2 import ToneMappingLoss
from utils.misc import load_yaml
from utils.metrics import calc_psnr, calc_ssim
from utils.io import tensor_to_numpy01


@torch.no_grad()
def main():
    cfg = load_yaml("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PairedNiiDataset(
        cfg["dataset"]["val_input_dir"],
        cfg["dataset"]["val_target_dir"],
        cfg["dataset"]["fixed_h"],
        cfg["dataset"]["fixed_w"],
        cfg["dataset"]["normalize_mode"],
        tuple(cfg["dataset"]["fixed_range"]),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = ResidualToneMapper(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
        delta_scale=cfg["model"]["delta_scale"],
    ).to(device)

    ckpt = torch.load(cfg["infer"]["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = ToneMappingLoss(**cfg["loss"]).to(device)

    losses, psnrs, ssims = [], [], []
    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        pred, delta = model(inp)
        loss, _ = criterion(pred, tgt, delta)
        losses.append(loss.item())
        psnrs.append(calc_psnr(tensor_to_numpy01(pred[0]), tensor_to_numpy01(tgt[0])))
        ssims.append(calc_ssim(tensor_to_numpy01(pred[0]), tensor_to_numpy01(tgt[0])))

    print(f"Test loss: {sum(losses)/len(losses):.4f}")
    print(f"Test PSNR: {sum(psnrs)/len(psnrs):.4f}")
    print(f"Test SSIM: {sum(ssims)/len(ssims):.4f}")


if __name__ == "__main__":
    main()