import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from datasets.paired_image_dataset import PairedNiiDataset
from models.residual_mapper import ResidualToneMapper
from losses.tone_losses import ToneMappingLoss
from utils.io import ensure_dir, save_gray_image, save_numpy_as_nii_gz, tensor_to_numpy01
from utils.metrics import calc_psnr, calc_ssim
from utils.misc import load_yaml, set_seed
import setproctitle
setproctitle.setproctitle("WJH_Student")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def make_new_pred_from_pred_input(pred_np, input_np, sigma=5.0, alpha=1.0):
    """
    pred_np, input_np: 2D numpy, range [0,1], float32
    new_pred = lowfreq(pred) + alpha * highfreq(input)
    """
    pred_np = pred_np.astype(np.float32)
    input_np = input_np.astype(np.float32)

    pred_low = cv2.GaussianBlur(pred_np, (0, 0), sigmaX=sigma, sigmaY=sigma)
    input_low = cv2.GaussianBlur(input_np, (0, 0), sigmaX=sigma, sigmaY=sigma)
    input_high = input_np - input_low

    new_pred = pred_low + alpha * input_high
    #new_pred = np.clip(new_pred, 0.0, 1.0)
    return new_pred

@torch.no_grad()
def validate(model, loader, criterion, device, save_vis_dir=None, epoch=None, global_step=None):
    model.eval()
    val_loss = 0.0
    psnr_list = []
    ssim_list = []
    save_sample_idx = None
    if save_vis_dir is not None and len(loader.dataset) > 0:
        save_sample_idx = int(torch.randint(low=0, high=len(loader.dataset), size=(1,)).item())

    for i, batch in enumerate(loader):
        inp = batch["input"].to(device, non_blocking=True)
        inp_base = batch["input_base"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)

        pred, delta = model(inp)
        loss, _ = criterion(pred, tgt, delta, inp_base)
        val_loss += loss.item()

        for b in range(inp.shape[0]):
            input_np = tensor_to_numpy01(inp_base[b])
            pred_np = tensor_to_numpy01(pred[b])
            target_np = tensor_to_numpy01(tgt[b])

            delta_scale = getattr(model, "delta_scale", 0.15)
            delta_np = tensor_to_numpy01((delta[b] + delta_scale) / (2.0 * delta_scale + 1e-8))

            # -------- 新增：后处理生成 new_pred --------
            new_pred_np = make_new_pred_from_pred_input(
                pred_np=pred_np,
                input_np=input_np,
                sigma=5.0,  # 可调: 5/7/9
                alpha=1.0  # 可调: 0.8/1.0/1.2
            )

            psnr_list.append(calc_psnr(pred_np, target_np))
            ssim_list.append(calc_ssim(pred_np, target_np))
            sample_idx = i * inp.shape[0] + b
            if save_sample_idx is not None and sample_idx == save_sample_idx:
                tag = f"epoch{epoch:03d}_step{global_step:07d}_val_{b}"

                pred_to_save = pred_np.astype("float32")
                new_pred_to_save = new_pred_np.astype("float32")
                input_to_save = input_np.astype("float32")
                target_to_save = target_np.astype("float32")
                delta_to_save = delta_np.astype("float32")

                norm_meta = batch.get("norm_meta", None)

                if norm_meta is not None:
                    mode = None
                    try:
                        if isinstance(norm_meta["mode"], (list, tuple)):
                            mode = norm_meta["mode"][b]
                        else:
                            mode = norm_meta["mode"]
                    except Exception:
                        mode = None

                    try:
                        if mode == "pair_independent":
                            inp_min = float(norm_meta["inp_min"][b])
                            inp_max = float(norm_meta["inp_max"][b])
                            tgt_min = float(norm_meta["tgt_min"][b])
                            tgt_max = float(norm_meta["tgt_max"][b])

                            pred_to_save = pred_np * (inp_max - inp_min) + inp_min
                            new_pred_to_save = new_pred_np * (4095 - 0) + 0
                            input_to_save = input_np * (inp_max - inp_min) + inp_min
                            target_to_save = target_np * (tgt_max - tgt_min) + tgt_min

                        elif mode == "input_reference":
                            ref_min = float(norm_meta["ref_min"][b])
                            ref_max = float(norm_meta["ref_max"][b])

                            pred_to_save = pred_np * (ref_max - ref_min) + ref_min
                            input_to_save = input_np * (ref_max - ref_min) + ref_min
                            target_to_save = target_np * (ref_max - ref_min) + ref_min

                        elif mode == "fixed_range":
                            ref_min = float(norm_meta["ref_min"][b])
                            ref_max = float(norm_meta["ref_max"][b])

                            pred_to_save = pred_np * (ref_max - ref_min) + ref_min
                            input_to_save = input_np * (ref_max - ref_min) + ref_min
                            target_to_save = target_np * (ref_max - ref_min) + ref_min

                    except Exception as e:
                        print(f"[validate] warning: failed to denormalize sample {b}: {e}")

                input_meta = batch.get("input_meta", None)
                if input_meta is not None:
                    try:
                        spacing = tuple(float(x[b]) for x in input_meta["spacing"])
                        origin = tuple(float(x[b]) for x in input_meta["origin"])
                        direction = tuple(float(x[b]) for x in input_meta["direction"])

                        save_numpy_as_nii_gz(
                            input_to_save,
                            os.path.join(save_vis_dir, f"{tag}_input.nii.gz"),
                            spacing=spacing,
                            origin=origin,
                            direction=direction,
                        )
                        save_numpy_as_nii_gz(
                            pred_to_save,
                            os.path.join(save_vis_dir, f"{tag}_pred.nii.gz"),
                            spacing=spacing,
                            origin=origin,
                            direction=direction,
                        )
                        save_numpy_as_nii_gz(
                            new_pred_to_save,
                            os.path.join(save_vis_dir, f"{tag}_new_pred.nii.gz"),
                            spacing=spacing,
                            origin=origin,
                            direction=direction,
                        )
                        save_numpy_as_nii_gz(
                            target_to_save,
                            os.path.join(save_vis_dir, f"{tag}_target.nii.gz"),
                            spacing=spacing,
                            origin=origin,
                            direction=direction,
                        )
                        save_numpy_as_nii_gz(
                            delta_to_save,
                            os.path.join(save_vis_dir, f"{tag}_delta_vis.nii.gz"),
                            spacing=spacing,
                            origin=origin,
                            direction=direction,
                        )

                        # 如果你也想保存 png，把下面四行取消注释
                        # save_gray_image(os.path.join(save_vis_dir, f"{tag}_input.png"), input_np)
                        # save_gray_image(os.path.join(save_vis_dir, f"{tag}_new_pred.png"), new_pred_np)
                        # save_gray_image(os.path.join(save_vis_dir, f"{tag}_pred.png"), pred_np)
                        # save_gray_image(os.path.join(save_vis_dir, f"{tag}_target.png"), target_np)
                        # save_gray_image(os.path.join(save_vis_dir, f"{tag}_delta_vis.png"), delta_np)

                    except Exception as e:
                        print(f"[validate] warning: failed to save nii with meta for sample {b}: {e}")
                        save_numpy_as_nii_gz(input_to_save, os.path.join(save_vis_dir, f"{tag}_input.nii.gz"))
                        save_numpy_as_nii_gz(pred_to_save, os.path.join(save_vis_dir, f"{tag}_pred.nii.gz"))
                        save_numpy_as_nii_gz(new_pred_to_save, os.path.join(save_vis_dir, f"{tag}_new_pred.nii.gz"))
                        save_numpy_as_nii_gz(target_to_save, os.path.join(save_vis_dir, f"{tag}_target.nii.gz"))
                        save_numpy_as_nii_gz(delta_to_save, os.path.join(save_vis_dir, f"{tag}_delta_vis.nii.gz"))
                else:
                    save_numpy_as_nii_gz(input_to_save, os.path.join(save_vis_dir, f"{tag}_input.nii.gz"))
                    save_numpy_as_nii_gz(pred_to_save, os.path.join(save_vis_dir, f"{tag}_pred.nii.gz"))
                    save_numpy_as_nii_gz(new_pred_to_save, os.path.join(save_vis_dir, f"{tag}_new_pred.nii.gz"))
                    save_numpy_as_nii_gz(target_to_save, os.path.join(save_vis_dir, f"{tag}_target.nii.gz"))
                    save_numpy_as_nii_gz(delta_to_save, os.path.join(save_vis_dir, f"{tag}_delta_vis.nii.gz"))

    val_loss /= max(len(loader), 1)
    psnr_mean = sum(psnr_list) / max(len(psnr_list), 1)
    ssim_mean = sum(ssim_list) / max(len(ssim_list), 1)
    return val_loss, psnr_mean, ssim_mean


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, "configs", "default.yaml")
    cfg = load_yaml(cfg_path)

    set_seed(cfg["seed"])

    save_dir = cfg["train"]["save_dir"]
    ensure_dir(save_dir)

    vis_dir = os.path.join(save_dir, "val_vis")
    ensure_dir(vis_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PairedNiiDataset(
        cfg["dataset"]["train_input_dir"],
        cfg["dataset"]["train_target_dir"],
        cfg["dataset"]["fixed_h"],
        cfg["dataset"]["fixed_w"],
        cfg["dataset"]["normalize_mode"],
        tuple(cfg["dataset"]["fixed_range"]),
        tuple(cfg["dataset"].get("aux_max_values", [1000.0, 2000.0])),
    )
    val_ds = PairedNiiDataset(
        cfg["dataset"]["val_input_dir"],
        cfg["dataset"]["val_target_dir"],
        cfg["dataset"]["fixed_h"],
        cfg["dataset"]["fixed_w"],
        cfg["dataset"]["normalize_mode"],
        tuple(cfg["dataset"]["fixed_range"]),
        tuple(cfg["dataset"].get("aux_max_values", [1000.0, 2000.0])),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    model = ResidualToneMapper(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
        delta_scale=cfg["model"]["delta_scale"],
    ).to(device)

    criterion = ToneMappingLoss(**cfg["loss"]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )

    best_ssim = -1.0
    global_step = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}"
        )

        for it, batch in pbar:
            inp = batch["input"].to(device, non_blocking=True)
            inp_base = batch["input_base"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)

            pred, delta = model(inp)
            loss, logs = criterion(pred, tgt, delta, inp_base)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1

            if (it + 1) % cfg["train"]["log_interval"] == 0 or (it + 1) == len(train_loader):
                pbar.set_postfix({
                    "step": global_step,
                    "loss": f"{logs['loss_total']:.4f}",
                    "l1": f"{logs['loss_l1']:.4f}",
                    "ssim": f"{logs['loss_ssim']:.4f}",
                    "grad": f"{logs['loss_grad']:.4f}",
                    "grad_in": f"{logs['loss_grad_input']:.4f}",
                    "keep": f"{logs['loss_keep']:.4f}",
                    "tv": f"{logs['loss_delta_tv']:.4f}",
                })

        scheduler.step()

        if (epoch + 1) % cfg["train"]["val_interval"] == 0:
            val_loss, val_psnr, val_ssim = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                save_vis_dir=vis_dir,
                epoch=epoch + 1,
                global_step=global_step,
            )

            print(
                f"[Val] epoch={epoch + 1} "
                f"step={global_step} "
                f"loss={val_loss:.4f} "
                f"psnr={val_psnr:.4f} "
                f"ssim={val_ssim:.4f}"
            )

            ckpt = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg,
            }

            torch.save(ckpt, os.path.join(save_dir, f"epoch{epoch + 1:03d}.pth"))
            torch.save(ckpt, os.path.join(save_dir, "latest.pth"))

            if val_ssim > best_ssim:
                best_ssim = val_ssim
                torch.save(ckpt, os.path.join(save_dir, "best.pth"))
                print(
                    f"Saved best model at epoch {epoch + 1}, "
                    f"step={global_step}, ssim={best_ssim:.4f}"
                )


if __name__ == "__main__":
    main()
