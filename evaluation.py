from metrices import masked_mae_np, masked_mape_np, masked_rmse_np
import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def evaluation(model, data_util, is_test: bool, device):
    rmse_loss = []
    mae_loss = []
    mape_loss = []

    loader = data_util.test_loader if is_test else data_util.valid_loader
    for inputs, targets in loader:
        model.eval()

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)  # A_wave)

        out_unnorm = data_util.unnormal(outputs.detach().cpu().numpy())
        target_unnorm = data_util.unnormal(targets.detach().cpu().numpy())

        mae_loss.append(masked_mae_np(target_unnorm, out_unnorm, 0))
        mape_loss.append(masked_mape_np(target_unnorm, out_unnorm, 0))
        rmse_loss.append(masked_rmse_np(target_unnorm, out_unnorm, 0))

    return np.mean(mae_loss), np.mean(mape_loss), np.mean(rmse_loss)
