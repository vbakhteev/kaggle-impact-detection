import torch

from src.data.loaders import get_image_dataloaders
from src.model.detector import get_effdet
from src.utils import seed_everything
from train import Fitter
from config import data as data_cfg, model as model_cfg, train as train_cfg


def main():
    seed_everything()
    train()


def train():
    loader_train, loaders_valid_fn = get_image_dataloaders(data_cfg, train_cfg)

    device = torch.device('cuda:0')
    net = get_effdet(model_cfg, num_classes=2).to(device)
    fitter = Fitter(model=net, device=device)

    try:
        fitter.fit(loader_train, loaders_valid_fn)
    except KeyboardInterrupt:
        pass

    fitter.store_best_parameters()


if __name__ == '__main__':
    main()