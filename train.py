"""Fit the LCMM-VAE on one dataset and save it."""
import argparse
import os

from src import LCMMVAE
from src.utils import ROOT, load_config


def main(args):
    config = load_config(args.config, args.seed)
    data = args.data or config['dataset']['data_path']
    if data is None:
        raise SystemExit(f"{args.config} sets no dataset.data_path: pass --data")
    name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = args.out or os.path.join(ROOT, 'results', 'train', name,
                                       f"seed{config['training']['seed']}")
    model = LCMMVAE(config).fit(data)
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, 'model.pt'))
    G = len(model.proportions_)
    class_sizes = model.labels_.value_counts().reindex(range(G), fill_value=0)
    print(f"{model.n_iter_} LCMM-phase epochs, training subjects per class "
          f"{class_sizes.tolist()}; written to {os.path.join(out_dir, 'model.pt')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit the LCMM-VAE (G=1 pretraining, em-EM initialisation, variational EM) "
                    "and save it to <out>/model.pt.")
    parser.add_argument('--config', default='configs/gait.yaml',
                        help="config name in configs/ or path to a YAML config")
    parser.add_argument('--data', default=None,
                        help="data directory; overrides dataset.data_path of the config")
    parser.add_argument('--seed', type=int, default=None, help="overrides training.seed")
    parser.add_argument('--out', default=None,
                        help="output directory (default results/train/<config>/seed<seed>)")
    main(parser.parse_args())
