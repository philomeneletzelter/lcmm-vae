"""Config overrides, the LCMM-VAE / LCMM-VAE (G=1) fits and result rows of the runners."""
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import trainer
from src.closed_forms import xtx_per_subject, latent_mean
from src.data_io import read_data
from src.datasets import get_dataloaders, visit_dataset
from src.initialisation import initialise
from src.model import build_model
from src.utils import ROOT, load_config as _load_config, set_seed, make_optimiser

OURS, G1, R0 = "LCMM-VAE", "LCMM-VAE (G=1)", "LCMM-VAE (R=0)"
METHODS = (OURS, R0, G1)  # table order
RUN_NAMES = {OURS: "lcmm_vae", G1: "lcmm_vae_g1", R0: "lcmm_vae_r0"}
RESULTS = str(ROOT / "results")


def load_config(name, seed, data_path=None, R=None, latent_dim=None):
    """
    src.utils.load_config with overrides; None keeps the config's value.

    Args:
    - name (str): config name in configs/, or a YAML path.
    - seed (int): training seed.
    - data_path (str): training data directory.
    - R (int): 0 drops the random-effect terms; otherwise the config must have one.
    - latent_dim (int): L.
    """
    config = _load_config(name, seed)
    if data_path is not None:
        config['dataset']['data_path'] = str(ROOT / data_path)
    terms = config['dataset']['covariates']['terms']
    if R == 0:
        config['dataset']['covariates']['terms'] = [term for term in terms
                                                    if term['block'] != 'random']
    elif R is not None and sum(term['block'] == 'random' for term in terms) != 1:
        raise ValueError(f"R={R} needs one random-effect term in {name}")
    if latent_dim is not None:
        config['model']['latent_dim'] = latent_dim
    return config


def to_raw(y, scaler):
    """Undo the training standardisation. [..., K]"""
    return np.asarray(y) * scaler['std'] + scaler['mean']


@torch.no_grad()
def posterior(model, loader, sweeps):
    """
    E-step on the visits of `loader`; mean-field sweeps start at pi.

    Returns:
    - gamma, m_g (Tensor): [N, G], [G, N, L, dR]
    - cache (dict): the encoder pass, from encode_loader.
    """
    cache = model.encode_loader(loader)
    N = loader.dataset.n_subjects
    XtX_R = xtx_per_subject(cache['xR'], cache['sid'], N)
    pi_start = model.prior.pi.detach().unsqueeze(0).expand(N, -1).clone()
    gamma, m_g, _, _ = model.e_step(cache, XtX_R, pi_start, sweeps)
    return gamma, m_g, cache


@torch.no_grad()
def reconstruct(result, x=None, sid=None):
    """
    Reconstruction (eq. hybrid) of the subjects of result['loader'] by result['recon'] after
    an E-step on their visits, at their visits or at design rows x of subjects sid.

    Args:
    - result (dict): a fits result.
    - x (Tensor): design rows [x_R | x_S]; None uses the loader's visits. [M, dR + dS]
    - sid (Tensor): subject code of each row of x. [M]

    Returns:
    - y_hat (Tensor): standardised reconstruction, on the CPU. [M, K]
    """
    model = result['recon']
    gamma, m_g, cache = posterior(model, result['loader'],
                                  result['config']['model']['estep_sweeps_infer'])
    if x is None:
        xS, xR, sid = cache['xS'], cache['xR'], cache['sid']
    else:
        xS, xR = model.split_x(x.to(model.log_vy.device))
        sid = sid.to(model.log_vy.device)
    _, B_g, _ = model.prior.class_coefficients()
    return model.decoder(latent_mean(B_g, gamma, m_g, xS, xR, sid)).cpu()


def eval_loader(config, data_dir, scaler, design):
    """Visits of a held-out data directory in file order, on the training scale."""
    Y, M, labels = read_data(data_dir)
    dataset, _ = visit_dataset(Y.astype('float32'), M.astype('float32'), labels, scaler,
                               config['dataset']['covariates'], design)
    return DataLoader(dataset, batch_size=config['dataset']['batch_size'], shuffle=False)


def _train(config, model, loader, save_dir, epochs=None):
    """trainer.train to the DEC stop, or for exactly `epochs` epochs; returns the epochs run."""
    history = trainer.train(
        model=model,
        optimiser=make_optimiser(model.parameters(), lr=config['training']['lr']),
        loader=loader,
        epochs=config['training']['epochs'] if epochs is None else epochs,
        tol=config['training']['tol'] if epochs is None else None,
        estep_sweeps=config['model']['estep_sweeps'],
        estep_sweeps_infer=config['model']['estep_sweeps_infer'],
        save_dir=save_dir)
    return len(history['train_loss'])


def _assign(result, model, test_dir):
    """Add the held-out loader, gamma [N, G], labels [N] and subject_ids [N] to result."""
    loader = (result['heldout_loader'] if test_dir is None else
              eval_loader(result['config'], test_dir, result['scaler'], result['design']))
    gamma = posterior(model, loader, result['config']['model']['estep_sweeps_infer'])[0]
    gamma = gamma.cpu().numpy()
    return dict(result, loader=loader, gamma=gamma, labels=gamma.argmax(1),
                subject_ids=loader.dataset.subject_ids)


def fit_lcmm_vae(config, seed, test_dir, out_dir):
    """
    LCMM-VAE fitted on the config's data (the pipeline of LCMMVAE.fit, keeping the
    heldout_fraction split), assigning the subjects of test_dir (None: the held-out split).

    Returns:
    - result (dict): train_loader, heldout_loader, widths, scaler, design, pretrained,
      model, recon (model), epochs, secs, config, the held-out assignment (_assign) and what
      fit_g1 reuses.
    """
    t0 = time.time()
    train_loader, heldout_loader, widths, scaler, design = get_dataloaders(config)
    scaler = {k: v.astype(np.float32) for k, v in scaler.items()}
    # advances the loader's shuffle generator: reproduces the published runs
    next(iter(train_loader))
    set_seed(seed)
    pretrained_model = build_model(config, train_loader.dataset.n_markers, widths)
    trainer.pretrain(
        model=pretrained_model,
        optimiser=make_optimiser(pretrained_model.parameters(),
                                 lr=config['pretraining']['lr']),
        loader=train_loader,
        epochs=config['pretraining']['epochs'],
        kl_annealing_epochs=config['pretraining']['kl_annealing_epochs'],
        save_dir=os.path.join(out_dir, 'pretrain'))
    result = dict(train_loader=train_loader, heldout_loader=heldout_loader, widths=widths,
                  scaler=scaler, design=design, pretrained=pretrained_model, config=config,
                  pretrain_secs=time.time() - t0,
                  shuffle_state=train_loader.generator.get_state())
    model = initialise(config=config, pretrained_model=pretrained_model,
                       train_loader=train_loader, widths=widths,
                       G=config['model']['n_clusters'], seed=seed)
    epochs = _train(config, model, train_loader, os.path.join(out_dir, 'lcmm'))
    result = _assign(dict(result, model=model, recon=model, epochs=epochs), model, test_dir)
    return dict(result, secs=time.time() - t0)


def fit_g1(lcmm_vae, seed, test_dir, out_dir):
    """
    LCMM-VAE (G=1): trained at G=1 from the pretraining of `lcmm_vae` (fit_lcmm_vae's), on
    its minibatch sequence and for its LCMM-phase epochs, then partitioned by a G-class
    em-EM start without further training.

    Returns:
    - result (dict): as fit_lcmm_vae, recon being the G=1 model.
    """
    t0 = time.time()
    config, train_loader = lcmm_vae['config'], lcmm_vae['train_loader']
    train_loader.generator.set_state(lcmm_vae['shuffle_state'])
    g1_model = initialise(config=config, pretrained_model=lcmm_vae['pretrained'],
                          train_loader=train_loader, widths=lcmm_vae['widths'], G=1,
                          seed=seed)
    epochs = _train(config, g1_model, train_loader, os.path.join(out_dir, 'lcmm'),
                    epochs=int(lcmm_vae['epochs']))
    model = initialise(config=config, pretrained_model=g1_model, train_loader=train_loader,
                       widths=lcmm_vae['widths'], G=config['model']['n_clusters'], seed=seed)
    result = _assign(dict(lcmm_vae, model=model, recon=g1_model, epochs=epochs), model,
                     test_dir)
    return dict(result, secs=lcmm_vae['pretrain_secs'] + time.time() - t0)


def fits(methods, load, seed, test_dir, run_dir):
    """
    Yield (method, fit_lcmm_vae-style result) in run order; LCMM-VAE (G=1) continues from
    the LCMM-VAE fit, fitted for it when not requested.

    Args:
    - methods (list): among OURS, G1, R0.
    - load (callable): method -> config.
    - seed (int): training seed.
    - test_dir (str): held-out data directory; None uses the heldout_fraction split.
    - run_dir (callable): method -> checkpoint directory.
    """
    ours = None
    for name in sorted(methods, key=[OURS, G1, R0].index):
        if name == G1:
            ours = ours or fit_lcmm_vae(load(OURS), seed, test_dir, run_dir(OURS))
            yield name, fit_g1(ours, seed, test_dir, run_dir(G1))
            continue
        result = fit_lcmm_vae(load(name), seed, test_dir, run_dir(name))
        if name == OURS:
            ours = result
        yield name, result


def append_row(results_dir, dataset, task, row):
    """Append one row to <results_dir>/<dataset>/<task>.csv."""
    path = os.path.join(results_dir, dataset, f"{task}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = pd.DataFrame([row])
    if os.path.exists(path):
        rows = pd.concat([pd.read_csv(path, float_precision="round_trip"), rows],
                         ignore_index=True)
    rows.to_csv(path, index=False)
