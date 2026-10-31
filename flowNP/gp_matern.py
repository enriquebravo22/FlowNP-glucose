''' GP baseline for the glucose project: exact GP per task with a Matérn 5/2 kernel,
constant mean and Gaussian observation noise. The hyperparameters are GLOBAL (shared by
all tasks): fitted once by maximizing the exact marginal likelihood of random train
windows, then each test task is conditioned on its own context.

Modes:
  fit   fit the hyperparameters with seed k (random subsample of train windows)
        -> output/results/glucose/<expname>/gp/seed<k>/hyper.json
  eval  -> glucose_np/results/GP/seed<k>/<task>.json  (add_gaussian, sigma includes noise)

Different seeds use different train subsamples, so the 3 seeds differ only slightly.
No GPU needed (a few minutes on CPU).

Examples:
  python gp_matern.py --mode=fit --seed=0
  python gp_matern.py --mode=eval --seed=0
'''

import os
import os.path as osp
import sys
import argparse
import json
import math
import time
import torch
from tqdm import tqdm

from utils.paths import results_path
from glucose import set_seed

ROOT = osp.dirname(osp.abspath(__file__))

ALL_TASKS = ['e1_test', 'e2_nc5', 'e2_nc10', 'e2_nc20', 'e2_nc50',
             'e3_gap30', 'e3_gap60', 'e3_gap120', 'e3_gap180', 'e3_fc30', 'e3_fc60']

JITTER = 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='fit', choices=['fit', 'eval'])
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--data_root', type=str, default=osp.join(ROOT, '..', 'glucose_np'))
    parser.add_argument('--model_name', type=str, default='GP')
    # Fit
    parser.add_argument('--fit_steps', type=int, default=1000)
    parser.add_argument('--fit_batch_size', type=int, default=256, help='train windows per step')
    parser.add_argument('--lr', type=float, default=0.02)
    # Eval
    parser.add_argument('--tasks', type=str, default=','.join(ALL_TASKS))
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--max_batches', type=int, default=None, help='debug: only first batches')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")

    sys.path.append(osp.abspath(args.data_root))
    global C, load_windows, TrainSampler, task_batches, Evaluator
    import config as C
    from config import load_windows
    from sampling import TrainSampler, task_batches
    from metrics import Evaluator

    args.root = osp.join(results_path, 'glucose', args.expname, 'gp', f'seed{args.seed}')
    os.makedirs(args.root, exist_ok=True)
    if args.mode == 'fit':
        fit(args)
    else:
        evaluate(args)


class MaternGP(torch.nn.Module):
    ''' Matérn 5/2 kernel  k(r) = s2 (1 + a + a^2/3) exp(-a),  a = sqrt(5) r / l,
    with constant mean and noise. Parameters in log space; float64 for stable Cholesky. '''
    def __init__(self, lengthscale=0.1, outputscale=1.0, noise=0.01, mean=0.0):
        super().__init__()
        d = dict(dtype=torch.float64)
        self.log_l = torch.nn.Parameter(torch.tensor(math.log(lengthscale), **d))
        self.log_s2 = torch.nn.Parameter(torch.tensor(math.log(outputscale), **d))
        self.log_noise = torch.nn.Parameter(torch.tensor(math.log(noise), **d))
        self.mean = torch.nn.Parameter(torch.tensor(mean, **d))

    def kernel(self, x1, x2):
        a = math.sqrt(5) * (x1 - x2.transpose(-1, -2)).abs() / self.log_l.exp()
        return self.log_s2.exp() * (1 + a + a ** 2 / 3) * torch.exp(-a)

    def chol(self, x):
        K = self.kernel(x, x)
        eye = torch.eye(x.shape[-2], dtype=x.dtype, device=x.device)
        return torch.linalg.cholesky(K + (self.log_noise.exp() + JITTER) * eye)

    def mll(self, x, y):
        ''' Exact log marginal likelihood per point, averaged over the batch. x, y: [B, N, 1] '''
        L = self.chol(x)
        r = y - self.mean
        alpha = torch.linalg.solve_triangular(L, r, upper=False)
        N = x.shape[-2]
        ll = -0.5 * (alpha ** 2).sum((-2, -1)) - torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1) \
            - 0.5 * N * math.log(2 * math.pi)
        return (ll / N).mean()

    def predict(self, xc, yc, xt):
        ''' Posterior predictive at xt given (xc, yc): mean and std INCLUDING noise. [B, Nt, 1] '''
        L = self.chol(xc)
        Kct = self.kernel(xc, xt)                                         # [B, Nc, Nt]
        v = torch.linalg.solve_triangular(L, Kct, upper=False)            # [B, Nc, Nt]
        alpha = torch.linalg.solve_triangular(L, yc - self.mean, upper=False)
        mu = self.mean + v.transpose(-1, -2) @ alpha                      # [B, Nt, 1]
        var_f = (self.log_s2.exp() - (v ** 2).sum(-2)).clamp_min(1e-12)   # [B, Nt]
        sigma = (var_f + self.log_noise.exp()).sqrt().unsqueeze(-1)
        return mu, sigma

    def hyper(self):
        return {'lengthscale': self.log_l.exp().item(), 'outputscale': self.log_s2.exp().item(),
                'noise': self.log_noise.exp().item(), 'mean': self.mean.item(),
                # in physical units, for the report
                'lengthscale_min': self.log_l.exp().item() / 2 * C.WINDOW_MIN}


def fit(args):
    set_seed(args.seed, args.device)
    sampler = TrainSampler(load_windows('train'), seed=args.seed)
    gp = MaternGP().to(args.device)
    opt = torch.optim.Adam(gp.parameters(), lr=args.lr)
    t0 = time.time()
    for step in range(1, args.fit_steps + 1):
        b = sampler.sample(batch_size=args.fit_batch_size, device=args.device)
        x, y = b.x.double(), b.y.double()          # whole window (context + target)
        opt.zero_grad()
        loss = -gp.mll(x, y)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == args.fit_steps:
            h = gp.hyper()
            print(f'step {step} mll/point {-loss.item():.4f} | l {h["lengthscale"]:.4f} '
                  f'({h["lengthscale_min"]:.0f} min) s2 {h["outputscale"]:.3f} '
                  f'noise {h["noise"]:.5f} mean {h["mean"]:.3f}')
    h = gp.hyper()
    h.update({'seed': args.seed, 'fit_steps': args.fit_steps, 'fit_batch_size': args.fit_batch_size,
              'final_mll_per_point': -loss.item(), 'time_s': time.time() - t0})
    with open(osp.join(args.root, 'hyper.json'), 'w') as f:
        json.dump(h, f, indent=2)
    print(f"Saved {osp.join(args.root, 'hyper.json')}")


def evaluate(args):
    with open(osp.join(args.root, 'hyper.json')) as f:
        h = json.load(f)
    gp = MaternGP(h['lengthscale'], h['outputscale'], h['noise'], h['mean']).to(args.device)
    print(f"Loaded hyper.json: l {h['lengthscale']:.4f} s2 {h['outputscale']:.3f} "
          f"noise {h['noise']:.5f} mean {h['mean']:.3f}")

    for task in args.tasks.split(','):
        out = osp.join(C.ROOT, 'results', args.model_name, f'seed{args.seed}', f'{task}.json')
        if osp.exists(out) and not args.overwrite:
            print(f'{task:10s} already in {out}, skipping (use --overwrite)')
            continue
        ev = Evaluator(task=task, model=args.model_name, seed=args.seed)
        t0 = time.time()
        batches = list(task_batches(task, batch_size=args.eval_batch_size, device=args.device))
        if args.max_batches is not None:
            batches = batches[:args.max_batches]
        for batch in tqdm(batches, desc=task, ascii=True):
            with torch.no_grad():
                mu, sigma = gp.predict(batch.xc.double(), batch.yc.double(), batch.xt.double())
            ev.add_gaussian(batch, mu, sigma)
        o = ev.save()['overall']
        print(f"{task:10s} LL {o['loglik_mgdl']:7.3f} | RMSE {o['rmse_mgdl']:6.2f} | "
              f"MAE {o['mae_mgdl']:6.2f} | CRPS {o['crps_mgdl']:6.2f} | "
              f"cov90 {o['cov90']:.3f} | {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
