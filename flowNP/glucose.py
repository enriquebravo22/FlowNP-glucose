''' FlowNP on CGM glucose windows (GlucoBench), using the shared data/eval code in glucose_np/.

Modes:
  train  train FNP with glucose_np's TrainSampler; early stopping on the flow-matching
         loss over the fixed e1_val tasks (best.tar) + resumable ckpt.tar.
  eval   evaluate best.tar on all experiments with glucose_np's Evaluator
         -> glucose_np/results/<model_name>/seed<k>/<task>.json
  e4     inference time vs number of solver steps on e1_test
         -> glucose_np/results/<model_name>/e4_timing.json (+ detailed json in args.root)

Examples:
  python glucose.py --mode=train --seed=0
  python glucose.py --mode=eval --seed=0
  python glucose.py --mode=e4 --seed=0 --e4_steps=1,2,5,10,20,50,100
'''

import os
import os.path as osp
import sys
from sys import argv
import argparse
import json
import time
import yaml
import numpy as np
import torch
from tqdm import tqdm

from utils.misc import load_module
from utils.paths import results_path
from utils.log import get_logger, RunningAverage

ROOT = osp.dirname(osp.abspath(__file__))

ALL_TASKS = ['e1_test', 'e2_nc5', 'e2_nc10', 'e2_nc20', 'e2_nc50',
             'e3_gap30', 'e3_gap60', 'e3_gap120', 'e3_gap180', 'e3_fc30', 'e3_fc60']


def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'e4'])
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--seed', type=int, default=0, help='training seed (0, 1, 2)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--data_root', type=str, default=osp.join(ROOT, '..', 'glucose_np'),
                        help='path to the shared glucose_np folder (persona A)')
    parser.add_argument('--model_name', type=str, default='FlowNP',
                        help='name used in glucose_np/results/ (keep "FlowNP" for final runs)')

    # Model
    parser.add_argument('--modelconfig', type=str, default=osp.join(ROOT, 'configs/glucose/fnp.yaml'))
    parser.add_argument('--model_args', type=str, default=None,
                        help='comma separated key=value overrides of the model config')

    # Train
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=2000)
    parser.add_argument('--save_freq', type=int, default=1000)

    # Eval
    parser.add_argument('--tasks', type=str, default=','.join(ALL_TASKS))
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--sample_method', type=str, default='midpoint', choices=['euler', 'midpoint'])
    parser.add_argument('--sample_steps', type=int, default=None, help='defaults to model timesteps')
    parser.add_argument('--ll_method', type=str, default='midpoint', choices=['euler', 'midpoint'])
    parser.add_argument('--ll_steps', type=int, default=None, help='defaults to model timesteps')
    parser.add_argument('--ll_div', type=str, default='hutchinson', choices=['hutchinson', 'exact'])
    parser.add_argument('--ll_probes', type=int, default=1, help='Hutchinson probes')
    parser.add_argument('--no_ll', action='store_true', help='skip the (expensive) ODE log-likelihood')
    parser.add_argument('--max_batches', type=int, default=None, help='debug: only first batches')
    parser.add_argument('--overwrite', action='store_true', help='re-evaluate tasks already in results/')

    # E4
    parser.add_argument('--e4_steps', type=str, default='1,2,5,10,20,50,100,200')
    parser.add_argument('--e4_methods', type=str, default='euler,midpoint',
                        help='first one goes to save_timing (figure); all go to the detailed json')
    parser.add_argument('--e4_ll', action='store_true', help='also compute LL at each step count')

    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")

    # glucose_np modules (config, sampling, metrics) are imported from data_root
    sys.path.append(osp.abspath(args.data_root))
    global C, load_windows, TrainSampler, task_batches, Evaluator, save_timing
    import config as C
    from config import load_windows
    from sampling import TrainSampler, task_batches
    from metrics import Evaluator, save_timing

    args.root = osp.join(results_path, 'glucose', args.expname, 'fnp', f'seed{args.seed}')

    with open(args.modelconfig, 'r') as f:
        args.model_config = yaml.safe_load(f)
    if args.model_args is not None:
        for arg in args.model_args.split(','):
            key, value = arg.split('=')
            args.model_config[key] = eval_arg(value)

    model_cls = getattr(load_module(osp.join(ROOT, 'models/fnp.py')), 'FNP')
    model = model_cls(**args.model_config).to(args.device)

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        evaluate(args, model)
    elif args.mode == 'e4':
        e4(args, model)


def eval_arg(arg):
    if arg.lower() in ['true', 'false']:
        return arg.lower() == 'true'
    for cast in (int, float):
        try:
            return cast(arg)
        except ValueError:
            pass
    return arg


def set_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


#####################################
# Sampling and likelihood with the learned velocity field
#####################################

def repeat_batch(batch, num_samples):
    ''' Tile the batch num_samples times along dim 0 (sample-major): [S*B, ...]. '''
    out = type(batch)()
    for k in ['xc', 'yc', 'xt']:
        out[k] = batch[k].repeat(num_samples, 1, 1)
    return out


def ode_sample(model, batch, num_samples, steps, method='midpoint'):
    ''' Integrate dy/dt = v(y, t) from t=0 (noise) to t=1 (data) with a fixed-step solver.
    Returns samples [S, B, Nt, 1] in normalized space. Number of velocity evaluations
    is steps (euler) or 2*steps (midpoint). '''
    B, Nt = batch.xt.shape[:2]
    rb = repeat_batch(batch, num_samples)
    y = torch.randn(num_samples * B, Nt, batch.yc.shape[-1], device=batch.xt.device)
    v = lambda y, t: model.predict_velocity(y, torch.tensor(t, device=y.device), rb)
    h = 1.0 / steps
    for k in range(steps):
        t = k * h
        if method == 'euler':
            y = y + h * v(y, t)
        else:
            y_mid = y + 0.5 * h * v(y, t)
            y = y + h * v(y_mid, t + 0.5 * h)
    return y.reshape(num_samples, B, Nt, -1)


def velocity_and_div(model, batch, y, t, div='hutchinson', probes=None):
    ''' Velocity v(y, t) [B, Nt, 1] and the trace of its Jacobian w.r.t. the whole target
    of each task [B].

    All target points of a task are coupled through attention, so the divergence is the
    trace of the (Nt x Nt) Jacobian. Tasks in the batch are independent, so summing
    the outputs over the batch before differentiating is safe.
    (flow_matching's exact_divergence treats each of the B*Nt rows as independent and
    would return Jacobian column sums instead of the trace, hence this function.)
    '''
    with torch.enable_grad():
        y = y.detach().requires_grad_(True)
        v = model.predict_velocity(y, torch.tensor(t, device=y.device), batch)
        if div == 'exact':
            d = torch.zeros(y.shape[0], device=y.device)
            for i in range(y.shape[1]):
                g = torch.autograd.grad(v[:, i, 0].sum(), y, retain_graph=True)[0]
                d = d + g[:, i, 0]
        else:
            d = 0.
            for k, z in enumerate(probes):
                g = torch.autograd.grad((v * z).sum(), y, retain_graph=k < len(probes) - 1)[0]
                d = d + (g * z).sum(dim=(1, 2))
            d = d / len(probes)
    return v.detach(), d.detach()


def ode_loglik(model, batch, steps, method='midpoint', div='hutchinson', num_probes=1):
    ''' Joint log p(yt | context) [B] (normalized space) by integrating the probability
    flow ODE backwards from t=1 (data) to t=0 (noise):
        log p1(y1) = log N(y0; 0, I) - int_0^1 div v(y_t, t) dt '''
    y = batch.yt.clone()
    probes = [torch.randint(0, 2, y.shape, device=y.device).float() * 2 - 1
              for _ in range(num_probes)] if div == 'hutchinson' else None
    f = lambda y, t: velocity_and_div(model, batch, y, t, div, probes)
    h = 1.0 / steps
    int_div = torch.zeros(y.shape[0], device=y.device)
    for k in range(steps):
        t = 1.0 - k * h
        if method == 'euler':
            v, d = f(y, t)
            y = y - h * v
        else:
            with torch.no_grad():
                v1 = model.predict_velocity(y, torch.tensor(t, device=y.device), batch)
            v, d = f(y - 0.5 * h * v1, t - 0.5 * h)
            y = y - h * v
        int_div = int_div + h * d
    log_p0 = -0.5 * (y ** 2 + np.log(2 * np.pi)).sum(dim=(1, 2))
    return log_p0 - int_div


#####################################
# Train
#####################################

def fm_val_loss(model, device, task='e1_val', seed=0, max_batches=None):
    ''' Flow-matching loss on fixed tasks with fixed noise and flow times (deterministic,
    comparable across checkpoints). Mean over all target points. '''
    gen = torch.Generator().manual_seed(seed)
    tot, n = 0., 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(task_batches(task, batch_size=64, device=device)):
            if max_batches is not None and i >= max_batches:
                break
            y0 = torch.randn(batch.yt.shape, generator=gen).to(device)
            t = torch.rand(batch.yt.shape, generator=gen).to(device)
            yt = t * batch.yt + (1 - t) * y0
            pred = model.predict_velocity(yt, t, batch)
            tot += ((pred - (batch.yt - y0)) ** 2).sum().item()
            n += batch.yt.numel()
    return tot / n


def train(args, model):
    if osp.exists(osp.join(args.root, 'ckpt.tar')) and not args.resume:
        raise FileExistsError(f'{args.root} already has a checkpoint (use --resume)')
    os.makedirs(args.root, exist_ok=True)
    with open(osp.join(args.root, f'args_{time.strftime("%Y%m%d-%H%M")}.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    set_seed(args.seed, args.device)
    sampler = TrainSampler(load_windows('train'), seed=args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    if args.resume:
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'), map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        sampler.rng = ckpt['sampler_rng']
        torch.set_rng_state(ckpt['torch_rng'].cpu())   # ckpt may come from another device
        logfilename, start_step, best_val = ckpt['logfilename'], ckpt['step'], ckpt['best_val']
    else:
        logfilename = osp.join(args.root, f'train_{time.strftime("%Y%m%d-%H%M")}.log')
        start_step, best_val = 1, float('inf')

    logger = get_logger(logfilename)
    ravg = RunningAverage()
    if not args.resume:
        logger.info(f"Experiment: glucose-{args.expname}-fnp-seed{args.seed}")
        logger.info(f"Command line: {' '.join(argv)}")
        logger.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}\n')

    for step in range(start_step, args.num_steps + 1):
        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(batch_size=args.train_batch_size, device=args.device)
        outs = model(batch)
        outs.loss.backward()
        optimizer.step()
        scheduler.step()
        ravg.update('loss', outs.loss)

        if step % args.print_freq == 0:
            logger.info(f'glucose:fnp:seed{args.seed} step {step} '
                        f'lr {optimizer.param_groups[0]["lr"]:.3e} [train_loss] {ravg.info()}')
            ravg.reset()

        if step % args.eval_freq == 0 or step == args.num_steps:
            val = fm_val_loss(model, args.device)
            line = f'glucose:fnp:seed{args.seed} step {step} [val_fm_loss] {val:.5f}'
            if val < best_val:
                best_val = val
                torch.save({'model': model.state_dict(), 'step': step, 'val_fm_loss': val,
                            'model_config': args.model_config}, osp.join(args.root, 'best.tar'))
                line += ' (best)'
            logger.info(line)

        if step % args.save_freq == 0 or step == args.num_steps:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'sampler_rng': sampler.rng,
                        'torch_rng': torch.get_rng_state(), 'logfilename': logfilename,
                        'step': step + 1, 'best_val': best_val},
                       osp.join(args.root, 'ckpt.tar'))


#####################################
# Eval
#####################################

def load_best(args, model):
    ckpt = torch.load(osp.join(args.root, 'best.tar'), map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded best.tar (step {ckpt['step']}, val_fm_loss {ckpt['val_fm_loss']:.5f})")
    return ckpt


def evaluate(args, model):
    load_best(args, model)
    os.makedirs(args.root, exist_ok=True)
    logger = get_logger(osp.join(args.root, 'eval.log'))
    sample_steps = args.sample_steps or model.timesteps
    ll_steps = args.ll_steps or model.timesteps
    logger.info(f"Command line: {' '.join(argv)}")

    for task in args.tasks.split(','):
        out = osp.join(C.ROOT, 'results', args.model_name, f'seed{args.seed}', f'{task}.json')
        if osp.exists(out) and not args.overwrite:
            logger.info(f'{task:10s} already in {out}, skipping (use --overwrite)')
            continue
        set_seed(1000 + args.seed, args.device)
        ev = Evaluator(task=task, model=args.model_name, seed=args.seed)
        t0 = time.time()
        batches = list(task_batches(task, batch_size=args.eval_batch_size, device=args.device))
        if args.max_batches is not None:
            batches = batches[:args.max_batches]
        for batch in tqdm(batches, desc=task, ascii=True):
            with torch.no_grad():
                samples = ode_sample(model, batch, args.num_samples, sample_steps, args.sample_method)
            ll = None if args.no_ll else ode_loglik(model, batch, ll_steps, args.ll_method,
                                                     args.ll_div, args.ll_probes)
            ev.add_samples(batch, samples, ll)
        o = ev.save()['overall']
        logger.info(f"{task:10s} LL {o['loglik_mgdl']:7.3f} | RMSE {o['rmse_mgdl']:6.2f} | "
                    f"MAE {o['mae_mgdl']:6.2f} | CRPS {o['crps_mgdl']:6.2f} | "
                    f"cov90 {o['cov90']:.3f} | {time.time() - t0:.0f}s")


#####################################
# E4: inference time vs solver steps
#####################################

def e4(args, model):
    load_best(args, model)
    os.makedirs(args.root, exist_ok=True)
    logger = get_logger(osp.join(args.root, 'e4.log'))
    logger.info(f"Command line: {' '.join(argv)}")
    device_name = torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu'

    batches = list(task_batches('e1_test', batch_size=args.eval_batch_size, device=args.device))
    if args.max_batches is not None:
        batches = batches[:args.max_batches]
    n_tasks = sum(len(b.task_id) for b in batches)

    rows = {m: [] for m in args.e4_methods.split(',')}
    for method in rows:
        for steps in [int(s) for s in args.e4_steps.split(',')]:
            set_seed(2000 + args.seed, args.device)
            with torch.no_grad():   # warm-up (cuda kernels, allocator)
                ode_sample(model, batches[0], args.num_samples, steps, method)
            ev = Evaluator(task='e1_test', model=args.model_name, seed=args.seed)
            elapsed = 0.
            for batch in tqdm(batches, desc=f'{method} {steps}', ascii=True):
                sync(args.device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    samples = ode_sample(model, batch, args.num_samples, steps, method)
                sync(args.device)
                elapsed += time.perf_counter() - t0
                ll = ode_loglik(model, batch, steps, method, args.ll_div, args.ll_probes) \
                    if args.e4_ll else None
                ev.add_samples(batch, samples, ll)
            o = ev.summary()['overall']
            row = {'steps': steps, 'method': method,
                   'nfe': steps * (2 if method == 'midpoint' else 1),
                   'time_per_task_s': elapsed / n_tasks,
                   'crps_mgdl': o['crps_mgdl'], 'rmse_mgdl': o['rmse_mgdl'],
                   'cov90': o['cov90'],
                   'loglik_mgdl': o['loglik_mgdl'] if args.e4_ll else None,
                   'num_samples': args.num_samples, 'batch_size': args.eval_batch_size,
                   'n_tasks': n_tasks, 'device': device_name}
            rows[method].append(row)
            logger.info(json.dumps(row))

    with open(osp.join(args.root, 'e4_all.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    if args.max_batches is None:
        save_timing(args.model_name, rows[args.e4_methods.split(',')[0]])


if __name__ == '__main__':
    main()
