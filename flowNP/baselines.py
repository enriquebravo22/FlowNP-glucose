''' CNP / ANP / CANP / TNP (TNPA, TNP-D) on CGM glucose windows (GlucoBench), using the
shared data/eval code in glucose_np/. Same structure as glucose.py (FlowNP), for persona C.

Models (--model) and the name they get in glucose_np/results/ (--model_name to change it):
  cnp   CNP                      Gaussian, one pass            -> "CNP"
  anp   ANP (latent path)        samples over z + obs. noise   -> "ANP"
  canp  CANP (attentive, no z)   Gaussian, one pass            -> "CANP" (use --model_name=ANP
                                                                  if it replaces the latent ANP)
  tnpa  TNP autoregressive       sequential samples            -> "TNP"   (used for H3 / E4)
  tnpd  TNP-D (diagonal)         Gaussian, one pass            -> "TNP-D"

Modes:
  train  TrainSampler + early stopping on the mean target log-likelihood over the fixed
         e1_val tasks (best.tar) + resumable ckpt.tar
  eval   evaluate best.tar on all experiments -> glucose_np/results/<model_name>/seed<k>/<task>.json
  e4     sampling time on e1_test (T4, 100 samples, batch 16, same as FlowNP)
         -> glucose_np/results/<model_name>/e4_timing.json

Examples:
  python baselines.py --model=cnp --mode=train --seed=0
  python baselines.py --model=cnp --mode=eval --seed=0
  python baselines.py --model=tnpa --mode=e4 --seed=0
'''

import os
import os.path as osp
import sys
from sys import argv
import argparse
import json
import time
import yaml
import torch
from tqdm import tqdm

from utils.misc import load_module, logmeanexp
from utils.paths import results_path
from utils.log import get_logger, RunningAverage
from glucose import eval_arg, set_seed, sync

ROOT = osp.dirname(osp.abspath(__file__))

ALL_TASKS = ['e1_test', 'e2_nc5', 'e2_nc10', 'e2_nc20', 'e2_nc50',
             'e3_gap30', 'e3_gap60', 'e3_gap120', 'e3_gap180', 'e3_fc30', 'e3_fc60']

# model -> (file in models/, class, default name in results/, kind)
#   kind: 'gauss' one-pass Gaussian predictive, 'latent' ANP, 'autoreg' TNPA
MODELS = {
    'cnp':  ('cnp.py',  'CNP',  'CNP',   'gauss'),
    'canp': ('canp.py', 'CANP', 'CANP',  'gauss'),
    'anp':  ('anp.py',  'ANP',  'ANP',   'latent'),
    'tnpa': ('tnpa.py', 'TNPA', 'TNP',   'autoreg'),
    'tnpd': ('tnpd.py', 'TNPD', 'TNP-D', 'gauss'),
}


def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--model', required=True, choices=list(MODELS))
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'e4'])
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--seed', type=int, default=0, help='training seed (0, 1, 2)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--data_root', type=str, default=osp.join(ROOT, '..', 'glucose_np'),
                        help='path to the shared glucose_np folder (persona A)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='name used in glucose_np/results/ (default: see MODELS)')

    # Model
    parser.add_argument('--modelconfig', type=str, default=None,
                        help='default: configs/glucose/<model>.yaml')
    parser.add_argument('--model_args', type=str, default=None,
                        help='comma separated key=value overrides of the model config')

    # Train
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--train_num_samples', type=int, default=4, help='ANP: z samples (IWAE loss)')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=2000)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--val_num_samples', type=int, default=50, help='ANP: z samples for val LL')

    # Eval
    parser.add_argument('--tasks', type=str, default=','.join(ALL_TASKS))
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--num_samples', type=int, default=100, help='ANP / TNPA samples per task')
    parser.add_argument('--max_batches', type=int, default=None, help='debug: only first batches')
    parser.add_argument('--overwrite', action='store_true', help='re-evaluate tasks already in results/')

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

    fname, cls, default_name, args.kind = MODELS[args.model]
    args.model_name = args.model_name or default_name
    args.root = osp.join(results_path, 'glucose', args.expname, args.model, f'seed{args.seed}')
    args.modelconfig = args.modelconfig or osp.join(ROOT, 'configs/glucose', f'{args.model}.yaml')

    with open(args.modelconfig, 'r') as f:
        args.model_config = yaml.safe_load(f) or {}
    if args.model_args is not None:
        for arg in args.model_args.split(','):
            key, value = arg.split('=')
            args.model_config[key] = eval_arg(value)

    model_cls = getattr(load_module(osp.join(ROOT, 'models', fname)), cls)
    model = model_cls(**args.model_config).to(args.device)

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        evaluate(args, model)
    elif args.mode == 'e4':
        e4(args, model)


#####################################
# Predictive distribution of each kind of model
#####################################

def target_ll(model, batch, kind, num_samples):
    ''' Log-likelihood of each target point [B, Nt] in normalized space.
      gauss:   log N(yt; mu, sigma) of the one-pass predictive
      latent:  log (1/S) sum_s N(yt; mu_s, sigma_s), z_s ~ p(z | context)  (per-point marginal)
      autoreg: teacher-forced autoregressive conditionals; their sum over Nt is the joint
               log p(yt | context) in the order of the targets (time order) '''
    if kind == 'gauss':
        return model.predict(batch.xc, batch.yc, batch.xt).log_prob(batch.yt).sum(-1)
    if kind == 'latent':
        py = model.predict(batch.xc, batch.yc, batch.xt, num_samples=num_samples)
        return logmeanexp(py.log_prob(batch.yt.unsqueeze(0)).sum(-1), dim=0)
    return model(batch, reduce_ll=False).tar_ll


def tnpa_sample(model, batch, num_samples):
    ''' Autoregressive samples [S, B, Nt, 1]: Nt sequential passes.

    TNPA.sample() with permute=True generates the targets in a random order per sample
    and returns them in that PERMUTED order (only mean/std are de-permuted), so the
    samples would not match batch.yt. Sample in time order instead (permute off). '''
    permute, model.permute = model.permute, False
    try:
        return model.sample(batch.xc, batch.yc, batch.xt, num_samples=num_samples)
    finally:
        model.permute = permute


def latent_sample(model, batch, num_samples):
    ''' ANP samples [S, B, Nt, 1] including observation noise (ANP.sample() returns
    only the decoder mean). One z per sample, shared by all targets of the task. '''
    return model.predict(batch.xc, batch.yc, batch.xt, num_samples=num_samples).sample()


#####################################
# Train
#####################################

def val_ll(model, args, task='e1_val', max_batches=None):
    ''' Mean target log-likelihood per point on the fixed e1_val tasks (normalized space).
    Fixed seed so the ANP z samples are the same for every checkpoint. '''
    set_seed(0, args.device)
    tot, n = 0., 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(task_batches(task, batch_size=64, device=args.device)):
            if max_batches is not None and i >= max_batches:
                break
            ll = target_ll(model, batch, args.kind, args.val_num_samples)
            tot += ll.sum().item()
            n += ll.numel()
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
        start_step, best_val = 1, -float('inf')

    logger = get_logger(logfilename)
    ravg = RunningAverage()
    tag = f'glucose:{args.model}:seed{args.seed}'
    if not args.resume:
        logger.info(f"Experiment: glucose-{args.expname}-{args.model}-seed{args.seed}")
        logger.info(f"Command line: {' '.join(argv)}")
        logger.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}\n')

    # ANP's forward needs num_samples in train mode (it fails with None)
    num_samples = args.train_num_samples if args.kind == 'latent' else None

    for step in range(start_step, args.num_steps + 1):
        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(batch_size=args.train_batch_size, device=args.device)
        outs = model(batch, num_samples=num_samples)
        outs.loss.backward()
        optimizer.step()
        scheduler.step()
        ravg.update('loss', outs.loss)

        if step % args.print_freq == 0:
            logger.info(f'{tag} step {step} lr {optimizer.param_groups[0]["lr"]:.3e} '
                        f'[train_loss] {ravg.info()}')
            ravg.reset()

        if step % args.eval_freq == 0 or step == args.num_steps:
            val = val_ll(model, args)
            line = f'{tag} step {step} [val_ll] {val:.5f}'
            if val > best_val:
                best_val = val
                torch.save({'model': model.state_dict(), 'step': step, 'val_ll': val,
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
    print(f"Loaded best.tar (step {ckpt['step']}, val_ll {ckpt['val_ll']:.5f})")
    return ckpt


def add_batch(ev, model, batch, args):
    ''' Predict one batch and pass it to the Evaluator (everything in normalized space). '''
    if args.kind == 'gauss':
        py = model.predict(batch.xc, batch.yc, batch.xt)
        ev.add_gaussian(batch, py.loc, py.scale)
    elif args.kind == 'latent':
        py = model.predict(batch.xc, batch.yc, batch.xt, num_samples=args.num_samples)
        ll = logmeanexp(py.log_prob(batch.yt.unsqueeze(0)).sum(-1), dim=0)
        ev.add_samples(batch, py.sample(), ll)
    else:
        ev.add_samples(batch, tnpa_sample(model, batch, args.num_samples),
                       model(batch, reduce_ll=False).tar_ll)


def evaluate(args, model):
    load_best(args, model)
    os.makedirs(args.root, exist_ok=True)
    logger = get_logger(osp.join(args.root, 'eval.log'))
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
                add_batch(ev, model, batch, args)
        o = ev.save()['overall']
        logger.info(f"{task:10s} LL {o['loglik_mgdl']:7.3f} | RMSE {o['rmse_mgdl']:6.2f} | "
                    f"MAE {o['mae_mgdl']:6.2f} | CRPS {o['crps_mgdl']:6.2f} | "
                    f"cov90 {o['cov90']:.3f} | {time.time() - t0:.0f}s")


#####################################
# E4: sampling time (same protocol as FlowNP in glucose.py)
#####################################

def e4(args, model):
    ''' Time to draw num_samples samples per task on e1_test with batch 16 (pass
    --eval_batch_size=16 to match FlowNP). TNPA needs one sequential pass per target
    point, so 'steps' = mean Nt: it is drawn as a point on FlowNP's steps axis.
    One-pass models get steps = 1 (sampling = one forward pass + Normal.sample). '''
    load_best(args, model)
    os.makedirs(args.root, exist_ok=True)
    logger = get_logger(osp.join(args.root, 'e4.log'))
    logger.info(f"Command line: {' '.join(argv)}")
    device_name = torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu'

    batches = list(task_batches('e1_test', batch_size=args.eval_batch_size, device=args.device))
    if args.max_batches is not None:
        batches = batches[:args.max_batches]
    n_tasks = sum(len(b.task_id) for b in batches)
    mean_nt = sum(b.xt.shape[0] * b.xt.shape[1] for b in batches) / n_tasks

    def draw(batch):
        if args.kind == 'autoreg':
            return tnpa_sample(model, batch, args.num_samples)
        if args.kind == 'latent':
            return latent_sample(model, batch, args.num_samples)
        return model.predict(batch.xc, batch.yc, batch.xt).sample([args.num_samples])

    set_seed(2000 + args.seed, args.device)
    with torch.no_grad():   # warm-up (cuda kernels, allocator)
        draw(batches[0])
    ev = Evaluator(task='e1_test', model=args.model_name, seed=args.seed)
    elapsed = 0.
    for batch in tqdm(batches, desc='e4', ascii=True):
        sync(args.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            samples = draw(batch)
        sync(args.device)
        elapsed += time.perf_counter() - t0
        ev.add_samples(batch, samples, None)
    o = ev.summary()['overall']
    steps = round(mean_nt, 1) if args.kind == 'autoreg' else 1
    row = {'steps': steps, 'method': args.model, 'nfe': steps,
           'time_per_task_s': elapsed / n_tasks,
           'crps_mgdl': o['crps_mgdl'], 'rmse_mgdl': o['rmse_mgdl'], 'cov90': o['cov90'],
           'loglik_mgdl': None, 'num_samples': args.num_samples,
           'batch_size': args.eval_batch_size, 'n_tasks': n_tasks, 'mean_nt': mean_nt,
           'device': device_name}
    logger.info(json.dumps(row))
    with open(osp.join(args.root, 'e4_all.json'), 'w') as f:
        json.dump([row], f, indent=2)
    if args.max_batches is None:
        save_timing(args.model_name, [row])


if __name__ == '__main__':
    main()
