"""Optional Weights & Biases logging for the training scripts.

With ``--wandb`` the run mirrors every tensorboardX scalar (``sync_tensorboard=True``),
so the existing ``tb_log.add_scalar`` calls need no changes. wandb is imported lazily
and only on rank 0, so it stays an optional dependency for everyone else.
"""
from pathlib import Path


def add_wandb_args(parser):
    group = parser.add_argument_group('Weights & Biases')
    group.add_argument('--wandb', action='store_true', default=False,
                       help='also log to Weights & Biases (rank 0 only); needs WANDB_API_KEY or `wandb login`')
    group.add_argument('--wandb_project', type=str, default='detzero', help='W&B project')
    group.add_argument('--wandb_entity', type=str, default=None,
                       help='W&B user or team (default: the default entity of the API key)')
    group.add_argument('--wandb_name', type=str, default=None,
                       help='run name (default: <cfg tag>/<extra_tag>)')
    group.add_argument('--wandb_tags', type=str, nargs='+', default=None, help='run tags')


def _to_plain(obj):
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def init_wandb(args, cfg, output_dir, job_type):
    """Start a W&B run on rank 0 when ``--wandb`` is given, otherwise return None.

    Must be called before the first tensorboardX ``SummaryWriter`` is created, so that
    wandb can patch tensorboardX and pick up every scalar logged afterwards.
    """
    if not args.wandb or cfg.LOCAL_RANK != 0:
        return None

    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or '%s/%s' % (cfg.TAG, args.extra_tag),
        group=cfg.EXP_GROUP_PATH,
        job_type=job_type,
        tags=args.wandb_tags,
        config={'args': _to_plain(vars(args)), 'cfg': _to_plain(cfg)},
        dir=str(Path(output_dir)),
        sync_tensorboard=True,
    )
