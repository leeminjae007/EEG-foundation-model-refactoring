"""One W&B initializer shared by all experiment stages."""

from datetime import datetime
import os
import socket
import sys

import wandb

from src.utils.reproducibility import source_tree_sha256


def init_wandb(
    stage,
    dataset,
    trial_name,
    job_type,
    config,
    output_dir,
    entity,
    run_id,
    resume,
):
    tracking = config['tracking']
    project = tracking['project']
    assert config['experiment']['trial_name'] == trial_name
    assert config['tracking']['group'] == trial_name
    assert (run_id is not None) == resume
    source_sha256 = source_tree_sha256()
    started_at = datetime.now().astimezone()
    run_date = started_at.strftime('%Y-%m-%d')
    tracked_config = dict(config)
    tracked_config['source_tree_sha256'] = source_sha256
    tracked_config['execution'] = {
        'run_date': run_date,
        'started_at': started_at.isoformat(timespec='seconds'),
        'git_commit': None,
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'command': ' '.join(sys.argv),
        'checkpoint_source': config.get('model', {}).get('checkpoint'),
    }
    run_name_suffix = tracking.get('run_name_suffix')
    suffix = f"__{run_name_suffix}" if run_name_suffix else ''
    run_name = (
        f"{trial_name}__{stage}__"
        f"{config.get('model', {}).get('pooling', 'default')}__"
        f"seed{config['experiment']['seed']}{suffix}__{run_date}"
    )
    tags = tracking.get('tags') or [
        dataset,
        'eeg-mae',
        f"seed-{config['experiment']['seed']}",
        f'source-{source_sha256[:12]}',
    ]
    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=trial_name,
        job_type=job_type,
        tags=tags,
        config=tracked_config,
        dir=output_dir,
        id=run_id,
        resume='must' if resume else None,
        mode=tracking['mode'],
    )
