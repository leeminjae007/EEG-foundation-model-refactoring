"""Add the optional policy hooks to a copied historical engine, never its source."""


def connected_source(original):
    if "def run_finetune(config, args, policy=None):" in original:
        if "policy.on_epoch_start(model, epoch)" not in original:
            raise ValueError("Incomplete existing policy connection")
        return original
    changes = [
        ("def run_finetune(config, args):", "def run_finetune(config, args, policy=None):"),
        ('    scheduler = GroupCosineScheduler(optimizer, optimization["epochs"] * updates,',
         '    scheduler_factory = GroupCosineScheduler if policy is None else policy.make_scheduler\n'
         '    scheduler = scheduler_factory(optimizer, optimization["epochs"] * updates,'),
        ('        trainer = DistributedDataParallel(model, device_ids=[device.index])',
         '        trainer = DistributedDataParallel(model, device_ids=[device.index],\n'
         '                                          find_unused_parameters=bool(policy and policy.head_first_epochs))'),
    ]
    if "def train_finetune_epoch(" in original:
        changes.extend([
            ('spec, config, args, device, rank, epoch, step, output)\n',
             'spec, config, args, device, rank, epoch, step, output, policy=policy)\n'),
            ('spec, config, args, device, rank, epoch, step, output):',
             'spec, config, args, device, rank, epoch, step, output, policy=None):'),
            ('    trainer.train()\n', '    trainer.train()\n    if policy is not None:\n        policy.on_epoch_start(model, epoch)\n'),
        ])
    else:
        changes.append(('        trainer.train()\n',
                        '        trainer.train()\n        if policy is not None:\n            policy.on_epoch_start(model, epoch)\n'))
    result = original
    for before, after in changes:
        if result.count(before) != 1:
            raise ValueError("Unknown historical engine layout: " + before)
        result = result.replace(before, after)
    compile(result, "connected_engine.py", "exec")
    return result
