import wandb


def setup_wandb(
    *,
    project: str,
    group: str,
    entity: str,
    name: str,
    tags: list[str],
    notes: str,
    config: dict,
):
    # Initialize wandb
    wandb.init(
        project=project,
        group=group,
        entity=entity,
        name=name,
        tags=tags,
        notes=notes,
        config=config,
    )

    # Watch the model for gradients and parameters
    return wandb
