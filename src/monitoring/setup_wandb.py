import wandb
from wandb.wandb_run import Run


def setup_wandb(
    *,
    project: str,
    group: str,
    entity: str | None,
    name: str,
    tags: list[str],
    notes: str,
    config: dict,
) -> Run:
    # Initialize wandb
    return wandb.init(
        project=project,
        group=group,
        entity=entity,
        name=name,
        tags=tags,
        notes=notes,
        config=config,
    )
