from torch import nn


def init_weights(module: nn.Module):
    # Linear layers and Conv layers
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    # Embedding layers (Be careful not to overwrite a pre-trained codebook if you loaded one)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # LayerNorm (Resets them to identity, undoing any damage from previous inits)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
