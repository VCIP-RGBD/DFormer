import torch
import torch.nn as nn
# from models.encoders.DFormer import LayerNorm
# from mmseg.models.decode_heads.mask_attention_cot import LayerNorm as LayerNorm1


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif (
            isinstance(m, norm_layer)
            or isinstance(m, nn.BatchNorm1d)
            or isinstance(m, nn.BatchNorm2d)
            or isinstance(m, nn.BatchNorm3d)
            or isinstance(m, nn.GroupNorm)
            or isinstance(m, nn.LayerNorm)
            # or isinstance(m, LayerNorm)
        ):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)

    # assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
    print(
        "Weight Decay:",
        len(group_decay),
        "Weight No Decay:",
        len(group_no_decay),
        "Total:",
        len(list(module.parameters())),
    )
    # for i in list(module.parameters()):
    #     print(type(i), i.size())
    # assert 1==2
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=0.0, lr=lr))
    return weight_group


def configure_optimizers(model, lr, weight_decay):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (
        torch.nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    )
    blacklist_weight_modules = (
        torch.nn.LayerNorm,
        torch.nn.Embedding,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.SyncBatchNorm,
        LayerNorm,
        nn.LazyBatchNorm1d,
        nn.LazyBatchNorm2d,
        nn.LazyBatchNorm3d,
    )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif "layer_scale" in fpn:
                # special case layer norm scaling parameters in the T5 model
                decay.add(fpn)
            elif "depth_scale" in fpn:
                # special case depth norm scaling parameters in the T5 model
                no_decay.add(fpn)

    # special case the position embedding parameter in the root GPT module as not decayed
    # no_decay.add("pos_emb")

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, (
        "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)
    )

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    return optim_groups
