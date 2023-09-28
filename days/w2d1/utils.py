from einops.einops import rearrange
import torch as t

def itpeek(tensor: t.Tensor):
    contains_nan = t.any(t.isnan(tensor)).item()
    contains_inf = t.any(t.isinf(tensor)).item()
    string = f"SHAPE {tuple(tensor.shape)} MEAN: {'{0:.4g}'.format(t.mean(tensor.float()).cpu().item())} STD: {'{0:.4g}'.format(t.std(tensor.float()).cpu().item())} {'CONTAINS_NAN! ' if contains_nan else ''}{'CONTAINS_INF! ' if contains_inf else ''}VALS [{' '.join(['{0:.4g}'.format(x) for x in t.flatten(tensor)[:10].cpu().tolist()])}{'...' if tensor.numel()>10 else ''}]"
    return string

def tpeek(name: str, tensor: t.Tensor, ret: bool = False):
    string = f"{name} {itpeek(tensor)}"
    if ret:
        return string
    print(string)

def has_not_null(obj, prop):
    return hasattr(obj, prop) and (getattr(obj, prop) is not None)


def copy_weight_bias(mine, theirs, transpose=False):
    if transpose:
        mine.weight = t.nn.Parameter(rearrange(theirs.weight, "a b -> b a"))
    else:
        mine.weight = theirs.weight

    theirs_has_bias = has_not_null(theirs, "bias")
    mine_has_bias = has_not_null(mine, "bias")
    if theirs_has_bias != mine_has_bias:
        print(mine.bias)
        raise AssertionError("yikes")
    if mine_has_bias and theirs_has_bias:
        mine.bias = theirs.bias



# def copy_weight_bias(mine, theirs, transpose=False):
#     if transpose:
#         mine.weight = t.nn.Parameter(rearrange(theirs.weight, "a b -> b a"))
#     else:
#         mine.weight = theirs.weight

#     theirs_has_bias = has_not_null(theirs, "bias")
#     mine_has_bias = has_not_null(mine, "bias")
#     if theirs_has_bias != mine_has_bias:
#         print(mine.bias)
#         raise AssertionError("yikes")
#     if mine_has_bias and theirs_has_bias:
#         mine.bias = theirs.bias