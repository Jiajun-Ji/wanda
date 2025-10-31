import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 

from .ablate import AblateGPT 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.decoder.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.decoder.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoder.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                # unstructured pruning
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, 
                                        prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


# ============================================================================
# Progressive Three-Tier Pruning Support for OPT
# ============================================================================

# Tier constants
TIER_DENSE = 0
TIER_2_4 = 1
TIER_TOPK = 2


def save_tier_map(tier_maps, filepath, iteration=None, ratios=None):
    """
    Save tier map to file.

    Args:
        tier_maps: Dictionary mapping layer names to tier maps
        filepath: Path to save file
        iteration: Iteration number (optional)
        ratios: Ratios dictionary (optional)
    """
    torch.save({
        'tier_map': tier_maps,
        'iteration': iteration,
        'ratios': ratios,
        'tier_constants': {
            'DENSE': TIER_DENSE,
            '2:4': TIER_2_4,
            'TOPK': TIER_TOPK
        }
    }, filepath)


def load_tier_map(filepath):
    """
    Load tier map from file.

    Args:
        filepath: Path to tier map file

    Returns:
        tier_map: Tier map tensor
        iteration: Iteration number
        ratios: Ratios dictionary
    """
    data = torch.load(filepath)
    return data['tier_map'], data['iteration'], data['ratios']


def apply_2_4_sparsity_to_block(block):
    """
    Apply 2:4 structured sparsity to a block.
    For every 4 consecutive elements, keep the top-2 with largest absolute values.

    Vectorized implementation for 50-100x speedup.

    Args:
        block: 2D tensor (block of weights)

    Returns:
        mask: Boolean mask (True = prune, False = keep)
        actual_kept: Number of weights kept
    """
    block_flat = block.flatten()
    n = block_flat.numel()

    # Truncate to multiple of 4
    num_groups = n // 4
    n_truncated = num_groups * 4

    if num_groups > 0:
        # Reshape to [num_groups, 4] for vectorized processing
        groups = block_flat[:n_truncated].reshape(num_groups, 4)

        # Get absolute values
        abs_groups = torch.abs(groups)

        # Find top-2 indices for each group (vectorized!)
        _, top2_indices = torch.topk(abs_groups, 2, dim=1, largest=True)

        # Create mask for all groups at once
        mask_groups = torch.ones(num_groups, 4, dtype=torch.bool, device=block.device)

        # Set top-2 to False (keep them)
        row_indices = torch.arange(num_groups, device=block.device).unsqueeze(1)
        mask_groups[row_indices, top2_indices] = False

        # Flatten back to 1D
        mask_flat = mask_groups.flatten()
    else:
        mask_flat = torch.ones(0, dtype=torch.bool, device=block.device)

    # Handle remainder (if n is not divisible by 4)
    remainder = n % 4
    if remainder > 0:
        remainder_group = block_flat[n_truncated:]
        # Keep top-2 if remainder >= 2, otherwise keep all
        if remainder >= 2:
            _, top_indices = torch.topk(torch.abs(remainder_group), min(2, remainder), largest=True)
            remainder_mask = torch.ones(remainder, dtype=torch.bool, device=block.device)
            remainder_mask[top_indices] = False
        else:
            remainder_mask = torch.zeros(remainder, dtype=torch.bool, device=block.device)

        mask_flat = torch.cat([mask_flat, remainder_mask])

    # Reshape back to block shape
    mask = mask_flat.reshape(block.shape)
    actual_kept = (~mask).sum().item()

    return mask, actual_kept


def compute_all_block_scores_unfold(W_metric, block_size=16):
    """
    Compute scores for all blocks using vectorized unfold operation.
    This is the fastest method - fully vectorized, GPU accelerated.

    Args:
        W_metric: Importance metric matrix [M, N]
        block_size: Block size (default 16)

    Returns:
        block_scores: [num_blocks_row, num_blocks_col] tensor
    """
    M, N = W_metric.shape

    # Padding to make dimensions divisible by block_size
    pad_M = (block_size - M % block_size) % block_size
    pad_N = (block_size - N % block_size) % block_size

    if pad_M > 0 or pad_N > 0:
        W_metric_padded = torch.nn.functional.pad(W_metric, (0, pad_N, 0, pad_M))
    else:
        W_metric_padded = W_metric

    M_pad, N_pad = W_metric_padded.shape
    num_blocks_row = M_pad // block_size
    num_blocks_col = N_pad // block_size

    # Reshape into blocks: [num_blocks_row, num_blocks_col, block_size, block_size]
    W_blocks = W_metric_padded.reshape(
        num_blocks_row, block_size,
        num_blocks_col, block_size
    ).permute(0, 2, 1, 3)

    # Compute sum for all blocks at once - fully vectorized!
    block_scores = W_blocks.sum(dim=(2, 3))  # [num_blocks_row, num_blocks_col]

    return block_scores


def compute_2_4_block_scores_batch(W, W_metric, mid_mask, block_size=16):
    """
    Batch compute scores for 2:4 blocks (only non-zero weights).

    Args:
        W: Weight matrix
        W_metric: Importance metric matrix
        mid_mask: Boolean mask [num_blocks_row, num_blocks_col] indicating 2:4 blocks
        block_size: Block size

    Returns:
        scores: [num_2_4_blocks] tensor
    """
    M, N = W.shape
    num_blocks_row = (M + block_size - 1) // block_size
    num_blocks_col = (N + block_size - 1) // block_size

    mid_indices = torch.nonzero(mid_mask)  # [N, 2]
    scores = torch.zeros(len(mid_indices), device=W.device)

    for idx, (i, j) in enumerate(mid_indices):
        i, j = i.item(), j.item()
        row_start = i * block_size
        row_end = min(row_start + block_size, M)
        col_start = j * block_size
        col_end = min(col_start + block_size, N)

        block_W = W[row_start:row_end, col_start:col_end]
        block_metric = W_metric[row_start:row_end, col_start:col_end]

        # Only compute score for non-zero weights
        nonzero_mask = (block_W != 0)
        if nonzero_mask.sum() > 0:
            scores[idx] = block_metric[nonzero_mask].sum()
        else:
            scores[idx] = 0

    return scores


def apply_2_4_sparsity_batch(W, block_indices_flat, num_blocks_col, block_size=16):
    """
    Batch apply 2:4 sparsity to multiple blocks.

    Args:
        W: Weight matrix
        block_indices_flat: [N] tensor of flattened block indices
        num_blocks_col: Number of blocks per column
        block_size: Block size
    """
    M, N = W.shape

    for flat_idx in block_indices_flat:
        flat_idx = flat_idx.item()
        block_i = flat_idx // num_blocks_col
        block_j = flat_idx % num_blocks_col

        row_start = block_i * block_size
        row_end = min(row_start + block_size, M)
        col_start = block_j * block_size
        col_end = min(col_start + block_size, N)

        block = W[row_start:row_end, col_start:col_end]
        mask, _ = apply_2_4_sparsity_to_block(block)
        W[row_start:row_end, col_start:col_end][mask] = 0


def apply_topk_sparsity_batch(W, block_indices_flat, num_blocks_col, block_size=16, k=10):
    """
    Batch apply top-k sparsity to multiple blocks.

    Args:
        W: Weight matrix
        block_indices_flat: [N] tensor of flattened block indices
        num_blocks_col: Number of blocks per column
        block_size: Block size
        k: Number of weights to keep per block
    """
    M, N = W.shape

    for flat_idx in block_indices_flat:
        flat_idx = flat_idx.item()
        block_i = flat_idx // num_blocks_col
        block_j = flat_idx % num_blocks_col

        row_start = block_i * block_size
        row_end = min(row_start + block_size, M)
        col_start = block_j * block_size
        col_end = min(col_start + block_size, N)

        block = W[row_start:row_end, col_start:col_end]
        block_flat = block.flatten()

        if block_flat.numel() > k:
            threshold = torch.topk(torch.abs(block_flat), k, largest=True)[0][-1]
            mask = torch.abs(block) < threshold
            W[row_start:row_end, col_start:col_end][mask] = 0


def progressive_three_tier_iteration(
    W, W_metric,
    current_tier_map,
    target_dense_ratio,
    target_2_4_ratio,
    target_topk_ratio,
    block_size=16,
    topk_per_block=10
):
    """
    Perform one iteration of progressive three-tier pruning.

    Two-stage degradation:
    1. Stage 1: Degrade Dense → 2:4
    2. Stage 2: Re-evaluate all 2:4 blocks (including newly degraded), degrade lowest to TopK

    Args:
        W: Weight matrix
        W_metric: Importance metric matrix
        current_tier_map: [num_blocks_row, num_blocks_col] tensor with tier labels
        target_dense_ratio: Target ratio for dense blocks
        target_2_4_ratio: Target ratio for 2:4 blocks
        target_topk_ratio: Target ratio for topk blocks
        block_size: Block size
        topk_per_block: Number of weights to keep in topk blocks

    Returns:
        updated_tier_map: Updated tier map
        stats: Statistics dictionary
    """
    M, N = W.shape
    num_blocks_row = (M + block_size - 1) // block_size
    num_blocks_col = (N + block_size - 1) // block_size
    total_blocks = num_blocks_row * num_blocks_col

    # Compute all block scores once (vectorized, very fast)
    all_block_scores = compute_all_block_scores_unfold(W_metric, block_size)

    # Flatten for easier indexing
    all_block_scores_flat = all_block_scores.flatten()
    current_tier_map_flat = current_tier_map.flatten()

    stats = {
        'dense_degraded': 0,
        'mid_2_4_degraded': 0,
        'total_blocks': total_blocks
    }

    # ========== Stage 1: Dense Degradation ==========

    # Get current dense blocks
    dense_mask = (current_tier_map_flat == TIER_DENSE)
    num_dense_current = dense_mask.sum().item()
    current_dense_ratio = num_dense_current / total_blocks

    # Calculate how many dense blocks to degrade
    dense_to_degrade_ratio = current_dense_ratio - target_dense_ratio
    num_dense_to_degrade = int(total_blocks * dense_to_degrade_ratio)

    if num_dense_to_degrade > 0:
        # Extract scores for dense blocks
        dense_scores = all_block_scores_flat[dense_mask]
        dense_indices = torch.nonzero(dense_mask).squeeze(-1)

        # Ensure dense_indices is 1D
        if dense_indices.dim() == 0:
            dense_indices = dense_indices.unsqueeze(0)

        # Sort and select lowest scoring blocks
        _, sorted_local_indices = torch.sort(dense_scores)
        blocks_to_degrade_local = sorted_local_indices[:num_dense_to_degrade]
        blocks_to_degrade_global = dense_indices[blocks_to_degrade_local]

        # Ensure blocks_to_degrade_global is 1D
        if blocks_to_degrade_global.dim() == 0:
            blocks_to_degrade_global = blocks_to_degrade_global.unsqueeze(0)

        # Apply 2:4 sparsity to these blocks
        apply_2_4_sparsity_batch(W, blocks_to_degrade_global, num_blocks_col, block_size)

        # Update tier map
        current_tier_map_flat[blocks_to_degrade_global] = TIER_2_4

        stats['dense_degraded'] = num_dense_to_degrade

    # ========== Stage 2: 2:4 Degradation ==========

    # Get all 2:4 blocks (including newly degraded ones)
    mid_mask = (current_tier_map_flat == TIER_2_4)
    num_mid_current = mid_mask.sum().item()
    current_2_4_ratio = num_mid_current / total_blocks

    # Calculate how many 2:4 blocks to degrade
    mid_to_degrade_ratio = current_2_4_ratio - target_2_4_ratio
    num_mid_to_degrade = int(total_blocks * mid_to_degrade_ratio)

    if num_mid_to_degrade > 0:
        # Re-compute scores for 2:4 blocks (only non-zero weights)
        mid_mask_2d = mid_mask.reshape(num_blocks_row, num_blocks_col)
        mid_scores = compute_2_4_block_scores_batch(W, W_metric, mid_mask_2d, block_size)
        mid_indices = torch.nonzero(mid_mask).squeeze(-1)

        # Ensure mid_indices is 1D
        if mid_indices.dim() == 0:
            mid_indices = mid_indices.unsqueeze(0)

        # Sort and select lowest scoring blocks
        _, sorted_local_indices = torch.sort(mid_scores)
        blocks_to_degrade_local = sorted_local_indices[:num_mid_to_degrade]
        blocks_to_degrade_global = mid_indices[blocks_to_degrade_local]

        # Ensure blocks_to_degrade_global is 1D
        if blocks_to_degrade_global.dim() == 0:
            blocks_to_degrade_global = blocks_to_degrade_global.unsqueeze(0)

        # Apply top-k sparsity to these blocks
        apply_topk_sparsity_batch(W, blocks_to_degrade_global, num_blocks_col, block_size, topk_per_block)

        # Update tier map
        current_tier_map_flat[blocks_to_degrade_global] = TIER_TOPK

        stats['mid_2_4_degraded'] = num_mid_to_degrade

    # Reshape tier map back to 2D
    updated_tier_map = current_tier_map_flat.reshape(num_blocks_row, num_blocks_col)

    # Calculate final tier distribution
    final_dense = (updated_tier_map == TIER_DENSE).sum().item()
    final_2_4 = (updated_tier_map == TIER_2_4).sum().item()
    final_topk = (updated_tier_map == TIER_TOPK).sum().item()

    stats['final_dense_blocks'] = final_dense
    stats['final_2_4_blocks'] = final_2_4
    stats['final_topk_blocks'] = final_topk
    stats['final_dense_ratio'] = final_dense / total_blocks
    stats['final_2_4_ratio'] = final_2_4 / total_blocks
    stats['final_topk_ratio'] = final_topk / total_blocks

    return updated_tier_map, stats


def prune_wanda_progressive_three_tier_opt(
    args, model, tokenizer, device=torch.device("cuda:0"),
    iteration_config=None,
    previous_tier_maps=None,
    block_size=16,
    topk_per_block=10,
    log_file=None,
    calib_dataset='wikitext2'
):
    """
    Progressive three-tier pruning with Wanda method for OPT models.

    Performs one iteration of progressive pruning:
    - Stage 1: Degrade Dense → 2:4
    - Stage 2: Re-evaluate all 2:4, degrade lowest → TopK
    - Finetune (done externally)

    Args:
        args: Arguments
        model: OPT model to prune
        tokenizer: Tokenizer
        device: Device
        iteration_config: Dict with 'iteration', 'dense', 'mid_2_4', 'topk' ratios
        previous_tier_maps: Dict mapping layer names to tier maps (from previous iteration)
        block_size: Block size
        topk_per_block: Number of weights to keep in topk blocks
        log_file: Path to log file (optional)
        calib_dataset: Calibration dataset ('wikitext2' or 'c4')

    Returns:
        tier_maps: Dict mapping layer names to tier maps
        stats: Statistics dictionary
    """
    # Helper function for logging
    def log_print(msg):
        """Print to console and optionally to log file."""
        print(msg)
        if log_file is not None:
            with open(log_file, 'a') as f:
                f.write(msg + '\n')

    use_cache = model.config.use_cache
    model.config.use_cache = False

    iteration = iteration_config['iteration']
    target_dense_ratio = iteration_config['dense']
    target_2_4_ratio = iteration_config['mid_2_4']
    target_topk_ratio = iteration_config['topk']

    log_print("="*80)
    log_print(f"Progressive Three-Tier Pruning (OPT) - Iteration {iteration}")
    log_print("="*80)
    log_print(f"Target ratios: Dense={target_dense_ratio*100:.0f}%, 2:4={target_2_4_ratio*100:.0f}%, TopK={target_topk_ratio*100:.0f}%")
    log_print(f"Block size: {block_size}x{block_size}")
    log_print(f"Top-K per block: {topk_per_block}")
    log_print("="*80)

    # Load calibration data
    log_print(f"\nLoading calibration data ({calib_dataset.upper()})...")
    dataloader, _ = get_loaders(calib_dataset, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    log_print("Dataset loading complete\n")

    with torch.no_grad():
        # OPT doesn't use position_ids
        inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device)

    # OPT uses model.model.decoder.layers instead of model.model.layers
    layers = model.model.decoder.layers
    tier_maps = {}

    global_stats = {
        'total_dense_degraded': 0,
        'total_2_4_degraded': 0,
        'total_blocks': 0,
        'final_dense_blocks': 0,
        'final_2_4_blocks': 0,
        'final_topk_blocks': 0
    }

    log_print("="*80)
    log_print("Starting progressive pruning...")
    log_print("="*80)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # Move tensors to correct device if using device_map
        # OPT uses model.decoder.layers.{i} instead of model.layers.{i}
        if f"model.decoder.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.decoder.layers.{i}"]
            inps, outs, attention_mask = (
                inps.to(dev), outs.to(dev), attention_mask.to(dev)
            )

        # Collect wrapped layers for activation computation
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                # OPT doesn't use position_ids
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        # Process each weight matrix in the layer
        layer_tier_maps = {}

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            layer_name = f"layer_{i}.{name}"

            # Get or initialize tier map
            if previous_tier_maps is not None and layer_name in previous_tier_maps:
                # Continue from previous iteration
                current_tier_map = previous_tier_maps[layer_name]
            else:
                # First iteration - initialize from all dense
                M, N = W.shape
                num_blocks_row = (M + block_size - 1) // block_size
                num_blocks_col = (N + block_size - 1) // block_size
                current_tier_map = torch.full((num_blocks_row, num_blocks_col), TIER_DENSE, dtype=torch.long, device=W.device)

            # Perform progressive iteration
            updated_tier_map, stats = progressive_three_tier_iteration(
                W, W_metric,
                current_tier_map,
                target_dense_ratio,
                target_2_4_ratio,
                target_topk_ratio,
                block_size,
                topk_per_block
            )

            layer_tier_maps[name] = updated_tier_map

            # Update global statistics
            global_stats['total_dense_degraded'] += stats['dense_degraded']
            global_stats['total_2_4_degraded'] += stats['mid_2_4_degraded']
            global_stats['total_blocks'] += stats['total_blocks']
            global_stats['final_dense_blocks'] += stats['final_dense_blocks']
            global_stats['final_2_4_blocks'] += stats['final_2_4_blocks']
            global_stats['final_topk_blocks'] += stats['final_topk_blocks']

        # Store tier maps for this layer
        layer_dense = 0
        layer_2_4 = 0
        layer_topk = 0

        for name, tier_map in layer_tier_maps.items():
            tier_maps[f"layer_{i}.{name}"] = tier_map

            # Count blocks for this layer
            layer_dense += (tier_map == TIER_DENSE).sum().item()
            layer_2_4 += (tier_map == TIER_2_4).sum().item()
            layer_topk += (tier_map == TIER_TOPK).sum().item()

        # Calculate layer sparsity
        layer_sparsity = 0.0
        layer_total_params = 0
        layer_pruned_params = 0

        for name in subset:
            W = subset[name].weight.data
            layer_total_params += W.numel()
            layer_pruned_params += (W == 0).sum().item()

        if layer_total_params > 0:
            layer_sparsity = layer_pruned_params / layer_total_params

        # Print layer statistics
        layer_total_blocks = layer_dense + layer_2_4 + layer_topk
        dense_pct = (layer_dense / layer_total_blocks * 100) if layer_total_blocks > 0 else 0
        mid_pct = (layer_2_4 / layer_total_blocks * 100) if layer_total_blocks > 0 else 0
        topk_pct = (layer_topk / layer_total_blocks * 100) if layer_total_blocks > 0 else 0

        log_print(f"Layer {i}: Dense→2:4: {global_stats['total_dense_degraded']}, 2:4→TopK: {global_stats['total_2_4_degraded']} | "
                  f"Dense: {layer_dense} ({dense_pct:.1f}%), 2:4: {layer_2_4} ({mid_pct:.1f}%), TopK: {layer_topk} ({topk_pct:.1f}%) | "
                  f"Sparsity: {layer_sparsity*100:.2f}%")

        # Update inputs for next layer
        for j in range(args.nsamples):
            with torch.no_grad():
                # OPT doesn't use position_ids
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # Print final statistics
    log_print("\n" + "="*80)
    log_print(f"Iteration {iteration} Complete")
    log_print("="*80)
    log_print(f"Total blocks: {global_stats['total_blocks']:,}")
    log_print(f"\nDegradation:")
    log_print(f"  Dense → 2:4: {global_stats['total_dense_degraded']:,} blocks")
    log_print(f"  2:4 → TopK:  {global_stats['total_2_4_degraded']:,} blocks")
    log_print(f"\nFinal distribution:")
    log_print(f"  Dense blocks: {global_stats['final_dense_blocks']:,} ({global_stats['final_dense_blocks']/global_stats['total_blocks']*100:.2f}%)")
    log_print(f"  2:4 blocks:   {global_stats['final_2_4_blocks']:,} ({global_stats['final_2_4_blocks']/global_stats['total_blocks']*100:.2f}%)")
    log_print(f"  TopK blocks:  {global_stats['final_topk_blocks']:,} ({global_stats['final_topk_blocks']/global_stats['total_blocks']*100:.2f}%)")
    log_print("="*80 + "\n")

    return tier_maps, global_stats
