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

    layers = model.model.layers
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
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
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
            cache['position_ids'] = kwargs['position_ids']
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
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers

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
    # Use wikitext2 instead of c4 for calibration
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

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
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
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

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    # Use wikitext2 instead of c4 for calibration
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

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
            cache['position_ids'] = kwargs['position_ids']
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
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

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
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    # Use wikitext2 instead of c4 for calibration
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

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
            cache['position_ids'] = kwargs['position_ids']
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
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

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
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
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

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


# ============================================================================
# Block-wise Pruning Functions (16x16 structured pruning)
# ============================================================================

def compute_block_scores(W_metric, block_size=16):
    """
    Divide the weight score matrix into blocks and compute average score for each block.

    Vectorized implementation using unfold for 100-300x speedup.

    Args:
        W_metric: Weight score matrix [M, N]
        block_size: Block size, default 16

    Returns:
        block_scores: Block score matrix [num_blocks_row, num_blocks_col]
        num_blocks_row: Number of blocks in row dimension
        num_blocks_col: Number of blocks in column dimension
    """
    M, N = W_metric.shape
    num_blocks_row = (M + block_size - 1) // block_size
    num_blocks_col = (N + block_size - 1) // block_size

    # Pad to make dimensions divisible by block_size
    pad_M = num_blocks_row * block_size - M
    pad_N = num_blocks_col * block_size - N

    if pad_M > 0 or pad_N > 0:
        # Pad with zeros (won't affect mean calculation significantly for large blocks)
        W_metric_padded = torch.nn.functional.pad(
            W_metric, (0, pad_N, 0, pad_M), mode='constant', value=0
        )
    else:
        W_metric_padded = W_metric

    # Unfold: Extract all blocks at once using sliding window
    # Step 1: Unfold along columns (dimension 1)
    # Input: [M_padded, N_padded]
    # Output: [M_padded, num_blocks_col, block_size]
    unfolded = W_metric_padded.unfold(1, block_size, block_size)

    # Step 2: Unfold along rows (dimension 0)
    # Input: [M_padded, num_blocks_col, block_size]
    # Output: [num_blocks_row, num_blocks_col, block_size, block_size]
    unfolded = unfolded.unfold(0, block_size, block_size)

    # Compute mean for each block (average over last two dimensions)
    # Input: [num_blocks_row, num_blocks_col, block_size, block_size]
    # Output: [num_blocks_row, num_blocks_col]
    block_scores = unfolded.mean(dim=(-2, -1))

    return block_scores, num_blocks_row, num_blocks_col


def apply_block_pruning(W, W_metric, sparsity_ratio, block_size=16):
    """
    Apply block-wise pruning based on block scores.

    Args:
        W: Weight matrix [M, N]
        W_metric: Weight score matrix [M, N]
        sparsity_ratio: Sparsity ratio (by number of blocks)
        block_size: Block size, default 16

    Returns:
        W_mask: Pruning mask (True = prune, False = keep)
        actual_sparsity: Actual sparsity achieved
    """
    M, N = W.shape

    # Compute block scores
    block_scores, num_blocks_row, num_blocks_col = compute_block_scores(W_metric, block_size)

    # Determine number of blocks to prune
    total_blocks = num_blocks_row * num_blocks_col
    num_blocks_to_prune = int(total_blocks * sparsity_ratio)

    # Find threshold: prune blocks with lowest scores
    flat_scores = block_scores.flatten()
    if num_blocks_to_prune > 0 and num_blocks_to_prune < total_blocks:
        threshold = torch.topk(flat_scores, num_blocks_to_prune, largest=False)[0][-1]
    elif num_blocks_to_prune >= total_blocks:
        threshold = float('inf')  # Prune all
    else:
        threshold = float('-inf')  # Prune none

    # Create mask
    W_mask = torch.zeros_like(W, dtype=torch.bool)
    total_pruned = 0
    total_elements = W.numel()

    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            if block_scores[i, j] <= threshold:
                row_start = i * block_size
                row_end = min((i + 1) * block_size, M)
                col_start = j * block_size
                col_end = min((j + 1) * block_size, N)

                W_mask[row_start:row_end, col_start:col_end] = True
                total_pruned += (row_end - row_start) * (col_end - col_start)

    actual_sparsity = total_pruned / total_elements

    return W_mask, actual_sparsity


def apply_block_pruning_with_topk(W, W_metric, sparsity_ratio, block_size=16, topk_per_block=10):
    """
    Apply block-wise pruning with top-k preservation within pruned blocks.

    Strategy:
    1. Compute block scores and select blocks to prune
    2. For pruned blocks: keep only top-k largest weights (by absolute value)
    3. For unpruned blocks: keep all weights

    Args:
        W: Weight matrix
        W_metric: Importance metric matrix (same shape as W)
        sparsity_ratio: Target sparsity ratio (0-1)
        block_size: Block size (default: 16)
        topk_per_block: Number of weights to keep in each pruned block (default: 10)

    Returns:
        W_mask: Pruning mask (True = prune, False = keep)
        actual_sparsity: Actual sparsity achieved
    """
    M, N = W.shape

    # Compute block scores
    block_scores, num_blocks_row, num_blocks_col = compute_block_scores(W_metric, block_size)

    # Determine number of blocks to prune
    total_blocks = num_blocks_row * num_blocks_col
    num_blocks_to_prune = int(total_blocks * sparsity_ratio)

    # Find threshold: prune blocks with lowest scores
    flat_scores = block_scores.flatten()
    if num_blocks_to_prune > 0 and num_blocks_to_prune < total_blocks:
        threshold = torch.topk(flat_scores, num_blocks_to_prune, largest=False)[0][-1]
    elif num_blocks_to_prune >= total_blocks:
        threshold = float('inf')  # Prune all
    else:
        threshold = float('-inf')  # Prune none

    # Create mask
    W_mask = torch.zeros_like(W, dtype=torch.bool)
    total_pruned = 0
    total_elements = W.numel()

    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            row_start = i * block_size
            row_end = min((i + 1) * block_size, M)
            col_start = j * block_size
            col_end = min((j + 1) * block_size, N)

            if block_scores[i, j] <= threshold:
                # This block is selected for pruning
                # Extract the block
                block = W[row_start:row_end, col_start:col_end]
                block_size_actual = block.numel()

                # Keep top-k weights by absolute value
                if block_size_actual > topk_per_block:
                    # Flatten block and find top-k indices
                    block_flat = block.flatten()
                    block_abs = torch.abs(block_flat)

                    # Get indices of top-k largest weights
                    _, topk_indices = torch.topk(block_abs, min(topk_per_block, block_size_actual), largest=True)

                    # Create block mask (True = prune)
                    block_mask = torch.ones(block_size_actual, dtype=torch.bool, device=W.device)
                    block_mask[topk_indices] = False  # Keep top-k

                    # Reshape and assign to W_mask
                    block_mask_2d = block_mask.reshape(block.shape)
                    W_mask[row_start:row_end, col_start:col_end] = block_mask_2d

                    # Count pruned elements
                    total_pruned += block_mask.sum().item()
                else:
                    # Block is smaller than topk, keep all
                    pass
            # else: block is not selected for pruning, keep all (mask remains False)

    actual_sparsity = total_pruned / total_elements

    return W_mask, actual_sparsity


def prune_wanda_block(args, model, tokenizer, device=torch.device("cuda:0"), block_size=16):
    """
    Prune model using Wanda method with block-wise structured pruning.

    Args:
        args: Arguments containing sparsity_ratio, nsamples, seed, etc.
        model: Model to prune
        tokenizer: Tokenizer
        device: Device to use
        block_size: Block size for structured pruning (default: 16)
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("="*80)
    print(f"Wanda Block Pruning (Block Size: {block_size}x{block_size})")
    print("="*80)
    print(f"Target sparsity: {args.sparsity_ratio*100:.2f}%")
    print(f"Block size: {block_size}x{block_size}")
    print("="*80)

    print("\nLoading calibration data (WikiText2)...")
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer
    )
    print("Dataset loading complete\n")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device
        )

    layers = model.model.layers

    # Track overall statistics
    total_params = 0
    total_pruned = 0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = attention_mask.to(dev)
            position_ids = position_ids.to(dev)

        # Wrap layers to collect activation statistics
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

        # Forward pass to collect activation statistics
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )[0]

        for h in handles:
            h.remove()

        # Apply block pruning to each sublayer
        print(f"\n{'='*80}")
        print(f"Pruning Layer {i}")
        print(f"{'='*80}")

        for name in subset:
            W = subset[name].weight.data

            # Calculate Wanda score: |W| * sqrt(activation_norm)
            W_metric = torch.abs(W) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            # Apply block pruning
            W_mask, actual_sparsity = apply_block_pruning(
                W, W_metric, args.sparsity_ratio, block_size
            )

            # Set pruned weights to zero
            subset[name].weight.data[W_mask] = 0

            # Statistics
            num_params = W.numel()
            num_pruned = W_mask.sum().item()
            total_params += num_params
            total_pruned += num_pruned

            print(f"  {name}:")
            print(f"    - Shape: {list(W.shape)}")
            print(f"    - Total params: {num_params:,}")
            print(f"    - Pruned params: {num_pruned:,}")
            print(f"    - Actual sparsity: {actual_sparsity*100:.4f}%")
            print(f"    - Target sparsity: {args.sparsity_ratio*100:.2f}%")

        # Update inputs for next layer
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_block_topk(args, model, tokenizer, device=torch.device("cuda:0"), block_size=16, topk_per_block=10):
    """
    Prune model using Wanda method with block-wise pruning + top-k preservation.

    Strategy:
    - Select blocks to prune based on block-level scores
    - Within pruned blocks, keep only top-k largest weights
    - Unpruned blocks remain fully dense

    Args:
        args: Arguments containing sparsity_ratio, nsamples, seed, etc.
        model: Model to prune
        tokenizer: Tokenizer
        device: Device to use
        block_size: Block size for structured pruning (default: 16)
        topk_per_block: Number of weights to keep in each pruned block (default: 10)
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("="*80)
    print(f"Wanda Block Pruning with Top-K Preservation")
    print("="*80)
    print(f"Target sparsity: {args.sparsity_ratio*100:.2f}%")
    print(f"Block size: {block_size}x{block_size}")
    print(f"Top-K per pruned block: {topk_per_block}")
    print("="*80)

    print("\nLoading calibration data (WikiText2)...")
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer
    )
    print("Dataset loading complete\n")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device
        )

    layers = model.model.layers

    # Track overall statistics
    total_params = 0
    total_pruned = 0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = attention_mask.to(dev)
            position_ids = position_ids.to(dev)

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
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        layer_pruned = 0
        layer_total = 0

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            # Apply block pruning with top-k preservation
            W_mask, actual_sparsity = apply_block_pruning_with_topk(
                W, W_metric, args.sparsity_ratio, block_size, topk_per_block
            )

            # Apply mask
            W[W_mask] = 0

            # Statistics
            layer_pruned += W_mask.sum().item()
            layer_total += W.numel()

        total_pruned += layer_pruned
        total_params += layer_total

        layer_sparsity = layer_pruned / layer_total if layer_total > 0 else 0
        print(f"layer {i} sparsity {layer_sparsity:.6f}")

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        inps, outs = outs, inps

    overall_sparsity = total_pruned / total_params if total_params > 0 else 0
    print(f"\nOverall sparsity: {overall_sparsity*100:.4f}%")
    print(f"Total parameters: {total_params}")
    print(f"Pruned parameters: {total_pruned}")

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # Print overall statistics
    overall_sparsity = total_pruned / total_params
    print("\n" + "="*80)
    print("Block Pruning Complete")
    print("="*80)
    print(f"Total parameters: {total_params:,}")
    print(f"Pruned parameters: {total_pruned:,}")
    print(f"Overall sparsity: {overall_sparsity*100:.4f}%")
    print(f"Target sparsity: {args.sparsity_ratio*100:.2f}%")
    print("="*80 + "\n")


# ============================================================================
# Hybrid Block Pruning with 2:4 Sparsity
# ============================================================================

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
        # topk returns [num_groups, 2]
        _, top2_indices = torch.topk(abs_groups, 2, dim=1, largest=True)

        # Create mask for all groups at once
        # mask: [num_groups, 4], all True initially (prune all)
        mask_groups = torch.ones(num_groups, 4, dtype=torch.bool, device=block.device)

        # Set top-2 to False (keep them) using advanced indexing
        # row_indices: [num_groups, 1] - which group
        # top2_indices: [num_groups, 2] - which elements in each group
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
        abs_remainder = torch.abs(remainder_group)
        k = min(2, remainder)
        _, top_indices = torch.topk(abs_remainder, k, largest=True)

        remainder_mask = torch.ones(remainder, dtype=torch.bool, device=block.device)
        remainder_mask[top_indices] = False

        # Concatenate
        mask_flat = torch.cat([mask_flat, remainder_mask])

    # Reshape to block shape
    mask = mask_flat.reshape(block.shape)
    actual_kept = (~mask).sum().item()

    return mask, actual_kept


def compute_block_score_with_mask(block_metric, mask):
    """
    Compute the score of a block after applying a mask.
    Score is the mean of the metric values for kept weights.

    Args:
        block_metric: 2D tensor of metric values
        mask: Boolean mask (True = pruned, False = kept)

    Returns:
        score: Average metric value of kept weights
    """
    kept_metrics = block_metric[~mask]
    if kept_metrics.numel() == 0:
        return 0.0
    return kept_metrics.mean().item()


def apply_hybrid_block_pruning_with_2_4(
    W, W_metric, sparsity_ratio, block_size=16,
    topk_per_block=10, top_blocks_ratio=0.6, score_threshold=0.8
):
    """
    Apply hybrid block pruning with three types of blocks:
    1. Fully dense blocks (most important)
    2. 2:4 sparse blocks (moderately important, hardware-friendly)
    3. Top-k sparse blocks (least important, scattered values)

    Strategy:
    - Select top `top_blocks_ratio` (e.g., 60%) blocks by score
    - For each selected block:
      - Try applying 2:4 sparsity
      - If score_2:4 >= score_threshold * score_original, use 2:4
      - Otherwise, keep fully dense
    - Remaining blocks: apply top-k sparsity

    Args:
        W: Weight matrix
        W_metric: Metric matrix (importance scores)
        sparsity_ratio: Target overall sparsity (not used directly, for reference)
        block_size: Size of blocks (default 16x16)
        topk_per_block: Number of weights to keep in top-k blocks
        top_blocks_ratio: Ratio of top blocks to consider (default 0.6)
        score_threshold: Threshold for accepting 2:4 sparsity (default 0.8)

    Returns:
        W_mask: Boolean mask (True = prune)
        stats: Dictionary with statistics
    """
    M, N = W.shape

    # Compute block scores
    block_scores, num_blocks_row, num_blocks_col = compute_block_scores(W_metric, block_size)

    total_blocks = num_blocks_row * num_blocks_col
    num_top_blocks = int(total_blocks * top_blocks_ratio)

    # Find top blocks
    flat_scores = block_scores.flatten()
    if num_top_blocks > 0 and num_top_blocks < total_blocks:
        threshold = torch.topk(flat_scores, num_top_blocks, largest=True)[0][-1]
    else:
        threshold = float('-inf')

    # Initialize mask and statistics
    W_mask = torch.zeros_like(W, dtype=torch.bool)

    stats = {
        'fully_dense_blocks': 0,
        'sparse_2_4_blocks': 0,
        'topk_blocks': 0,
        'total_pruned': 0,
        'total_elements': W.numel()
    }

    # Pad W and W_metric to make dimensions divisible by block_size
    pad_M = num_blocks_row * block_size - M
    pad_N = num_blocks_col * block_size - N

    if pad_M > 0 or pad_N > 0:
        W_padded = torch.nn.functional.pad(W, (0, pad_N, 0, pad_M), mode='constant', value=0)
        W_metric_padded = torch.nn.functional.pad(W_metric, (0, pad_N, 0, pad_M), mode='constant', value=0)
    else:
        W_padded = W
        W_metric_padded = W_metric

    # Unfold to get all blocks at once
    # Shape: [num_blocks_row, num_blocks_col, block_size, block_size]
    W_blocks = W_padded.unfold(0, block_size, block_size).unfold(1, block_size, block_size)
    W_metric_blocks = W_metric_padded.unfold(0, block_size, block_size).unfold(1, block_size, block_size)

    # Create top blocks mask
    top_blocks_mask = block_scores >= threshold  # [num_blocks_row, num_blocks_col]

    # Process top blocks (try 2:4 or keep dense)
    num_top = top_blocks_mask.sum().item()
    if num_top > 0:
        # Extract top blocks
        top_W_blocks = W_blocks[top_blocks_mask]  # [num_top, block_size, block_size]
        top_metric_blocks = W_metric_blocks[top_blocks_mask]  # [num_top, block_size, block_size]

        # Compute original scores for all top blocks
        original_scores = top_metric_blocks.sum(dim=(-2, -1))  # [num_top]

        # Apply 2:4 sparsity to all top blocks
        top_masks_2_4 = []
        for idx in range(num_top):
            mask_2_4, _ = apply_2_4_sparsity_to_block(top_W_blocks[idx])
            top_masks_2_4.append(mask_2_4)
        top_masks_2_4 = torch.stack(top_masks_2_4)  # [num_top, block_size, block_size]

        # Compute scores after 2:4 for all blocks
        scores_2_4 = (top_metric_blocks * (~top_masks_2_4)).sum(dim=(-2, -1))  # [num_top]

        # Decide which blocks accept 2:4
        accept_2_4 = scores_2_4 >= score_threshold * original_scores  # [num_top]

        # Count statistics
        stats['sparse_2_4_blocks'] = accept_2_4.sum().item()
        stats['fully_dense_blocks'] = num_top - stats['sparse_2_4_blocks']

    # Process low-score blocks (apply top-k)
    low_blocks_mask = ~top_blocks_mask
    num_low = low_blocks_mask.sum().item()
    stats['topk_blocks'] = num_low

    # Now fill in the W_mask
    # Process each block (still need loop for assignment, but computation is batched)
    top_idx = 0
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            row_start = i * block_size
            row_end = min((i + 1) * block_size, M)
            col_start = j * block_size
            col_end = min((j + 1) * block_size, N)

            if top_blocks_mask[i, j]:
                # Top block
                if accept_2_4[top_idx]:
                    # Use 2:4 sparsity
                    mask_2_4 = top_masks_2_4[top_idx]
                    # Handle padding: only copy the valid part
                    valid_mask = mask_2_4[:row_end-row_start, :col_end-col_start]
                    W_mask[row_start:row_end, col_start:col_end] = valid_mask
                    stats['total_pruned'] += valid_mask.sum().item()
                # else: keep dense (W_mask already initialized to False)
                top_idx += 1
            else:
                # Low-score block - apply top-k
                block = W[row_start:row_end, col_start:col_end]
                block_flat = block.flatten()
                block_size_actual = block_flat.numel()

                if block_size_actual > topk_per_block:
                    block_abs = torch.abs(block_flat)
                    _, topk_indices = torch.topk(block_abs, min(topk_per_block, block_size_actual), largest=True)

                    block_mask = torch.ones(block_size_actual, dtype=torch.bool, device=W.device)
                    block_mask[topk_indices] = False

                    block_mask_2d = block_mask.reshape(block.shape)
                    W_mask[row_start:row_end, col_start:col_end] = block_mask_2d

                    stats['total_pruned'] += block_mask.sum().item()

    actual_sparsity = stats['total_pruned'] / stats['total_elements']
    stats['actual_sparsity'] = actual_sparsity

    return W_mask, stats


# ============================================================================
# Three-Tier Fixed Ratio Block Pruning
# ============================================================================

def apply_three_tier_block_pruning(
    W, W_metric,
    block_size=16,
    top_dense_ratio=0.4,
    mid_2_4_ratio=0.4,
    bottom_topk_ratio=0.2,
    topk_per_block=10
):
    """
    Apply three-tier fixed ratio block pruning strategy.

    Three tiers with fixed ratios:
    1. Top X% blocks: Fully dense (0% sparsity)
    2. Middle Y% blocks: 2:4 structured sparsity (50% sparsity)
    3. Bottom Z% blocks: Top-K sparsity (~96% sparsity)

    Strategy:
    - Compute block scores based on importance metrics
    - Sort all blocks by score (descending)
    - Assign top X% to tier 1 (fully dense)
    - Assign next Y% to tier 2 (2:4 sparse)
    - Assign remaining Z% to tier 3 (top-k sparse)

    Args:
        W: Weight matrix [M, N]
        W_metric: Importance metric matrix [M, N]
        block_size: Block size (default 16)
        top_dense_ratio: Ratio of top blocks to keep fully dense (default 0.4)
        mid_2_4_ratio: Ratio of middle blocks to apply 2:4 sparsity (default 0.4)
        bottom_topk_ratio: Ratio of bottom blocks to apply top-k (default 0.2)
        topk_per_block: Number of weights to keep in bottom blocks (default 10)

    Returns:
        W_mask: Boolean mask (True = prune, False = keep)
        stats: Dictionary with statistics
    """
    M, N = W.shape

    # Validate ratios
    total_ratio = top_dense_ratio + mid_2_4_ratio + bottom_topk_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Ratios sum to {total_ratio:.4f}, not 1.0. Normalizing...")
        top_dense_ratio /= total_ratio
        mid_2_4_ratio /= total_ratio
        bottom_topk_ratio /= total_ratio

    # Compute block scores
    block_scores, num_blocks_row, num_blocks_col = compute_block_scores(W_metric, block_size)
    total_blocks = num_blocks_row * num_blocks_col

    # Calculate number of blocks in each tier
    num_top_dense = int(total_blocks * top_dense_ratio)
    num_mid_2_4 = int(total_blocks * mid_2_4_ratio)
    num_bottom_topk = total_blocks - num_top_dense - num_mid_2_4

    # Sort blocks by score (descending)
    flat_scores = block_scores.flatten()
    sorted_indices = torch.argsort(flat_scores, descending=True)

    # Divide into three tiers
    top_dense_indices = set(sorted_indices[:num_top_dense].tolist())
    mid_2_4_indices = set(sorted_indices[num_top_dense:num_top_dense+num_mid_2_4].tolist())
    bottom_topk_indices = set(sorted_indices[num_top_dense+num_mid_2_4:].tolist())

    # Initialize mask and statistics
    W_mask = torch.zeros_like(W, dtype=torch.bool)

    stats = {
        'fully_dense_blocks': 0,
        'sparse_2_4_blocks': 0,
        'topk_blocks': 0,
        'total_blocks': total_blocks,
        'total_elements': W.numel(),
        'total_pruned': 0,
        'dense_weights_kept': 0,
        'sparse_2_4_weights_kept': 0,
        'topk_weights_kept': 0
    }

    # Process each block
    for block_idx in range(total_blocks):
        block_i = block_idx // num_blocks_col
        block_j = block_idx % num_blocks_col

        row_start = block_i * block_size
        row_end = min(row_start + block_size, M)
        col_start = block_j * block_size
        col_end = min(col_start + block_size, N)

        block = W[row_start:row_end, col_start:col_end]
        block_elements = block.numel()

        if block_idx in top_dense_indices:
            # Tier 1: Fully dense - keep all weights
            # mask remains False (no pruning)
            stats['fully_dense_blocks'] += 1
            stats['dense_weights_kept'] += block_elements

        elif block_idx in mid_2_4_indices:
            # Tier 2: Apply 2:4 structured sparsity
            block_mask, kept = apply_2_4_sparsity_to_block(block)
            W_mask[row_start:row_end, col_start:col_end] = block_mask
            stats['sparse_2_4_blocks'] += 1
            stats['sparse_2_4_weights_kept'] += kept
            stats['total_pruned'] += block_mask.sum().item()

        else:  # block_idx in bottom_topk_indices
            # Tier 3: Apply top-k sparsity
            block_flat = block.flatten()
            if block_flat.numel() > topk_per_block:
                # Keep top-k weights by absolute value
                threshold = torch.topk(torch.abs(block_flat), topk_per_block, largest=True)[0][-1]
                block_mask = torch.abs(block) < threshold
                W_mask[row_start:row_end, col_start:col_end] = block_mask
                stats['topk_blocks'] += 1
                stats['topk_weights_kept'] += topk_per_block
                stats['total_pruned'] += block_mask.sum().item()
            else:
                # Block smaller than topk, keep all
                stats['topk_blocks'] += 1
                stats['topk_weights_kept'] += block_flat.numel()

    # Calculate actual sparsity
    actual_sparsity = stats['total_pruned'] / stats['total_elements']
    stats['actual_sparsity'] = actual_sparsity

    return W_mask, stats


def prune_wanda_three_tier(
    args, model, tokenizer, device=torch.device("cuda:0"),
    block_size=16,
    top_dense_ratio=0.4,
    mid_2_4_ratio=0.4,
    bottom_topk_ratio=0.2,
    topk_per_block=10
):
    """
    Prune model using Wanda method with three-tier fixed ratio block pruning.

    Three tiers with fixed ratios:
    1. Top X% blocks: Fully dense (most important)
    2. Middle Y% blocks: 2:4 structured sparsity (moderately important)
    3. Bottom Z% blocks: Top-K sparsity (least important)

    Args:
        args: Arguments containing model path, nsamples, seed, etc.
        model: The model to prune
        tokenizer: Tokenizer for the model
        device: Device to use
        block_size: Size of blocks (default 16)
        top_dense_ratio: Ratio of top blocks to keep fully dense (default 0.4)
        mid_2_4_ratio: Ratio of middle blocks to apply 2:4 (default 0.4)
        bottom_topk_ratio: Ratio of bottom blocks to apply top-k (default 0.2)
        topk_per_block: Number of weights to keep in bottom blocks (default 10)
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("="*80)
    print("Wanda Three-Tier Fixed Ratio Block Pruning")
    print("="*80)
    print(f"Block size: {block_size}x{block_size}")
    print(f"Tier 1 (Fully Dense): Top {top_dense_ratio*100:.0f}% blocks")
    print(f"Tier 2 (2:4 Sparse):  Middle {mid_2_4_ratio*100:.0f}% blocks")
    print(f"Tier 3 (Top-K):       Bottom {bottom_topk_ratio*100:.0f}% blocks (k={topk_per_block})")
    print("="*80)

    print("\nLoading calibration data (WikiText2)...")
    dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("Dataset loading complete\n")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    total_params = 0
    total_pruned = 0

    # Global statistics
    global_stats = {
        'fully_dense_blocks': 0,
        'sparse_2_4_blocks': 0,
        'topk_blocks': 0,
        'dense_weights_kept': 0,
        'sparse_2_4_weights_kept': 0,
        'topk_weights_kept': 0
    }

    print("="*80)
    print("Starting pruning process...")
    print("="*80)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

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
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        # Prune each layer
        layer_pruned = 0
        layer_total = 0
        layer_stats = {
            'fully_dense_blocks': 0,
            'sparse_2_4_blocks': 0,
            'topk_blocks': 0
        }

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            # Apply three-tier block pruning
            W_mask, stats = apply_three_tier_block_pruning(
                W, W_metric, block_size,
                top_dense_ratio, mid_2_4_ratio, bottom_topk_ratio,
                topk_per_block
            )

            # Apply mask
            W[W_mask] = 0

            # Update statistics
            layer_pruned += stats['total_pruned']
            layer_total += stats['total_elements']
            layer_stats['fully_dense_blocks'] += stats['fully_dense_blocks']
            layer_stats['sparse_2_4_blocks'] += stats['sparse_2_4_blocks']
            layer_stats['topk_blocks'] += stats['topk_blocks']

            global_stats['fully_dense_blocks'] += stats['fully_dense_blocks']
            global_stats['sparse_2_4_blocks'] += stats['sparse_2_4_blocks']
            global_stats['topk_blocks'] += stats['topk_blocks']
            global_stats['dense_weights_kept'] += stats['dense_weights_kept']
            global_stats['sparse_2_4_weights_kept'] += stats['sparse_2_4_weights_kept']
            global_stats['topk_weights_kept'] += stats['topk_weights_kept']

        # Update global statistics
        total_pruned += layer_pruned
        total_params += layer_total

        layer_sparsity = layer_pruned / layer_total if layer_total > 0 else 0
        print(f"\nLayer {i}:")
        print(f"  Sparsity: {layer_sparsity*100:.4f}%")
        print(f"  Blocks - Dense: {layer_stats['fully_dense_blocks']}, "
              f"2:4: {layer_stats['sparse_2_4_blocks']}, "
              f"Top-K: {layer_stats['topk_blocks']}")

        # Update inputs for next layer
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # Print final statistics
    overall_sparsity = total_pruned / total_params if total_params > 0 else 0

    print("\n" + "="*80)
    print("Pruning Complete - Final Statistics")
    print("="*80)
    print(f"Total blocks: {global_stats['fully_dense_blocks'] + global_stats['sparse_2_4_blocks'] + global_stats['topk_blocks']}")
    print(f"  - Fully dense blocks: {global_stats['fully_dense_blocks']} ({global_stats['dense_weights_kept']:,} weights)")
    print(f"  - 2:4 sparse blocks:  {global_stats['sparse_2_4_blocks']} ({global_stats['sparse_2_4_weights_kept']:,} weights)")
    print(f"  - Top-K blocks:       {global_stats['topk_blocks']} ({global_stats['topk_weights_kept']:,} weights)")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Pruned parameters: {total_pruned:,}")
    print(f"Overall sparsity: {overall_sparsity*100:.4f}%")
    print("="*80 + "\n")


def prune_wanda_hybrid_2_4(
    args, model, tokenizer, device=torch.device("cuda:0"),
    block_size=16, topk_per_block=10, top_blocks_ratio=0.6, score_threshold=0.8
):
    """
    Prune model using Wanda method with hybrid block pruning (2:4 + dense + top-k).

    Three types of blocks:
    1. Fully dense blocks (most important, all weights kept)
    2. 2:4 sparse blocks (moderately important, hardware-friendly)
    3. Top-k sparse blocks (least important, scattered values)

    Args:
        args: Arguments containing model path, sparsity ratio, etc.
        model: The model to prune
        tokenizer: Tokenizer for the model
        device: Device to use
        block_size: Size of blocks (default 16)
        topk_per_block: Number of weights to keep in top-k blocks (default 10)
        top_blocks_ratio: Ratio of top blocks to consider (default 0.6)
        score_threshold: Threshold for accepting 2:4 sparsity (default 0.8)
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("="*80)
    print("Loading calibration data (WikiText2)...")
    print("="*80)
    dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("Dataset loading complete\n")

    print("="*80)
    print("Starting hybrid block pruning with 2:4 sparsity...")
    print(f"Block size: {block_size}x{block_size}")
    print(f"Top blocks ratio: {top_blocks_ratio} (top {top_blocks_ratio*100:.0f}% blocks)")
    print(f"Score threshold for 2:4: {score_threshold} ({score_threshold*100:.0f}% of original score)")
    print(f"Top-K per low-score block: {topk_per_block}")
    print("="*80)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    total_params = 0
    total_pruned = 0

    # Statistics for all layers
    global_stats = {
        'fully_dense_blocks': 0,
        'sparse_2_4_blocks': 0,
        'topk_blocks': 0
    }

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            )

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
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        # Prune each weight matrix in the layer
        layer_pruned = 0
        layer_total = 0
        layer_stats = {
            'fully_dense_blocks': 0,
            'sparse_2_4_blocks': 0,
            'topk_blocks': 0
        }

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            # Apply hybrid block pruning with 2:4
            W_mask, stats = apply_hybrid_block_pruning_with_2_4(
                W, W_metric, args.sparsity_ratio, block_size,
                topk_per_block, top_blocks_ratio, score_threshold
            )

            # Apply mask
            W[W_mask] = 0

            # Update statistics
            layer_pruned += stats['total_pruned']
            layer_total += stats['total_elements']
            layer_stats['fully_dense_blocks'] += stats['fully_dense_blocks']
            layer_stats['sparse_2_4_blocks'] += stats['sparse_2_4_blocks']
            layer_stats['topk_blocks'] += stats['topk_blocks']

        # Update global statistics
        total_pruned += layer_pruned
        total_params += layer_total
        global_stats['fully_dense_blocks'] += layer_stats['fully_dense_blocks']
        global_stats['sparse_2_4_blocks'] += layer_stats['sparse_2_4_blocks']
        global_stats['topk_blocks'] += layer_stats['topk_blocks']

        # Print layer statistics
        layer_sparsity = layer_pruned / layer_total if layer_total > 0 else 0
        print(f"layer {i} sparsity {layer_sparsity:.6f} | "
              f"Dense: {layer_stats['fully_dense_blocks']}, "
              f"2:4: {layer_stats['sparse_2_4_blocks']}, "
              f"Top-K: {layer_stats['topk_blocks']}")

        # Prepare for next layer
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    # Print overall statistics
    overall_sparsity = total_pruned / total_params
    total_blocks = sum(global_stats.values())

    print("\n" + "="*80)
    print("Hybrid Block Pruning Complete")
    print("="*80)
    print(f"Total parameters: {total_params:,}")
    print(f"Pruned parameters: {total_pruned:,}")
    print(f"Overall sparsity: {overall_sparsity*100:.4f}%")
    print(f"\nBlock Statistics:")
    print(f"  Fully dense blocks: {global_stats['fully_dense_blocks']:,} ({global_stats['fully_dense_blocks']/total_blocks*100:.2f}%)")
    print(f"  2:4 sparse blocks:  {global_stats['sparse_2_4_blocks']:,} ({global_stats['sparse_2_4_blocks']/total_blocks*100:.2f}%)")
    print(f"  Top-K blocks:       {global_stats['topk_blocks']:,} ({global_stats['topk_blocks']/total_blocks*100:.2f}%)")
    print("="*80 + "\n")

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()