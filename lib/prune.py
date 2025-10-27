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

    block_scores = torch.zeros(num_blocks_row, num_blocks_col,
                               device=W_metric.device, dtype=W_metric.dtype)

    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            row_start = i * block_size
            row_end = min((i + 1) * block_size, M)
            col_start = j * block_size
            col_end = min((j + 1) * block_size, N)

            block = W_metric[row_start:row_end, col_start:col_end]
            block_scores[i, j] = block.mean()  # Use mean to handle variable block sizes

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