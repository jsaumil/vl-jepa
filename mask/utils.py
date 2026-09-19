import torch

def apply_masks(x, masks, concat=True):
    """
    x = [b,n,d]
    masks = [b,k] contains k patches in [n]
    """
    masks = [masks] if not isinstance(masks, list) else masks
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1,1,x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x
    
    return torch.cat(all_x, dim=0)

def get_complement_masks(masks, n):
    """
    masks = [b, k] contains k patch indices in [0, n)
    Returns complement = [b, n-k] containing the remaining indices
    """
    b, k = masks.shape
    device = masks.device

    # Start with all indices marked as "keep" (True)
    keep = torch.ones(b, n, dtype=torch.bool, device=device)
    # Mark the masked indices as False (these are the ones we want to remove)
    keep.scatter_(1, masks, False)

    # The complement indices are those still marked True
    complement = torch.arange(n, device=device).unsqueeze(0).expand(b, -1)
    complement_indices = complement[keep].view(b, n - k)

    return complement_indices