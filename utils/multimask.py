import torch.nn as nn

class MultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks=None):
        if masks is None:
            return self.backbone(x)
        
        if (masks is not None) and not isinstance(masks, list):
            masks = [masks]
        outs = []
        for m in masks:
            outs += [self.backbone(x, masks=m)]
        return outs
    
class PredictorMultiMaskWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt):
        if type(ctxt) is not list:
            ctxt = [ctxt]
        if type(tgt) is not list:
            tgt = [tgt]
        if type(masks_ctxt) is not list:
            masks_ctxt = [masks_ctxt]
        if type(masks_tgt) is not list:
            masks_tgt = [masks_tgt]

        outs = []

        for i, (zi, hi, mc, mt) in enumerate(zip(ctxt, tgt, masks_ctxt, masks_tgt)):
            outs += [self.backbone(zi, hi, mc, mt, mask_index=i)]
        return outs