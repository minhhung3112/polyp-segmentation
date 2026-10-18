import sys
import os

proj_root = r'D:\Paper\Polyp_Segmentation_Research\PolypSeg_Project'
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import torch
from models import get_model, list_models
from loss import get_loss

all_models = sorted(list_models())
print(f"Total registered models: {len(all_models)}")
print("Available models:", all_models)

dummy_input = torch.randn(2, 3, 352, 352)
gt = torch.zeros(2, 1, 352, 352)
gt[:, :, 80:240, 80:240] = 1.0

loss_fn = get_loss('structure_loss')

results = []
for model_name in all_models:
    try:
        model = get_model(model_name)
        model.train()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Forward pass
        out = model(dummy_input)
        if isinstance(out, (tuple, list)):
            shapes = [tuple(o.shape) for o in out]
            final_pred = out[-1]
            loss = sum(loss_fn(o, gt) for o in out)
        else:
            shapes = [tuple(out.shape)]
            final_pred = out
            loss = loss_fn(out, gt)
            
        # Backward pass test
        loss.backward()
        
        # Verify output shape matches 352x352
        assert final_pred.shape == (2, 1, 352, 352), f"Shape mismatch: expected (2, 1, 352, 352), got {final_pred.shape}"
        
        results.append({
            'model': model_name,
            'status': 'PASS',
            'params_M': total_params / 1e6,
            'outputs': shapes,
            'loss': loss.item()
        })
    except Exception as e:
        results.append({
            'model': model_name,
            'status': f'FAIL: {e}',
            'params_M': 0,
            'outputs': [],
            'loss': 0.0
        })

print('\n' + '='*85)
print(f"{'MODEL':<18} | {'STATUS':<8} | {'PARAMS (M)':<12} | {'OUTPUT SHAPES':<25} | {'LOSS':<8}")
print('='*85)
for r in results:
    shapes_str = str(r['outputs'][0]) if len(r['outputs']) == 1 else f"{len(r['outputs'])} scales"
    print(f"{r['model']:<18} | {r['status']:<8} | {r['params_M']:>8.2f}M    | {shapes_str:<25} | {r['loss']:>8.4f}")
print('='*85)
