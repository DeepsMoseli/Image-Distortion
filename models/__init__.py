# -------- models/__init__.py --------
MODEL_REGISTRY = {}

def register_model(name):
    """Decorator to register models by name."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)

# import individual model definitions
from .unet import UNet
from .resnet34_unet import ResNetUNet
from .uformer import UFormer
from .nafnet import NAFNetModel
