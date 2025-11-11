"""
Framework factory utilities.
Automatically builds registered framework implementations
based on configuration.

Each framework module (e.g., M1.py, QwenFast.py) should register itself:
    from starVLA.model.framework.framework_registry import FRAMEWORK_REGISTRY

    @FRAMEWORK_REGISTRY.register("InternVLA-M1")
    def build_model_framework(config):
        return InternVLA_M1(config=config)
"""

import pkgutil
import importlib
from starVLA.model.tools import FRAMEWORK_REGISTRY


try:
    pkg_path = __path__
except NameError:
    pkg_path = None

# Auto-import all framework submodules to trigger registration
# Import each submodule individually and continue on error so a single
# broken framework file does not prevent other frameworks from registering.
if pkg_path is not None:
    for _, module_name, _ in pkgutil.iter_modules(pkg_path):
        mod_name = f"{__name__}.{module_name}"
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            # Print a helpful warning and continue importing other modules.
            # Use traceback for full context during debugging.
            import traceback
            print(f"Warning: Failed to import framework submodule {mod_name}: {e}")
            traceback.print_exc()
            continue
        
def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: Identifier string (e.g. "InternVLA-M1")
    Returns:
        nn.Module: Instantiated framework model.
    """

    if not hasattr(cfg.framework, "name"): 
        cfg.framework.name = cfg.framework.framework_py  # Backward compatibility for legacy config yaml
        
    if cfg.framework.name == "QwenOFT":
        from starVLA.model.framework.QwenOFT import Qwenvl_OFT
        return Qwenvl_OFT(cfg)
    elif cfg.framework.name == "QwenFast":
        from starVLA.model.framework.QwenFast import Qwenvl_Fast
        return Qwenvl_Fast(cfg)

    
    # auto detect from registry
    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(f"Framework {cfg.framework.name} is not implemented.")
    
    MODLE_CLASS = FRAMEWORK_REGISTRY[framework_id]
    return MODLE_CLASS(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
