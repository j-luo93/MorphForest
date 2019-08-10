from arglib import create_registry

try:
    registry = create_registry('mf')
    from . import mf_config
except:
    raise ImportError('Failed to get a config registry.')
