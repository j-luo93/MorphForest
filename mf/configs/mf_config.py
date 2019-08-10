from . import registry

register = registry.register


@register
class EngToy:
    lang: str = 'eng.toy'
