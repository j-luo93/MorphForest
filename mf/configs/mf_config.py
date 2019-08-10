from . import registry

register = registry.register


@register
class EngToy:
    lang: str = 'eng.toy'


@register
class Eng:
    lang: str = 'en'
    data_path: str = 'data/en/'


@register
class EngWithWordVectors(Eng):
    use_word_vectors: bool = True
    strict_wv: bool = False
