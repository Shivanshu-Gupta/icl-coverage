import attr
import cattr
from collections.abc import Mapping
from collections import UserList
from typing import IO, Iterable, Text, Union, get_origin, get_args
from pathlib import Path
from itertools import product
from copy import deepcopy


converter = cattr.Converter()
converter.register_structure_hook(Path, lambda d, t: Path(d))
converter.register_unstructure_hook( Path, lambda d: str(d) )       # type: ignore

def nest_dict(flat, sep='.'):
    def _nest_dict_rec(k, v, out, sep='.'):
        k, *rest = k.split(sep, 1)
        if rest:
            _nest_dict_rec(rest[0], v, out.setdefault(k, {}))
        else:
            out[k] = v

    result = {}
    for k, v in flat.items():
        _nest_dict_rec(k, v, result, sep=sep)
    return result

def default_value(default):
    return attr.ib(default=attr.Factory(lambda: default))


@attr.s(auto_attribs=True)
class DictDataClass(Mapping):
    """
        Allow dict-like access to attributes using ``[]`` operator in addition to dot-access.
        Easy serisation to/deserilation from nested dict, flattened dict, json and yaml.
    """

    def __iter__(self):
        return iter(vars(self))

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __len__(self):
        return len(vars(self))

    def to_dict(self):
        """ Serialize to a nested dict """

        # this doesn't always produce a json serializable dict (eg. with Path objects)
        # return attr.asdict(self)

        global converter
        return converter.unstructure(self)

    def to_flattened_dict(self, sep='.', _parent_key=''):
        """ Seralize to a flattened dict using the given separator `sep` """
        # _d = pd.json_normalize(attr.asdict(self), sep=sep).iloc[0].to_dict()
        flat_d = {}
        for k, v in self.items():
            if _parent_key: k = _parent_key + sep + k
            if isinstance(v, DictDataClass):
                flat_d.update(v.to_flattened_dict(sep=sep, _parent_key=k))
            else:
                flat_d[k] = v
        return flat_d

    @classmethod
    def from_flattened_dict(cls, d: dict, sep='.'):
        """ Deserialize from a flattened dict `d` using the given separator `sep` """
        return cls.from_dict(nest_dict(d, sep=sep))

    @classmethod
    def from_dict(cls, d: dict):
        """ Deserialize from a nested dict `d` """
        global converter
        converter = converter.copy()
        disambiguators = cls._get_all_disambiguators()
        for union_type, func in disambiguators.items():
            converter.register_structure_hook(union_type, lambda o, t, hook=func: converter.structure(o, hook(o, t)))
        return converter.structure(d, cls)

    @classmethod
    def get_disambiguators(cls):
        return {}

    @classmethod
    def _get_all_disambiguators(cls):
        disambiguators = cls.get_disambiguators()
        for _, t in cls.__annotations__.items():
            try:
                if issubclass(t, DictDataClass):
                    disambiguators.update(t._get_all_disambiguators())
            except TypeError as e:
                if str(e) == 'issubclass() arg 1 must be a class': # t is a generic type not a class
                    if get_origin(t) == Union:
                        for _t in get_args(t):
                            if issubclass(_t, DictDataClass):
                                disambiguators.update(_t._get_all_disambiguators())
        return disambiguators

    def to_json(self, fp: IO[str]):
        """ Serialize to a json file  """
        import json; json.dump(self.to_dict(), fp)

    @classmethod
    def from_json(cls, fp):
        """ Deserialize from a json file  """
        import json
        return cls.from_dict(json.load(fp))

    def to_yaml(self, stream: IO[str]):
        """ Serialize to a yaml file  """
        import yaml; yaml.dump(self.to_dict(), stream)

    @classmethod
    def from_yaml(cls, stream: Union[bytes, IO[bytes], str, IO[Text]]):
        """ Deserialize from a yaml file  """
        import yaml
        return cls.from_dict(yaml.load(stream, Loader=yaml.FullLoader))

class Settings(UserList):
    def __init__(self, data):
        super().__init__(data)

class Parameters(DictDataClass):
    def get_settings(self, key_order=None):
        keys = []
        value_lists = []
        key_order = key_order or list(self.keys())
        assert set(key_order) == set(self.keys()), 'key_order must contain all the keys'
        for k in key_order:
            v = getattr(self, k)
            if isinstance(v, Parameters):
                value_lists.append(v.get_settings())
                keys.append(k)
            elif isinstance(v, list):
                if len(v) == 0:
                    raise ValueError(f"Empty settings list for {k}")
                if isinstance(v[0], Parameters):    # Each value in settings list itself extends GridMixin so needs to be explored in the search space
                    value_lists.append([s for _v in v for s in _v.get_settings()])
                else:
                    value_lists.append(v)
                keys.append(k)
        _setting = deepcopy(self)
        settings = []
        for values in product(*value_lists):
            for k, v in zip(keys, values):
                setattr(_setting, k, v)
            settings.append(deepcopy(_setting))
        return settings

class InstantiationMixin:
    """
        Mixin that enables direct instantiation of object from a `DictDataClass`
        containing parameters of a particular class.
        Note: If a `DictDataClass` is given this mixin then all the "sub-parameter"
        that are also of type `DictDataClass` need to have this mixin to be
        recursively instantiated.
    """
    def instantiate(self, **kwargs):
        """
            Recursively instantiates an instance of the class specified in the
            type attribute using parameters from this `DictDataClass` instance
            overridden using `kwargs`.

            `kwargs` should be a nested dict containing any additional parameters
            required to instantiate the class and its constructor arguments.
            At the very list it should values of all positional arguments not
            specified in the `DictDataClass`.

            Example:
            ```
                class SimpleTagger:
                    def __init__(self, embedding_param=50, encoder=None):
                    self.embedding_param = embedding_param
                    self.encoder = encoder

                @attr.s(auto_attribs=True)
                class EncoderParams(Parameters, InstantiationMixin):
                    type: str = 'torch.nn.LSTM'
                    hidden_size: int = 100
                    num_layers: int = 1

                @attr.s(auto_attribs=True)
                class ModelParams(Parameters, InstantiationMixin):
                    type: type = SimpleTagger
                    embedding_param: Union[int, str] = 50
                    encoder: Optional[EncoderParams] = None

                mp = ModelParams(encoder=EncoderParams())
            ```
            For the above since `input_size` is a required positional argument
            for `torch.nn.LSTM`, `mp = m.instantiate(encoder={'input_size': 10})`
            will work, but not `mp = m.instantiate()`.
        """
        if not hasattr(self, 'type'):
            raise ValueError('Missing type attribute.')
        parameters = deepcopy(vars(self))
        _type = parameters.pop('type')
        if isinstance(_type, str) and _type == '':
            return None
        if _type:
            instantiated_attrs = {}
            for attr_name, attr_params in parameters.items():
                if isinstance(attr_params, InstantiationMixin):
                    if attr_name in kwargs:
                        instantiated_attrs[attr_name] = attr_params.instantiate(**kwargs[attr_name])
                        kwargs.pop(attr_name)
                    else:
                        instantiated_attrs[attr_name] = attr_params.instantiate()
            parameters.update(instantiated_attrs)
            parameters.update(kwargs)
            if isinstance(_type, str):
                _type = _type.split('.')
                module_name, class_name = '.'.join(_type[:-1]), _type[-1]
                import importlib
                module = importlib.import_module(module_name)
                _class = getattr(module, class_name)
            elif isinstance(_type, type):
                _class = _type
            instance = _class(**parameters)
            return instance
