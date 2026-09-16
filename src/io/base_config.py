"""
Base class for experiment configuration dataclasses.

Subclass this with a plain `@dataclass`, give every field a JSON-friendly
default (str, int, float, bool, list, dict, Path, Optional[...] of any of
those, or another dataclass), and you get `.save()` / `.load()` for free:

    @dataclass
    class MyExperimentConfig(BaseConfig):
        learning_rate: float = 1e-3
        num_epochs: int = 100
        data_path: Optional[str] = None

    cfg = MyExperimentConfig(num_epochs=50)
    cfg.save("run/config.json")
    cfg2 = MyExperimentConfig.load("run/config.json")

This is meant to be reused across experiment types (CGH, training runs,
calibration sweeps, ...): each gets its own `BaseConfig` subclass, and all
of them share the same save/load behaviour -- including reconstruction of
nested dataclass fields (e.g. a `ModuleFlags` field on a model config).

Note: field-type introspection relies on real type objects being present
in `__annotations__` at class-definition time, so modules that define
`BaseConfig` subclasses should NOT use `from __future__ import
annotations` (this file doesn't either, for the same reason).
"""
import json
import typing
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar, Union

T = TypeVar("T", bound="BaseConfig")


def _unwrap_optional(field_type: Any) -> Any:
    """Optional[X] is really Union[X, None]; return X. Anything else is returned unchanged."""
    if typing.get_origin(field_type) is typing.Union:
        args = [a for a in typing.get_args(field_type) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return field_type


def _coerce_value(field_type: Any, value: Any) -> Any:
    """
    Turn a plain value decoded from JSON back into whatever `field_type`
    expects. Handles the two cases plain JSON can't represent natively:
    `Path` fields, and nested dataclass fields (including nested
    `BaseConfig` subclasses). Everything else (str/int/float/bool/list/
    dict) already round-trips through JSON unchanged.
    """
    if value is None:
        return None
    field_type = _unwrap_optional(field_type)

    if field_type is Path:
        return Path(value)

    if is_dataclass(field_type) and isinstance(value, dict):
        if isinstance(field_type, type) and issubclass(field_type, BaseConfig):
            return field_type._from_dict(value)
        # Only pass init=True fields to the constructor -- a plain (non-BaseConfig)
        # dataclass may have derived fields (init=False, e.g. computed in
        # __post_init__) that show up in `value` via asdict() but aren't
        # accepted as constructor kwargs (e.g. OpticsGeometry's P/Q/M/N/...).
        kwargs = {
            f.name: _coerce_value(f.type, value[f.name])
            for f in fields(field_type) if f.init and f.name in value
        }
        return field_type(**kwargs)

    return value


def _default_json(obj: Any) -> Any:
    """`json.dump(..., default=...)` hook for values `asdict()` doesn't serialize natively."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Cannot JSON-serialize object of type {type(obj)!r}")


@dataclass
class BaseConfig:
    """
    Parent for experiment configuration dataclasses. Adds JSON save/load;
    subclasses just declare their own fields as a normal `@dataclass`
    (with `BaseConfig` as the base) and get this behaviour for free.
    """

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Union[str, Path]) -> Path:
        """Write this config to `path` as JSON, creating parent directories
        if needed. Returns the path written to."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_default_json)
        return path

    @classmethod
    def load(cls: Type[T], path: Union[str, Path]) -> T:
        """Reconstruct a config of this class from a JSON file written by `save`."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls: Type[T], data: dict) -> T:
        kwargs = {
            f.name: _coerce_value(f.type, data[f.name])
            for f in fields(cls) if f.name in data
        }
        return cls(**kwargs)