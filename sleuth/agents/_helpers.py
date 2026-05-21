from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


def validate_model(model_cls: type[T], data: dict[str, Any]) -> T:
    validator = getattr(model_cls, "model_validate", None)
    if callable(validator):
        return validator(data)
    return model_cls.parse_obj(data)
