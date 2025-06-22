"""Module contains the `Namespace` class, this is the base class used to return results from a model prediction.

It functions similarly to `types.SimpleNamespace` but with some nice additional features.

`Namespace` can be used as a stand-in for the Python's `dict` type and can be used to easily pass around keyword parameters by unpacking with **.

`Namespace` attributes can be accessed both with dot-notation `namespace.my_attribute` and using a `str` key `namespace["my_attribute"]`.

Example Usage:
```
def predict(image : torch.Tensor) -> Namespace:
    boxes = ...
    scores = ...
    return Namespace(boxes=boxes, scores=scores)

def postprocess(
    boxes : torch.Tensor,
    scores : Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    ...

result = predict(image)
boxes = postprocess(**result)
```
"""

from typing import Any, Tuple, Iterable, Optional
from collections.abc import Mapping

import torch

from .hints import Device


class Namespace(Mapping):
    """`Namespace` functions like a Python built-in `dict` but with some nice additional features.

    Features include:
    - Access to attributes via dot notation.
    - Unpacking with **.
    - Fancy indexing tensor-like attribute.
    - Device management for torch.Tensor attributes.

    """

    def __init__(self, **kwargs: dict[str, Any]):
        """Constructor.

        Arguments:
            kwargs (dict[str, Any]): key:value mapping for this `Namespace`,
        """
        assert all((isinstance(k, str) and k.isidentifier()) for k in kwargs.keys())
        super().__setattr__("_data", kwargs)  # avoid using self __setattr__

    def keys(self) -> Iterable[str]:  # noqa
        return self._data.keys()

    def values(self) -> Iterable[Any]:  # noqa
        return self._data.values()

    def items(self) -> Iterable[Tuple[str, Any]]:  # noqa
        return self._data.items()

    def _index_by_tensor(self, index: torch.Tensor, key: str) -> Any:
        """Index an element of this namespace using a tensor `index`. If the element does not support fancy indexing then it is returned without change."""
        value = self._data[key]
        # TODO make use of hasattr("__getitem__")
        if isinstance(value, (list, tuple)):
            if index.ndim != 1:
                raise ValueError(
                    f"Expected flat index into Namespace element '{key}' but got shape {list(index.shape)}"
                )
            if index.dtype == torch.bool:
                if index.numel() != len(value):
                    raise ValueError(
                        f"Expected index into Namespace element '{key}' shape to be {[len(value)]} but got {[index.numel()]}"
                    )
                return type(value)([item for item, flag in zip(value, index) if flag])
            elif torch.is_integral(index):
                return type(value)([value[i.item()] for i in index])
            else:
                raise TypeError(
                    f"Expected index into Namespace element '{key}' to be an index type but got type {type(index)}"
                )
        elif isinstance(value, torch.Tensor):
            return value[index]  # index in the usual way
        else:
            return value  # no indexing can happen, just return the value as is.

    def _index_by_slice(self, index: int | slice, key: str) -> Any:
        """Index an element of this namespace using slice or int `index`. If the element does not support indexing (doesnt have the `__getitem__` method) then it is returned without change."""
        value = self._data[key]
        if hasattr(value, "__getitem__"):
            return value[index]
        else:
            return value  # no indexing can happen, just return the value as is.

    def cpu(self) -> "Namespace":
        """Move any torch.Tensor in this Namespace to cpu."""
        return self.to("cpu")

    def cuda(self) -> "Namespace":
        """Move any torch.Tensor in this Namespace to the default CUDA device."""
        return self.to("cuda")

    def to(self, device: Device, **kwargs) -> "Namespace":
        """Send any torch.Tensor to the given `device`.

        Args:
            device (str | torch.device): device to send tensors to
            kwargs (dict, optional): additional optional arguments (unused).
        """

        def _to(value: Any):
            if isinstance(value, torch.Tensor):
                return value.to(device=device)
            else:
                return value

        return Namespace(**{k: _to(v) for k, v in self.items()})

    def __getitem__(self, key: Any) -> Any:  # noqa
        if isinstance(key, str):
            try:
                # return the Namespace attribute with this name
                return self._data[key]
            except KeyError:
                raise AttributeError(key)
        elif isinstance(key, (int, slice)):
            # index each tensor in the namespace (if this is possible)
            return Namespace(
                **{k: self._index_by_slice(key, k) for k, v in self.items()}
            )
        elif isinstance(key, torch.Tensor):
            return Namespace(
                **{k: self._index_by_tensor(key, k) for k, v in self.items()}
            )
        else:
            raise ValueError(
                f"Argument: `key` expected str or index type but got {type(key)}"
            )

    def __setitem__(self, key: str, value: Any):  # noqa
        if not isinstance(key, str) or not key.isidentifier():
            raise ValueError(f"Argument: `key` expected valid identifier but got {key}")
        self._data[key] = value

    def __setattr__(self, key: str, value: Any):  # noqa
        if key in self.__dict__ or key in type(self).__dict__:
            raise AttributeError(
                f"Attribute `{key}` conflicts with an existing internal attribute. `Namespace` does not allow direct assignment to internal attributes."
            )
        else:
            self._data[key] = value

    def to_tuple(self, keys: Optional[Iterable[str]] = None) -> tuple[Any, ...]:
        """Convert this `Namespace` to a tuple.

        The `keys` argument is used to control the order in which the attributes appear in the tuple, it can also be used to select a subset of attributes.

        Without the `keys` argument, the tuple elements will be ordered in the order that the attributes were added to the Namespace - the behaviour is the same as for the built-in `dict` type (the same as `Namespace.values`).

        Args:
            keys (Optional[Iterable[str]], optional): keys to select. Defaults to None.

        Returns:
            tuple[Any,...]: tuple of attribute values.
        """
        if keys is None:
            return tuple(self.values())  # dicts are ordered (at least in cython)!
        else:
            return tuple(self._data.get(k, None) for k in keys)

    def __getattr__(self, key):  # noqa
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(key)

    def __iter__(self):  # noqa
        return iter(self._data)

    def __len__(self):  # noqa
        return len(self._data)

    def __repr__(self):  # noqa
        return f"{self.__class__.__name__}({self._data})"


# DetectionResult : TypeAlias = Namespace
