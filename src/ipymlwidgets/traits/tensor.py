from typing import TYPE_CHECKING, Any, Optional, Union, Literal, TypeAlias
from traitlets import TraitType, Undefined
import numpy as np

if TYPE_CHECKING:
    try:
        import torch
    except ImportError:
        pass

    try:
        import tensorflow as tf
    except ImportError:
        pass

# Type aliases - these work for type checking regardless of import success
PTTensor: TypeAlias = "torch.Tensor"
TFTensor: TypeAlias = "tf.Tensor"
NPTensor: TypeAlias = "np.ndarray"
SupportedTensor: TypeAlias = Union[NPTensor, PTTensor, TFTensor]
ConvertTarget: TypeAlias = Literal["np", "pt", "tf"]


class OptionalDependency:
    """Helper class for managing optional dependencies with conversion utilities."""

    def __init__(
        self,
        name: str,
        code: str,
        tensor_type: Optional[str],
        import_name: Optional[str],
    ) -> None:
        """Initialize optional dependency.

        Args:
            name (str): Human-readable name of the dependency.
            code (str): Code used as the conversion Literal.
            tensor_type (str): Name of the tensor type.
            import_name (str): Import name (if different from name). Defaults to None.
        """
        self.name = name
        self.code = code
        self.tensor_type = tensor_type
        self.import_name = import_name or name
        self._module: Optional[Any] = None
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        """Check if the dependency is available.

        Returns:
            bool: True if dependency is available and importable.
        """
        if self._available is None:
            try:
                self._module = __import__(self.import_name)
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    @property
    def module(self) -> Any:
        """Get the imported module.

        Returns:
            Any: The imported module.

        Raises:
            ImportError: If the dependency is not available.
        """
        if self.available:
            return self._module
        raise ImportError(f"{self.name} is not available")

    def __bool__(self) -> bool:
        """Boolean representation of availability."""
        return self.available


class TorchDependency(OptionalDependency):
    """PyTorch optional dependency with conversion utilities."""

    def __init__(self) -> None:
        super().__init__("PyTorch", "pt", "torch.Tensor", "torch")

    def to_numpy(self, tensor: PTTensor) -> np.ndarray:
        """Convert torch tensor to numpy array.

        Args:
            tensor (TorchTensor): PyTorch tensor to convert.

        Returns:
            np.ndarray: Converted numpy array.

        Raises:
            ImportError: If PyTorch is not available.
        """
        if not self.available:
            raise ImportError("PyTorch is required for tensor conversion")

        # Handle different tensor types and devices
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        return tensor.numpy()

    def from_numpy(self, array: np.ndarray) -> PTTensor:
        """Convert numpy array to torch tensor.

        Args:
            array (np.ndarray): Numpy array to convert.

        Returns:
            TorchTensor: Converted PyTorch tensor.

        Raises:
            ImportError: If PyTorch is not available.
        """
        if not self.available:
            raise ImportError("PyTorch is required for tensor conversion")

        return self.module.from_numpy(array)

    def is_tensor(self, obj: Any) -> bool:
        """Check if object is a PyTorch tensor.

        Args:
            obj (Any): Object to check.

        Returns:
            bool: True if object is a PyTorch tensor.
        """
        if not self.available:
            return False
        return isinstance(obj, self.module.Tensor)

    def to_numpy_image(self, tensor: PTTensor) -> np.ndarray:
        """Convert a torch.Tensor to numpy array suitable for image display.

        Converts from CHW [0-1] format to HWC [0-255] uint8 format.

        Args:
            tensor (PTTensor): PyTorch tensor with shape (C, H, W) and values in [0, 1]. Supports 1 channel (grayscale), 3 channels (RGB), or 4 channels (RGBA).

        Returns:
            np.ndarray[H,W,4]: Numpy array with shape (H, W, 4) and dtype uint8 in range [0, 255].
        """
        if tensor.ndim == 3:
            tensor = tensor.permute(1, 2, 0)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        else:
            raise ValueError(
                f"Argument: `tensor` expected 2D or 3D image tensor but got {tensor.ndim}D"
            )
        # tensor is now in HWC format
        if tensor.shape[-1] == 1:
            tensor = tensor.expand(-1, -1, 4)  # RGBA
        elif tensor.shape[-1] == 3:
            tensor = self.module.cat([tensor, self.module.ones_like(tensor[...,:1])], dim=-1)
        elif tensor.shape[-1] == 4:
            pass  # already RGBA
        else:
            raise ValueError(
                f"Argument: `tensor` expected 1, 3, or 4 channels but got {tensor.shape[-1]}"
            )

        if tensor.dtype.is_floating_point:
            tensor = self.module.clamp(tensor, 0.0, 1.0)
            tensor = (tensor * 255).to(self.module.uint8)
        else:
            tensor = tensor.to(self.module.uint8)
        return self.to_numpy(tensor)


class TensorFlowDependency(OptionalDependency):
    """TensorFlow optional dependency with conversion utilities."""

    def __init__(self) -> None:
        super().__init__("TensorFlow", "tf", "tensorflow.Tensor", "tensorflow")

    def to_numpy(self, tensor: TFTensor) -> np.ndarray:
        """Convert TensorFlow tensor to numpy array.

        Args:
            tensor (TFTensor): TensorFlow tensor to convert.

        Returns:
            np.ndarray: Converted numpy array.

        Raises:
            ImportError: If TensorFlow is not available.
        """
        if not self.available:
            raise ImportError("TensorFlow is required for tensor conversion")

        return tensor.numpy()

    def from_numpy(self, array: np.ndarray) -> TFTensor:
        """Convert numpy array to TensorFlow tensor.

        Args:
            array (np.ndarray): Numpy array to convert.

        Returns:
            TFTensor: Converted TensorFlow tensor.

        Raises:
            ImportError: If TensorFlow is not available.
        """
        if not self.available:
            raise ImportError("TensorFlow is required for tensor conversion")

        return self.module.constant(array)

    def is_tensor(self, obj: Any) -> bool:
        """Check if object is a TensorFlow tensor.

        Args:
            obj (Any): Object to check.

        Returns:
            bool: True if object is a TensorFlow tensor.
        """
        if not self.available:
            return False
        return isinstance(obj, self.module.Tensor)

    def to_numpy_image(self, tensor: TFTensor) -> np.ndarray:
        raise NotImplementedError("COMING SOON")  # TODO


# This is not optional, but makes things cleaner in the conversion logic
class NumpyDependency(OptionalDependency):
    """Numpy optional dependency with conversion utilities."""

    def __init__(self) -> None:
        super().__init__("Numpy", "np", "numpy.ndarray", "numpy")

    def to_numpy(self, tensor: np.ndarray) -> np.ndarray:  # noqa
        return tensor

    def from_numpy(self, array: np.ndarray) -> np.ndarray:  # noqa
        return array

    def is_tensor(self, obj: Any) -> bool:  # noqa
        return isinstance(obj, np.ndarray)

    def to_numpy_image(self, array: np.ndarray) -> np.ndarray:
        assert array.ndim == 3
        if array.ndim == 2:
            array = array[..., np.newaxis]
        if array.shape[2] == 1:
            array = array.repeat(3, axis=2)
        elif array.shape[2] == 3:
            array = np.concatenate([array, np.full_like(array[..., :1], 255)], axis=2)
        elif array.shape[2] == 4:
            pass  # already HWC RGBA
        else:
            raise ValueError(
                f"Argument: `array` expected 1, 3, or 4 channels but got {array.shape[2]}"
            )
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0.0, 1.0)
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
        return array


# Create dependency instances
PT_DEP: TorchDependency = TorchDependency()
TF_DEP: TensorFlowDependency = TensorFlowDependency()
NP_DEP: NumpyDependency = NumpyDependency()

OPTIONAL_DEPS = [NP_DEP, PT_DEP, TF_DEP]


class Tensor(TraitType):
    """A trait type for numpy arrays with optional framework support and conversion."""

    def __init__(
        self,
        convert_to: Optional[ConvertTarget] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Tensor trait.

        Args:
            convert_to (Optional[ConvertTarget]): Target format for conversion ('np', 'pt', 'tf').
                Defaults to None.
            **kwargs (Any): Additional arguments passed to TraitType.
        """
        super().__init__(**kwargs)
        # this will always contain numpy!
        self._dependencies = {dep.code: dep for dep in OPTIONAL_DEPS if dep.available}
        if not (convert_to is None or convert_to in self._dependencies):
            codes = list(self._dependencies.keys())
            raise ValueError(
                f"Argument: `convert_to` expected one of {codes} but got {convert_to}"
            )
        self._convert_to = convert_to

        types_str = ", ".join(
            dep.tensor_type for dep in self._dependencies.values() if dep.available
        )
        self.info_text = f"[{types_str}]"

    def get_dependency(self, obj: Any, value: Any) -> Optional[OptionalDependency]:
        """Get the dependency for the current tensor value.

        The dependency can be used to do direct data conversions.

        Args:
            obj (Any): The object that owns this trait.
            value (Any): The value of this trait.

        Returns:
            Optional[OptionalDependency]: The dependency object for the value of this trait or None if the value is None.
        """
        if value is None or value is Undefined:
            return value
        for dep in self._dependencies.values():
            if dep.is_tensor(value):
                return dep
        self.error(obj, value)

    def validate(self, obj: Any, value: Any) -> Optional[SupportedTensor]:
        """Validate and optionally convert the tensor value.

        Args:
            obj (Any): The object that owns this trait.
            value (Any): The value to validate.

        Returns:
            Optional[SupportedTensor]: The validated (and possibly converted) value.
        """
        if value is None or value is Undefined:
            return value  # nothing to do here

        # perform type checking and get the code of the dependency (if it is valid)
        dep = self.get_dependency(obj, value)  # will never be None...
        # perform the conversion if requested
        if self._convert_to is None or self._convert_to == dep.code:
            return value  # no conversion was requested
        # convert to intermediate format (numpy)
        value = dep.to_numpy(value)
        # convert to target format
        return self._dependencies[self._convert_to].from_numpy(value)

    def __eq__(self, other: Any) -> bool:  # noqa
        """Use identity comparison to avoid tensor/array comparison issues."""
        return self is other

    def __ne__(self, other: Any) -> bool:  # noqa
        """Use identity comparison to avoid tensor/array comparison issues."""
        return self is not other
