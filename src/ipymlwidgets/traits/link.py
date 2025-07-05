"""Re-implementation of traitlets.link and traitlets.directional_link that avoids issues with tensor traits."""

from traitlets import HasTraits, TraitError
import typing as t
import contextlib


def _validate_link(*tuples: t.Any) -> None:
    """Validate arguments for traitlet link functions"""
    for tup in tuples:
        if not len(tup) == 2:
            raise TypeError(
                "Each linked traitlet must be specified as (HasTraits, 'trait_name'), not %r"
                % t
            )
        obj, trait_name = tup
        if not isinstance(obj, HasTraits):
            raise TypeError("Each object must be HasTraits, not %r" % type(obj))
        if trait_name not in obj.traits():
            raise TypeError(f"{obj!r} has no trait {trait_name!r}")


class link:

    updating = False

    def __init__(self, source: t.Any, target: t.Any, transform: t.Any = None) -> None:
        _validate_link(source, target)
        self.source, self.target = source, target
        self._transform, self._transform_inv = (
            transform if transform else (lambda x: x,) * 2
        )

        self.link()

    def link(self) -> None:
        try:
            setattr(
                self.target[0],
                self.target[1],
                self._transform(getattr(self.source[0], self.source[1])),
            )

        finally:
            self.source[0].observe(self._update_target, names=self.source[1])
            self.target[0].observe(self._update_source, names=self.target[1])

    @contextlib.contextmanager
    def _busy_updating(self) -> t.Any:
        self.updating = True
        try:
            yield
        finally:
            self.updating = False

    def _update_target(self, change: t.Any) -> None:
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.target[0], self.target[1], self._transform(change.new))
            # this will never happen for us as we are not in an async context TODO true?
            # if getattr(self.source[0], self.source[1]) != change.new:
            #     raise TraitError(
            #         f"Broken link {self}: the source value changed while updating "
            #         "the target."
            #     )

    def _update_source(self, change: t.Any) -> None:
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.source[0], self.source[1], self._transform_inv(change.new))
            # this will never happen for us as we are not in an async context TODO true?

            # if getattr(self.target[0], self.target[1]) != change.new:
            #     raise TraitError(
            #         f"Broken link {self}: the target value changed while updating "
            #         "the source."
            #     )

    def unlink(self) -> None:
        self.source[0].unobserve(self._update_target, names=self.source[1])
        self.target[0].unobserve(self._update_source, names=self.target[1])


class dlink:

    updating = False

    def __init__(self, source: t.Any, target: t.Any, transform: t.Any = None) -> None:
        self._transform = transform if transform else lambda x: x
        _validate_link(source, target)
        self.source, self.target = source, target
        self.link()

    def link(self) -> None:
        try:
            setattr(
                self.target[0],
                self.target[1],
                self._transform(getattr(self.source[0], self.source[1])),
            )
        finally:
            self.source[0].observe(self._update, names=self.source[1])

    @contextlib.contextmanager
    def _busy_updating(self) -> t.Any:
        self.updating = True
        try:
            yield
        finally:
            self.updating = False

    def _update(self, change: t.Any) -> None:
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.target[0], self.target[1], self._transform(change.new))

    def unlink(self) -> None:
        self.source[0].unobserve(self._update, names=self.source[1])
