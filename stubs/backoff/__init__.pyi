"""Type stubs for backoff library."""

from typing import Any, Callable, Optional, Type, TypeVar, Union

F = TypeVar("F", bound=Callable[..., Any])

def expo(
    base: float = 2,
    factor: float = 1,
    max_value: Optional[float] = None,
    jitter: Optional[Callable[[float], float]] = None,
) -> Callable[[int], float]: ...
def constant(interval: float = 1) -> Callable[[int], float]: ...
def linear(
    slope: float = 1,
    start: float = 0,
    jitter: Optional[Callable[[float], float]] = None,
) -> Callable[[int], float]: ...
def on_exception(
    wait_gen: Callable[[int], float],
    exception: Union[Type[Exception], tuple[Type[Exception], ...]],
    max_tries: Optional[int] = None,
    max_time: Optional[float] = None,
    jitter: Optional[Callable[[float], float]] = None,
    on_success: Optional[Callable[[Any], None]] = None,
    on_backoff: Optional[Callable[[Any], None]] = None,
    on_giveup: Optional[Callable[[Any], None]] = None,
    raise_on_giveup: bool = True,
    logger: Optional[str] = None,
    backoff_log_level: int = ...,
    giveup_log_level: int = ...,
    **wait_gen_kwargs: Any
) -> Callable[[F], F]: ...
def on_predicate(
    wait_gen: Callable[[int], float],
    predicate: Optional[Callable[..., bool]] = None,
    max_tries: Optional[int] = None,
    max_time: Optional[float] = None,
    jitter: Optional[Callable[[float], float]] = None,
    on_success: Optional[Callable[[Any], None]] = None,
    on_backoff: Optional[Callable[[Any], None]] = None,
    on_giveup: Optional[Callable[[Any], None]] = None,
    logger: Optional[str] = None,
    backoff_log_level: int = ...,
    giveup_log_level: int = ...,
    **wait_gen_kwargs: Any
) -> Callable[[F], F]: ...
