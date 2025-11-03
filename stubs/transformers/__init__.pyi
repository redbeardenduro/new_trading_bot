"""Type stubs for transformers library."""

from typing import Any, Dict, List, Optional, Union

class AutoTokenizer:
    """Auto tokenizer class."""

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: Any
    ) -> AutoTokenizer: ...
    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Any]: ...

class AutoModelForSequenceClassification:
    """Auto model for sequence classification."""

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: Any
    ) -> AutoModelForSequenceClassification: ...
    def __call__(self, **kwargs: Any) -> Any: ...

class Pipeline:
    """Transformers pipeline."""

    def __init__(
        self,
        task: str,
        model: Optional[Union[str, Any]] = None,
        tokenizer: Optional[Union[str, Any]] = None,
        **kwargs: Any
    ) -> None: ...
    def __call__(
        self, inputs: Union[str, List[str]], **kwargs: Any
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]: ...

def pipeline(
    task: str,
    model: Optional[Union[str, Any]] = None,
    tokenizer: Optional[Union[str, Any]] = None,
    **kwargs: Any
) -> Pipeline: ...
