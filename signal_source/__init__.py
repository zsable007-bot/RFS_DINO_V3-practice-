"""signal_source：可编程调用的 IQ 信号生成库（CW / 2FSK / LFM）。"""

from . import cw, fsk2, lfm

CLASSES = ("CW", "2FSK", "LFM")

_MODULES = {"CW": cw, "2FSK": fsk2, "LFM": lfm}


def generate(class_name, output_path, **params):
    """按类别分发到对应模块的 generate()。"""
    try:
        module = _MODULES[class_name]
    except KeyError:
        raise ValueError(
            f"Unknown class_name {class_name!r}; expected one of {CLASSES}"
        ) from None
    return module.generate(output_path, **params)


__all__ = ["CLASSES", "generate", "cw", "fsk2", "lfm"]
