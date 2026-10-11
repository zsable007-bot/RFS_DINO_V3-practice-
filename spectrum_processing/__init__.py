"""spectrum_processing：IQ → 事件检测 → 时频图截取（CW / 2FSK / LFM 共用）。"""

from .pipeline import process_record

__all__ = ["process_record"]
