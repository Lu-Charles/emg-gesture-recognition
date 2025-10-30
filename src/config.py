from dataclasses import dataclass

@dataclass(frozen=True)
class StreamConfig:
    fs_hz: int = 500
    baud: int = 230400

    # Windowing for features
    win_sec: float = 0.250      # 250 ms
    hop_sec: float = 0.050      # 50 ms
    # Filters
    bandpass_lo: float = 20.0
    bandpass_hi: float = 200.0
    notch_hz: float = 60.0
    notch_q: float = 30.0