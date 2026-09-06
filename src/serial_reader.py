import serial
import time

def open_serial(port: str, baud: int, timeout: float = 1.0) -> serial.Serial:
    ser = serial.Serial(port, baudrate=baud, timeout=timeout)
    time.sleep(2.0)            # macOS USB settle
    ser.reset_input_buffer()
    return ser

def read_csv_stream(ser, max_lines=None):
    """
    Minimal, robust CSV reader.
    Expects lines: t_us,ch0,ch1
    """
    n = 0
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 3:
            continue
        try:
            yield int(parts[0]), int(parts[1]), int(parts[2])
            n += 1
            if max_lines and n >= max_lines:
                return
        except ValueError:
            continue
