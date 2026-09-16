import serial
import time

PORT = "/dev/cu.usbmodemB43A45B32C382"  # your port
BAUD = 230400

ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(2)
ser.reset_input_buffer()

print("Listening...")
for _ in range(20):
    line = ser.readline().decode(errors="ignore").strip()
    if line:
        print(line)

ser.close()
