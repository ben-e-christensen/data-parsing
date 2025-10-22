import serial
import time

# --- CONFIG ---
PORT = "/dev/ttyUSB0"   # Adjust this if needed (e.g. "/dev/ttyACM0" on Linux, "COM3" on Windows)
BAUD = 115200

# --- Open serial connection ---
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # give the board time to reset
    print(f"Connected to {PORT} at {BAUD} baud.\n")
except serial.SerialException as e:
    print(f"❌ Could not open serial port {PORT}: {e}")
    exit()

# --- Read loop ---
try:
    while True:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(line)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nExiting.")
    ser.close()
