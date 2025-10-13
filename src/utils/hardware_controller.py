# hardware_controller.py
from periphery import GPIO
import serial
import time
import threading
import json # <-- 1. Impor library JSON

# --- Your existing Buzzer Controller (Unchanged) ---
class BuzzerController:
    """Controls the buzzer connected to a GPIO pin."""
    def __init__(self, gpio_num):
        self.gpio_num = gpio_num
        self.buzzer = GPIO(self.gpio_num, "out")
        self.buzzer.write(False)  # Ensure buzzer is off initially

    def _beep_thread(self, duration=0.2):
        """The actual beep logic that runs in a separate thread."""
        self.buzzer.write(True)
        time.sleep(duration)
        self.buzzer.write(False)

    def beep(self, duration=0.2):
        """Starts a non-blocking beep."""
        thread = threading.Thread(target=self._beep_thread, args=(duration,), daemon=True)
        thread.start()

    def cleanup(self):
        """Cleans up GPIO resources."""
        self.buzzer.write(False)
        self.buzzer.close()


# --- Updated Motor Controller and other hardware for JSON Communication ---
class MotorController:
    """
    Manages serial (UART) communication using JSON format.
    """
    def __init__(self, port, baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.connect()

    def connect(self):
        """Establishes the serial connection."""
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            print(f"[MotorController] Successfully connected to {self.port}")
            return True
        except serial.SerialException as e:
            print(f"[MotorController] ERROR: Could not open port {self.port}. {e}")
            self.ser = None
            return False


    # for led control
    def set_led_brightness(self, level):
        """Mengirim perintah untuk mengatur level kecerahan LED."""
        command = {"cmd": "SET_BRIGHTNESS", "level": level}
        return self.send_command(command)


    # for speed control
    def set_motor_speed(self, level):
        """Mengirim perintah untuk mengatur level kecepatan motor."""
        command = {"cmd": "SET_SPEED", "level": level}
        return self.send_command(command)

    def send_command(self, command_dict):
        """
        Mengubah dictionary menjadi string JSON dan mengirimkannya.

        Args:
            command_dict (dict): Perintah dalam format dictionary (e.g., {"cmd": "START"}).
        """
        if not self.ser or not self.ser.is_open:
            print("[MotorController] ERROR: Not connected. Cannot send command.")
            return False

        try:
            # 2. Ubah dictionary menjadi string JSON
            json_string = json.dumps(command_dict)

            # Tambahkan newline sebagai penanda akhir pesan
            full_message = json_string + '\n'

            self.ser.write(full_message.encode('utf-8'))
            print(f"[MotorController] Sent JSON: {json_string}")
            return True
        except Exception as e:
            print(f"[MotorController] ERROR: Failed to send JSON data. {e}")
            return False

    def start_motor(self):
        """Mengirim perintah untuk menyalakan motor."""
        # 3. Buat dictionary perintahnya
        command = {"cmd": "START MOTOR"}
        return self.send_command(command)

    def stop_motor(self):
        """Mengirim perintah untuk mematikan motor."""
        command = {"cmd": "STOP MOTOR"}
        return self.send_command(command)

    def request_battery_voltage(self):
        """Mengirim permintaan untuk mendapatkan data tegangan baterai."""
        command = {"cmd": "BAT"}
        return self.send_command(command)

    #PLUGGED OR NOT
    def request_system_status(self):
        """Mengirim permintaan untuk mendapatkan status sistem (seperti status AC)."""
        command = {"cmd": "GET_STATUS"}
        return self.send_command(command)

    def read_response(self):
        """Membaca dan mem-parsing respons JSON dari microcontroller."""
        if not self.ser or not self.ser.is_open or self.ser.in_waiting == 0:
            return None # Tidak ada data untuk dibaca

        try:
            response_line = self.ser.readline().decode('utf-8').strip()
            if response_line:
                # Ubah string JSON yang diterima menjadi dictionary Python
                response_dict = json.loads(response_line)
                print(f"[MotorController] Received JSON: {response_dict}")
                return response_dict
        except json.JSONDecodeError:
            print(f"[MotorController] ERROR: Received invalid JSON: {response_line}")
        except Exception as e:
            print(f"[MotorController] ERROR: Failed to read data. {e}")

        return None

    # start and stop charging
    def start_charging(self):
        """Mengirim perintah untuk memulai urutan charging."""
        command = {"cmd": "START_CHARGE"}
        return self.send_command(command)

    def stop_charging(self):
        """Mengirim perintah untuk menghentikan charging."""
        command = {"cmd": "STOP_CHARGE"}
        return self.send_command(command)


    def close(self):
        """Closes the serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"[MotorController] Disconnected from {self.port}")


# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    MOTOR_PORT = 'COM8'
    print(f"--- Testing MotorController with JSON on port {MOTOR_PORT} ---")
    motor = MotorController(port=MOTOR_PORT)

    if motor.ser:
        try:
            print("\nMengirim perintah START...")
            motor.start_motor()
            time.sleep(3)

            print("\nMengirim perintah STOP...")
            motor.stop_motor()
            time.sleep(1)

            print("\nMeminta data baterai...")
            motor.request_battery_voltage()
            # Coba baca respons dari uC (untuk simulasi)
            # Di tes nyata, ESP32 Anda akan mengirim balasan.
            response = motor.read_response()
            if response:
                print(f"Respons diterima: {response}")
            else:
                print("Tidak ada respons diterima (sesuai ekspektasi dalam tes ini).")


        finally:
            motor.close()
    else:
        print("\nCould not run motor test because the serial port failed to open.")
