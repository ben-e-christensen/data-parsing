// --- Battery monitor for Adafruit Feather ESP32-V2 ---
// Reads the battery voltage every 5 seconds and prints to Serial.

const int BATTERY_PIN = A13;     // Feather V2 Li-Ion sense pin (labelled "VBAT")
const float ADC_REF = 3.3;       // ADC reference voltage
const int ADC_MAX = 4095;        // 12-bit ADC on ESP32
const float DIVIDER_RATIO = 2.0; // internal 1/2 divider on VBAT pin
const unsigned long PRINT_INTERVAL = 5000; // 5 seconds

unsigned long lastPrint = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Feather V2 Battery Voltage Monitor");
}

void loop() {
  unsigned long now = millis();
  if (now - lastPrint >= PRINT_INTERVAL) {
    lastPrint = now;

    int raw = analogRead(BATTERY_PIN);
    float vMeasured = (raw * ADC_REF) / ADC_MAX; // voltage seen by ADC
    float vBattery  = vMeasured * DIVIDER_RATIO; // actual battery voltage

    Serial.print("Battery voltage: ");
    Serial.print(vBattery, 2);
    Serial.println(" V");
  }
}
