
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <math.h>
#include "secrets.h"

// Use your laptop's current Wi-Fi IPv4 address

const char* SERVER_URL =
  "http://10.148.244.226:8000/api/readings";

// ---------- Sensor configuration ----------
const int SENSOR_PIN = A0;
const int NUM_SAMPLES = 100;


// Preliminary calibration from your measurements
const float ADC_ZERO = 414.97;
const float ADC_COUNTS_PER_AMP = 42.35;
const float SUPPLY_VOLTAGE = 6.0;  // Temporary nominal value

// ---------- Data transmission ----------
const unsigned long SEND_INTERVAL = 5000;
unsigned long lastSend = 0;

// ---------- Calculate averaged ADC ----------
float readAverageADC() {
  long total = 0;

  for (int i = 0; i < NUM_SAMPLES; i++) {
    total += analogRead(SENSOR_PIN);
    delay(2);
  }

  return total / (float)NUM_SAMPLES;
}

// ---------- Estimate current ----------
float calculateCurrent(float adc) {
  float difference = fabs(adc - ADC_ZERO);
  return difference / ADC_COUNTS_PER_AMP;
}

// ---------- Wi-Fi connection ----------
void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("Wi-Fi connected!");

  Serial.print("ESP IP address: ");
  Serial.println(WiFi.localIP());

  Serial.print("Backend URL: ");
  Serial.println(SERVER_URL);
}

// ---------- Send data to backend ----------
void sendReading(float adc, float currentA) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi disconnected. Reconnecting...");
    WiFi.reconnect();
    return;
  }

  WiFiClient client;
  HTTPClient http;

  http.setTimeout(4000);

  if (!http.begin(client, SERVER_URL)) {
    Serial.println("ERROR: Could not initialize HTTP.");
    return;
  }

  http.addHeader("Content-Type", "application/json");

  String payload = "{\"device\":\"ESP12E\",\"adc\":";
  payload += String(adc, 2);
  payload += ",\"current_a\":";
  payload += String(currentA, 4);
  payload += ",\"voltage_v\":";
  payload += String(SUPPLY_VOLTAGE, 3);
  payload += "}";

  Serial.print("Sending: ");
  Serial.println(payload);

  int httpCode = http.POST(payload);

  if (httpCode > 0) {
    Serial.print("HTTP status: ");
    Serial.println(httpCode);

    Serial.print("Server response: ");
    Serial.println(http.getString());
  } else {
    Serial.print("HTTP request failed: ");
    Serial.println(HTTPClient::errorToString(httpCode));
  }

  http.end();
}

// ---------- Setup ----------
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println();
  Serial.println("Smart Energy Monitor");
  Serial.println("ESP-12E + ACS712");

  connectWiFi();
}

// ---------- Main loop ----------
void loop() {
  float averageADC = readAverageADC();
  float currentA = calculateCurrent(averageADC);

  Serial.print("Average ADC: ");
  Serial.println(averageADC, 2);

  Serial.print("Estimated Current: ");
  Serial.print(currentA, 3);
  Serial.println(" A");

  if (millis() - lastSend >= SEND_INTERVAL) {
    lastSend = millis();
    sendReading(averageADC, currentA);
  }

  delay(100);
}
