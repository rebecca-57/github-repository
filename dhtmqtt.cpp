#include <Arduino.h>
#include <DHT.h>
#include <ESP8266WiFi.h>
#include <Ticker.h>
#include <AsyncMqttClient.h>
#include <SPI.h>
#include <Wire.h>

#define WIFI_SSID "wifi name"
#define WIFI_PASSWORD "wifi password"

// Raspberry Pi (laptop) Mosquitto MQTT Broker so that ESP connects to broker
#define MQTT_HOST IPAddress(laptop IP)
#define MQTT_PORT 1883

// MQTT Topics
#define MQTT_PUB_TEMP "esp/dht/temperature"
#define MQTT_PUB_HUM "esp/dht/humidity"

// Digital pin connected to the DHT sensor
#define DHTPIN 5

// define sensor type
#define DHTTYPE DHT11

// Initialize DHT sensor
DHT dht(DHTPIN, DHTTYPE);

// Variables to hold sensor readings
float temp;
float hum;

// Create an AsyncMqttClient object to handle the MQTT client
// timers to reconnect to your MQTT broker and router when it disconnects
AsyncMqttClient mqttClient;
Ticker mqttReconnectTimer;

WiFiEventHandler wifiConnectHandler;
WiFiEventHandler wifiDisconnectHandler;
Ticker wifiReconnectTimer;

unsigned long previousMillis = 0;   // last time temperature was published
const long interval = 10000;        // Interval between sensor readings

//connects ESP to router
void connectToWifi() {
  Serial.println(F("Connecting to Wi-Fi..."));
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

void connectToMqtt();

void onWifiConnect(const WiFiEventStationModeGotIP& event) {
  Serial.println(F("Connected to Wi-Fi."));
  connectToMqtt();
}

void onWifiDisconnect(const WiFiEventStationModeDisconnected& event) {
  Serial.println(F("Disconnected from Wi-Fi."));
  mqttReconnectTimer.detach(); // ensure we don't reconnect to MQTT while reconnecting to Wi-Fi
  wifiReconnectTimer.once(2, connectToWifi);
}

//connects ESP to mqtt broker
void connectToMqtt() {
  Serial.println(F("Connecting to MQTT..."));
  mqttClient.connect();
}

void onMqttConnect(bool sessionPresent) {
  Serial.println(F("Connected to MQTT."));
  Serial.print(F("Session present: "));
  Serial.println(sessionPresent);
}

void onMqttDisconnect(AsyncMqttClientDisconnectReason reason) {
  Serial.println(F("Disconnected from MQTT."));

  if (WiFi.isConnected()) {
    mqttReconnectTimer.once(2, connectToMqtt);
  }
}

/*void onMqttSubscribe(uint16_t packetId, uint8_t qos) {
  Serial.println("Subscribe acknowledged.");
  Serial.print("  packetId: ");
  Serial.println(packetId);
  Serial.print("  qos: ");
  Serial.println(qos);
}

void onMqttUnsubscribe(uint16_t packetId) {
  Serial.println("Unsubscribe acknowledged.");
  Serial.print("  packetId: ");
  Serial.println(packetId);
}*/

void onMqttPublish(uint16_t packetId) {
  Serial.print(F("Publish acknowledged."));
  Serial.print(F("  packetId: "));
  Serial.println(packetId);
}

void setup() {
  Serial.begin(115200);
  Serial.println();

  dht.begin();

//allows MQTT broker and Wi-Fi connection to reconnect if connection is lost
  wifiConnectHandler = WiFi.onStationModeGotIP(onWifiConnect);
  wifiDisconnectHandler = WiFi.onStationModeDisconnected(onWifiDisconnect);

  mqttClient.onConnect(onMqttConnect);
  mqttClient.onDisconnect(onMqttDisconnect);
  //mqttClient.onSubscribe(onMqttSubscribe);
  //mqttClient.onUnsubscribe(onMqttUnsubscribe);
  mqttClient.onPublish(onMqttPublish);
  mqttClient.setServer(MQTT_HOST, MQTT_PORT);

  connectToWifi();
}

void loop() {
  unsigned long currentMillis = millis();
  // Every X seconds it publishes a new MQTT message
  if (currentMillis - previousMillis >= interval) {
    // Save the last time a new reading was published
    previousMillis = currentMillis;
    // New DHT sensor readings
    hum = dht.readHumidity();
    // Read temperature as Celsius (the default)
    temp = dht.readTemperature();
    // Read temperature as Fahrenheit (isFahrenheit = true)
    //temp = dht.readTemperature(true);

    // Publish an MQTT message on topic esp/dht/temperature (topic,qos,retain flag, sensor reading(payload))
    uint16_t packetIdPub1 = mqttClient.publish(MQTT_PUB_TEMP, 1, true, String(temp).c_str());
    Serial.printf("Publishing on topic %s at QoS 1, packetId: %i ", MQTT_PUB_TEMP, packetIdPub1);
    Serial.printf("Temperature: %.2f deg C\n", temp);

    // Publish an MQTT message on topic esp/dht/humidity
    uint16_t packetIdPub2 = mqttClient.publish(MQTT_PUB_HUM, 1, true, String(hum).c_str());
    Serial.printf("Publishing on topic %s at QoS 1, packetId %i: ", MQTT_PUB_HUM, packetIdPub2);
    Serial.printf("Humidity: %.2f %%\n", hum);
    Serial.printf("Time: %lu s\n", (currentMillis/1000));
  }
}
