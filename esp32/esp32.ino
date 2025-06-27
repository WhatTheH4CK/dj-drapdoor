#include <WiFi.h>
#include <WebServer.h>

const char* SSID     = "subscribe-what-the-hack";
const char* PASSWORD = "jAKAskja21Fdh1239jf";
WebServer server(2000);

void handleRoot() {
  int pin   = server.arg("pin").toInt();
  String s = server.arg("state");
  if ((pin==14||pin==15) && (s=="ON"||s=="OFF")) {
    digitalWrite(pin, s=="ON" ? HIGH : LOW);
  }
  server.send(200, "text/plain", "OK");
}

void setup() {
  Serial.begin(115200);
  pinMode(14, OUTPUT); digitalWrite(14, LOW);
  pinMode(15, OUTPUT); digitalWrite(15, LOW);
  WiFi.begin(SSID, PASSWORD);
  while (WiFi.status()!=WL_CONNECTED) delay(500);
  server.on("/", HTTP_GET, handleRoot);
  server.begin();
  Serial.println(WiFi.localIP());
}

void loop() {
  server.handleClient();
}
