#include <Servo.h>

Servo lock;

int lock_deg = 0;
int unlock_deg = 0;
int incomingByte = 0;

void setup() {
  Serial.begin(19200);
}

void doorlock() {
  incomingByte = Serial.read();
  lock.write(lock_deg);
  if( incomingByte == o){
    Serial.end();
    lock.write(unlock_deg);
    delay(60000);
    lock.write(lock_deg);
    Serial.begin(19200);
  }
}

void loop() {
    doorlock();
}
