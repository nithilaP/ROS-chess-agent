// Define pin connections & motor's steps per revolution
const int xDIR = 2;
const int xSTEP = 3;
const int yDIR = 4;
const int ySTEP = 5;
const int zDIR = 6;
const int zSTEP= 7;
const int stepsPerRevolution = 200;

void setup()
{
  // Declare pins as Outputs
  pinMode(xDIR, OUTPUT);
  pinMode(xSTEP, OUTPUT);
  pinMode(yDIR, OUTPUT);
  pinMode(ySTEP, OUTPUT);
  pinMode(zDIR, OUTPUT);
  pinMode(zSTEP, OUTPUT);
  Serial.begin(9600);

  float POS[2] = {0,0};
  float userdata[2] = {0,0};
  bool pincher = false; //true = actively pinching
}

  
void loop()
{
  if (Serial.available() > 0) {
    for (int i=0; i<2;i++) {
      userdata[i] = Serial.read();
    }
    
    Serial.println("Sending to coords: ");
    Serial.println(userdata[0], userdata[1]);
  }
  
  float xdis = userdata[0] - POS[0];
  float ydis = userdata[1] - POS[1]; //distance in inches needed to travel
  // one revolution = 1.88" TEST, thus 200 steps = 1.88"
  float xSteps = 200*xdis/1.88;
  float ySteps = 200*ydis/1.88;
  
  // Set motor direction
  if (xdis > 0) {
    digitalWrite(xDIR, HIGH);
  } else {
    digitalWrite(xDIR, LOW);
  }
  if (ydis > 0) {
    digitalWrite(yDIR, HIGH);
  } else {
    digitalWrite(yDIR, LOW);
  }  

  // Spin motors slowly
  for(int x = 0; x < xSteps; x++)
  {
    digitalWrite(xSTEP, HIGH);
    delayMicroseconds(2000);
    digitalWrite(xSTEP, LOW);
    delayMicroseconds(2000);
  }
  for(int y = 0; y < ySteps; y++)
  {
    digitalWrite(ySTEP, HIGH);
    delayMicroseconds(2000);
    digitalWrite(ySTEP, LOW);
    delayMicroseconds(2000);
  }
  
  delay(1000); // Wait a second

  //Now use pincher
  if (pincher) {
    digitalWrite(zDIR,HIGH);
  }else {
    digitalWrite(zDIR,LOW);
  }
    for(int z = 0; z < 100; z++)//only needs a half rotation (stepsPerRevolution / 2)
  {
    digitalWrite(zSTEP, HIGH);
    delayMicroseconds(2000);
    digitalWrite(zSTEP, LOW);
    delayMicroseconds(2000);
  }
  delay(1000);

  POS[0] = userdata[0];
  POS[1] = userdata[1];
  
  Serial.println("move complete"); //Sanity check
}
