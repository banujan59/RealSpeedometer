#include "ClusterControllerPi3BP.h"

#include <thread>
#include <iostream>
namespace 
{
    constexpr int speedPWMPin = 1;
    
    constexpr int speedMinPWm = 0;
    constexpr int speedMaxPWm = 850;

}
ClusterControllerPi3BP::ClusterControllerPi3BP()
{
    pinMode (speedPWMPin, PWM_OUTPUT) ; /* set PWM pin as output */
}

void ClusterControllerPi3BP::SetSpeed(int speed)
{
    int pwmValue = speed * speedMaxPWm / 100; // TODO update this. this considers 100 as the max speed.
    
    if(pwmValue > speedMaxPWm)
        pwmValue = speedMaxPWm;
    if(pwmValue < speedMinPWm)
        pwmValue = speedMinPWm;

    pwmWrite(speedPWMPin, pwmValue);
}

void ClusterControllerPi3BP::SetRPM(int rpm)
{

}

void ClusterControllerPi3BP::SetFuel(int fuelPercent)
{

}

void ClusterControllerPi3BP::SetEngineTemp(int engineTempPercent) 
{

}