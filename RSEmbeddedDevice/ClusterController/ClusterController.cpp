#include "ClusterController.h"
#include "ClusterControllerPi3BP.h"
#include "../Car.h"

#include <string>
#include <iostream>

namespace
{
    constexpr int CLUSTER_UPDATE_DELAY = 10; // in ms
    bool clusterUpdateStopSignal;
    std::unique_ptr<std::thread> clusterUpdateThread;
}

ClusterController::ClusterController()
{
    if (wiringPiSetup () == -1)
        std::cerr << "Wiring Pi Setup failed!" << std::endl;
}

ClusterController::~ClusterController()
{
    StopClusterUpdate();
}


void ClusterController::StartClusterUpdate(std::shared_ptr<ClusterServer> serverPtr) 
{
    clusterUpdateStopSignal = false;
    clusterUpdateThread.reset(new std::thread(&ClusterController::ClusterUpdateThread, this, serverPtr));
}

void ClusterController::StopClusterUpdate() 
{
    if(clusterUpdateThread != nullptr)
    {
        clusterUpdateStopSignal = true;
        clusterUpdateThread->join();
        clusterUpdateThread = nullptr;
    }
}

void ClusterController::ClusterUpdateThread(std::shared_ptr<ClusterServer> serverPtr)
{
    while(!clusterUpdateStopSignal)   
    {
        Car car = serverPtr->GetLatestCarData();
        this->SetSpeed(car.speed);
        this->SetRPM(car.rpm);
        this->SetFuel(car.fuelPercent);
        this->SetEngineTemp(car.engineTempPercent);
        std::this_thread::sleep_for(std::chrono::milliseconds(CLUSTER_UPDATE_DELAY));
    }
}

void ClusterController::SetSpeed(int speed) {}
void ClusterController::SetRPM(int rpm) {}
void ClusterController::SetFuel(int fuelPercent) {}
void ClusterController::SetEngineTemp(int engineTempPercent) {}

std::shared_ptr<ClusterController> ClusterController::ClusterControllerFactory()
{
    std::string cmd = "whoami";
    std::string data;

    FILE * stream;
    const int max_buffer = 256;
    
    char buffer[max_buffer];
    cmd.append(" 2>&1");
    stream = popen(cmd.c_str(), "r");

    if (stream) 
    {
        while (!feof(stream))
            if (fgets(buffer, max_buffer, stream) != NULL) data.append(buffer);

        pclose(stream);
    }

    data = data.substr(0, data.length() - 1); // to remove the '\n' at the end
    if(data != "root")
    {
        std::cout << "WARNING: Not running as root! cluster control disabled." << std::endl;
        return std::make_shared<ClusterController>();
    }

    else
    {
        // Custom code for Raspberry Pi 3B+
        return std::make_shared<ClusterControllerPi3BP>();
    }
}