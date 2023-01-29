#pragma once

#include "../ClusterServer/ClusterServer.h"

#include <memory>
#include <thread>
#include <wiringPi.h>

class ClusterController {
    public:
    static std::shared_ptr<ClusterController> ClusterControllerFactory();
    ClusterController();
    virtual ~ClusterController();

    virtual void StartClusterUpdate(std::shared_ptr<ClusterServer> serverPtr);
    virtual void StopClusterUpdate();

    protected:
    virtual void SetSpeed(int speed);
    virtual void SetRPM(int rpm);
    virtual void SetFuel(int fuelPercent);
    virtual void SetEngineTemp(int engineTempPercent);

    private:
    virtual void ClusterUpdateThread(std::shared_ptr<ClusterServer> serverPtr);
};