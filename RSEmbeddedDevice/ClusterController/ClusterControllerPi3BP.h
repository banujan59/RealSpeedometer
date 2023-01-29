#include "ClusterController.h"

class ClusterControllerPi3BP : public ClusterController
{
    public:
    ClusterControllerPi3BP();
    void SetSpeed(int speed) override;
    void SetRPM(int rpm) override;
    void SetFuel(int fuelPercent) override;
    void SetEngineTemp(int engineTempPercent) override;
};