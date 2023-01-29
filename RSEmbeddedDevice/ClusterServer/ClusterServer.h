#pragma once

#include "../Car.h"

class ClusterServer
{
    protected:
    Car m_car;

    public:
    virtual Car GetLatestCarData() = 0;
};