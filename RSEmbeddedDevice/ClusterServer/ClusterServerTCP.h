#pragma once

#include "ClusterServer.h"

#include <string>
#include <thread>
#include <mutex>

class ClusterServerTCP : public ClusterServer
{
    int m_sockfd;
    int m_connfd;
    bool m_serverInitSuccess;

    bool m_receiveStopSignal;
    std::unique_ptr<std::thread> m_receiveThread;
    std::mutex m_carDataMutex;

public:
    ClusterServerTCP(std::string serverPort);
    ~ClusterServerTCP();
    bool ServerInitSuccess() const;
    void StartReceivingMessages();
    void StopReceivingMessages();

    virtual Car GetLatestCarData() override;

private:
    void ReceiveMessage();
};