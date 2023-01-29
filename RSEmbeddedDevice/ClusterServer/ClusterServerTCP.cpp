#include "ClusterServerTCP.h"

#include <iostream>

#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <inttypes.h>
#include <fcntl.h>


namespace {
    constexpr int RECEIVE_ERROR_MAX_COUNT = 5;

    constexpr int SPEED_SHIFT_BITS = 32;

    constexpr int RPM_SHIFT_BITS = 16;
    constexpr int RPM_MASK = 0xFFFF;

    constexpr int FUEL_SHIFT_BITS = 8;
    constexpr int FUEL_MASK = 0xFF;

    constexpr int ENGINE_TEMP_MASK = 0xFF;
}

ClusterServerTCP::ClusterServerTCP(std::string serverPort) : m_receiveStopSignal(false), m_receiveThread(nullptr)
{
    m_serverInitSuccess = false;

    // Create a socket
    m_sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (m_sockfd < 0)
    {
        std::cerr << "Error creating socket" << std::endl;
        return;
    }
    std::cout << "Socket created!\n";

    // Bind the socket to a local address and port
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;

    int port = 0;
    try {
        port = std::stoi(serverPort);
    }
    catch(std::exception& e)
    {
        std::cerr << "Exception converting the server port" << std::endl;
        return;
    }
    
    serv_addr.sin_port = htons(port);
    if (bind(m_sockfd, (struct sockaddr*) &serv_addr, sizeof(serv_addr)) < 0)
    {
        std::cerr << "Error binding socket" << std::endl;
        return;
    }
    std::cout << "Socket Binded!\n";

    // Listen for incoming connections
    listen(m_sockfd, 1);

    // Accept a connection
    struct sockaddr_in cli_addr;
    socklen_t cli_len = sizeof(cli_addr);
    m_connfd = accept(m_sockfd, (struct sockaddr*) &cli_addr, &cli_len);
    
    if (m_connfd < 0)
    {
        std::cerr << "Error accepting connection" << std::endl;
        return;
    }

    std::cout << "Connected to client !" << std::endl;
    m_serverInitSuccess = true;
}

ClusterServerTCP::~ClusterServerTCP()
{
    // Close the connection and the socket
    StopReceivingMessages();
    close(m_connfd);
    close(m_sockfd);
}

bool ClusterServerTCP::ServerInitSuccess() const 
{
    return m_serverInitSuccess;
}

void ClusterServerTCP::StartReceivingMessages()
{
    if(ServerInitSuccess())
    {
        m_receiveStopSignal = false;
        m_receiveThread.reset(new std::thread(&ClusterServerTCP::ReceiveMessage, this));
    }
}

void ClusterServerTCP::StopReceivingMessages()
{
    if(m_receiveThread != nullptr)
    {
        m_receiveStopSignal = true;  
        m_receiveThread->join();
        m_receiveThread = nullptr;
    } 
}

Car ClusterServerTCP::GetLatestCarData()
{
    std::lock_guard<std::mutex> guard(m_carDataMutex);
    return m_car;
}

void ClusterServerTCP::ReceiveMessage()
{
    int errorCounter = 0;

    while(!m_receiveStopSignal)
    {
        char buffer[8];
        auto n = read(m_connfd, buffer, 8);

        if (n <= 0)
        {
            std::cerr << "Error reading from socket" << std::endl;
            errorCounter++;

            if(errorCounter > RECEIVE_ERROR_MAX_COUNT)
            {
                std::cerr << "Too many bad reading. Ending task." << std::endl;
                m_receiveStopSignal = true;
            }

            continue;
        }

        errorCounter = 0;

        // Convert the data to a uint64_t variable
        uint64_t data;
        memcpy(&data, buffer, sizeof(uint64_t));

        m_carDataMutex.lock();
        m_car.speed = data >> SPEED_SHIFT_BITS;
        m_car.rpm = (data >> RPM_SHIFT_BITS) & RPM_MASK;
        m_car.fuelPercent = (data >> FUEL_SHIFT_BITS) & FUEL_MASK;
        m_car.engineTempPercent = data & ENGINE_TEMP_MASK;
        m_carDataMutex.unlock();
        printf("Speed %d\tRPM %d\tFuel%% %d\tEngine Temp%% %d\n", m_car.speed, m_car.rpm, m_car.fuelPercent, m_car.engineTempPercent);

        // ACK: 
        write(m_connfd, buffer, static_cast<size_t>(n));
    }
}