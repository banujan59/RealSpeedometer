#include "ClusterServer/ClusterServerTCP.h"
#include "ClusterController/ClusterController.h"

#include <unistd.h>
#include <iostream>

///
/// Excpects argv[1] to be the server port
///
int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        std::cerr << "Not enough args..." << std::endl;
        std::cout << "Usage: SpeedometerEmbedded portNumber" << std::endl;
        std::cout << "portNumber is the port of the TCP connection to listen to." << std::endl;
        return EXIT_FAILURE;
    }

    std::shared_ptr<ClusterServerTCP> server(new ClusterServerTCP(argv[1]));
    if(server->ServerInitSuccess())
    {
        server->StartReceivingMessages();
    }

    else 
    {
        std::cerr << "Could not start ClusterServer" << std::endl;
        return EXIT_FAILURE;
    }

    auto clusterController = ClusterController::ClusterControllerFactory();
    clusterController->StartClusterUpdate(server);

    std::cout << "SpeedometerEmbedded started!" << std::endl;
    std::cout << "Press enter to terminate program:" << std::endl;
	std::cin.get();
    // Assumes the destructors terminates the threads
    return EXIT_SUCCESS;
}