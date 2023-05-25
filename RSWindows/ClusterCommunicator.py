import socket
from threading import Thread, Lock, Semaphore

class ClusterCommunicator:
    def __init__(self, ip_address, port, disableCommunicator):
        self.__isDisabled = disableCommunicator
        
        if not self.__isDisabled:
            self.__Connect(ip_address=ip_address, port=port)
        
    def __Connect(self, ip_address, port):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((str(ip_address), int(port)))
        
        self._latestDataToSend = 0
        self._dataMutex = Lock()

        self._terminateThreadSignal = False
        self._dataReceivedSemaphore = Semaphore(0)
        self._sendThread = Thread(target=self._SendDataThread)
        self._sendThread.start()

    def SendData(self, data):
        if not self.__isDisabled:
            self._dataMutex.acquire()
            self._latestDataToSend = data
            self._dataMutex.release()
            
            self._dataReceivedSemaphore.release()

    def _SendDataThread(self):
        while not self._terminateThreadSignal:
            with self._dataReceivedSemaphore:
                if self._terminateThreadSignal:
                    continue

                self._dataMutex.acquire()
                dataBytes = self._latestDataToSend.to_bytes(8, 'little')
                self._dataMutex.release()

                self._socket.sendall(dataBytes)
                self._socket.recv(64)

    def __del__(self):
        if not self.__isDisabled:
            self._terminateThreadSignal = True
            self._dataReceivedSemaphore.release()
            self._sendThread.join()

            self._socket.close()