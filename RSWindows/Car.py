from threading import Thread
#from ClusterCommunicator import ClusterCommunicator

class Car:
    def __init__ (self):
        self.__fuelPercent = 0
        self.__engineTempPercent = 0
        self.__rpm = 0
        self.__speed = 0
    
    def SetData(self, newSpeed, newRPM, newFuelPercent, newEngineTempPercent):
        self.__speed = int(newSpeed)
        self.__rpm = int(newRPM)
        self.__fuelPercent = int(newFuelPercent)
        self.__engineTempPercent = int(newEngineTempPercent)

    def encodeData(self):
        # Data bit design:
        # Minimum requirement: 
        # Fuel percentage (0 - 100) : 8 bits
        # Engine temp percentage (0-100): 8 bits
        # RPM (0 - 10 000): 16 bits
        # Speed: (0 - 600) : 24 bits
        # 64 bit required

        bit = self.__speed << 16
        bit |= self.__rpm
        bit = bit << 8
        bit |= self.__fuelPercent
        bit = bit << 8
        bit |= self.__engineTempPercent
        return bit

    def __del__(self):
        print('Destructor called.')



# # To test the encoding:
# myCar = Car()
# myCar.SetData(100, 2500, 0, 0)
# bit = myCar.encodeData()
# print("Value is:", bit)