#Copyright 2023 NXP 
from machine import UART
uart = UART(1)
uart.write("UART WRITE TEST")
uart.read(4)