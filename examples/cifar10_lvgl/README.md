# NXP Application Code Hub

[<img src="https://mcuxpresso.nxp.com/static/icon/nxp-logo-color.svg" width="100"/>](https://www.nxp.com)

## CIFAR10 with LVGL

**This example demonstrates the utilization of the Cifar10 TF-lite Micro model on OpenART.**

**The model is capable of classifying a picture captured from a camera.**

**A straightforward UI has been implemented using the lvgl python library.**

#### Boards: EVK-MIMXRT1060

#### Categories: AI/ML

#### Peripherals: ADC, CLOCKS, FLASH, GPIO, PWM, TIMER, UART, DISPLAY, I2C, I2S, USB, VIDEO, SDMMC, SENSOR, PINCTRL

#### Toolchains: MDK

## Table of Contents

1. [Software](#step1)
2. [Hardware](#step2)
3. [Setup](#step3)
4. [Run steps](#step4)
5. [FAQs](#step5)
6. [Support](#step6)
7. [Release Notes](#step7)

## 1. Software <a name="step1"></a>

* OpenMV IDE ([Download | OpenMV](https://openmv.io/pages/download))

## 2. Hardware <a name="step2"></a>

* i.MXRT1060 EVK ([MIMXRT1060-EVK Product Information|NXP](https://www.nxp.com/part/MIMXRT1060-EVK#/))
* OV7225 / MT9M114 Camera module
* LCD Panel: [RK043FN66HS](https://www.nxp.com/part/RK043FN66HS-CTG)
* SD Card
* Two usb cables


i.MXRT1060 EVK as :

![](images/board_front.png)

## 3. Setup <a name="step3"></a>

Connect Debug USB and USB OTG with laptop through cables.
Reset Boards, two COM devices show up in laptop

![](images/devices.png)

## 4. Run steps <a name="step4"></a>

1. **Copy the model file, labels.txt, and cifar10_lvgl.py into the SD Card.**
2. **Insert the SD Card into the device and reset the board.**
3. **Open the OpenMV IDE and connect the device.**
4. **Execute the cifar10_lvgl.py script in the OpenMV IDE.**
5. **Show a picture to the camera to test the model; the result will be displayed in the top left label of the UI.**

![](images\example.jpg)

## 5. FAQs <a name="step5"></a>



## 6. Support <a name="step6"></a>

[OpenMV | Small - Affordable - Expandable](https://openmv.io/)

[MicroPython - Python for microcontrollers](https://micropython.org/)

#### Project Metadata

<!----- Boards ----->

[![Board badge](https://img.shields.io/badge/Board-EVK–MIMXRT1060-blue)](https://github.com/search?q=org%3Anxp-appcodehub+EVK-MIMXRT1060+in%3Areadme&type=Repositories)

<!----- Categories ----->

[![Category badge](https://img.shields.io/badge/Category-AI/ML-yellowgreen)](https://github.com/search?q=org%3Anxp-appcodehub+aiml+in%3Areadme&type=Repositories)

<!----- Peripherals ----->

[![Peripheral badge](https://img.shields.io/badge/Peripheral-ADC-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+adc+in%3Areadme&type=Repositories) [![Peripheral badge](https://img.shields.io/badge/Peripheral-CLOCKS-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+clocks+in%3Areadme&type=Repositories) [![Peripheral badge](https://img.shields.io/badge/Peripheral-FLASH-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+flash+in%3Areadme&type=Repositories) [![Peripheral badge](https://img.shields.io/badge/Peripheral-GPIO-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+gpio+in%3Areadme&type=Repositories) [![Peripheral badge](https://img.shields.io/badge/Peripheral-PWM-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+pwm+in%3Areadme&type=Repositories) [![Peripheral badge](https://img.shields.io/badge/Peripheral-TIMER-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+timer+in%3Areadme&type=Repositories) [![Peripheral badge](https://img.shields.io/badge/Peripheral-UART-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+uart+in%3Areadme&type=Repositories) [![Peripheral badge](https://img.shields.io/badge/Peripheral-DISPLAY-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+display+in%3Areadme&type=Repositories) [![Peripheral badge](https://img.shields.io/badge/Peripheral-I2C-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+i2c+in%3Areadme&type=Repositories) [![Peripheral badge](https://img.shields.io/badge/Peripheral-I2S-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+i2s+in%3Areadme&type=Repositories) [![Peripheral badge](https://img.shields.io/badge/Peripheral-USB-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+usb+in%3Areadme&type=Repositories) [![Peripheral badge](https://img.shields.io/badge/Peripheral-VIDEO-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+video+in%3Areadme&type=Repositories) [![Peripheral badge](https://img.shields.io/badge/Peripheral-SDMMC-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+sdmmc+in%3Areadme&type=Repositories) [![Peripheral badge](https://img.shields.io/badge/Peripheral-SENSOR-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+sensor+in%3Areadme&type=Repositories) [![Peripheral badge](https://img.shields.io/badge/Peripheral-PINCTRL-yellow)](https://github.com/search?q=org%3Anxp-appcodehub+pinctrl+in%3Areadme&type=Repositories)

<!----- Toolchains ----->

[![Toolchain badge](https://img.shields.io/badge/Toolchain-MDK-orange)](https://github.com/search?q=org%3Anxp-appcodehub+mdk+in%3Areadme&type=Repositories)

Questions regarding the content/correctness of this example can be entered as Issues within this GitHub repository.

> **Warning**: For more general technical questions regarding NXP Microcontrollers and the difference in expected funcionality, enter your questions on the [NXP Community Forum](https://community.nxp.com/)

[![Follow us on Youtube](https://img.shields.io/badge/Youtube-Follow%20us%20on%20Youtube-red.svg)](https://www.youtube.com/@NXP_Semiconductors)
[![Follow us on LinkedIn](https://img.shields.io/badge/LinkedIn-Follow%20us%20on%20LinkedIn-blue.svg)](https://www.linkedin.com/company/nxp-semiconductors)
[![Follow us on Facebook](https://img.shields.io/badge/Facebook-Follow%20us%20on%20Facebook-blue.svg)](https://www.facebook.com/nxpsemi/)
[![Follow us on Twitter](https://img.shields.io/badge/Twitter-Follow%20us%20on%20Twitter-white.svg)](https://twitter.com/NXP)

## 7. Release Notes <a name="step7"></a>

| Version | Description / Update                    |                                  Date |
| :-----: | --------------------------------------- | ------------------------------------: |
|   1.0   | Initial release on Application Code Hub | August 14 <sup>th </sup> 2023 |
