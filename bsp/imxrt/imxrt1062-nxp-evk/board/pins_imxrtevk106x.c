// args= ['.\\make-pins.py', '-b', '.\\pins.csv', '--af', '.\\mimxrt105x_af.csv', '--pinmap', '.\\mimxrt105x_pinmap.csv', '--prefix', 'mimxrt105x_prefix.c', '--qstr', '.\\pins_qstr.h', '--af-const', '.\\pins_af_const.h', '--af-py', '.\\pins_af.py']
// This file was automatically generated by make-pins.py
//
// --af .\mimxrt105x_af.csv
// --pinmap .\mimxrt105x_pinmap.csv
// --board .\pins.csv
// --prefix mimxrt105x_prefix.c
/*
 * Copyright (c) 2006-2018, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */
// <MCU>_prefix.c becomes the initial portion of the generated pins file.

#include <stdio.h>

#include <stdio.h>
#include "mpy_pin.h"
#include "py/obj.h"
#include "py/mphal.h"
#include <drivers/pin.h>

#define AF(af_idx, af_fn, af_unit, af_type, af_ptr, af_inSelReg, af_inSelVal) \
{ \
    { &pin_af_type }, \
    .name = MP_QSTR_AF ## af_idx ## _ ## af_fn ## af_unit, \
    .idx = (af_idx), \
    .fn = AF_FN_ ## af_fn, \
    .unit = (af_unit), \
    .reg = (af_ptr), \
    .inSelReg = (af_inSelReg), \
    .inSelVal = (af_inSelVal), \
}

#define PIN(p_name, p_board_name, p_port, p_pin, p_af, p_adc_num, p_adc_channel, p_afReg, p_cfgReg) \
{ \
    { &mpy_pin_type }, \
    .name = MP_QSTR_ ## p_name, \
    .board_name = MP_QSTR_ ## p_board_name, \
    .port = PORT_ ## p_port, \
    .pin = (p_pin), \
    .num_af = (sizeof(p_af) / sizeof(pin_af_obj_t)), \
    .pin_mask = (1UL << ((p_pin) & 0x1f)), \
    .gpio = GPIO ## p_port, \
    .af = p_af, \
    .adc_num = p_adc_num, \
    .adc_channel = p_adc_channel, \
    .afReg = p_afReg, \
    .cfgReg = p_cfgReg, \
}

const pin_af_obj_t pin_AD_B0_01_af[] = {
  AF( 0, PWM     ,  2, PWMB3           , PWM2       , 0x401F8484U, 0x2U), // FLEXPWM2_PWMB3
  //( 1, XBAR    ,  1, XBAR_INOUT15    , XBAR1      , 0x401F8648U, 0x0U), // XBAR1_XBAR_INOUT15
  //( 2, ANATOP  ,  0, 24M_OUT         , ANATOP     , 0          , 0   ), // ANATOP_24M_OUT
  //( 3, ANATOP  ,  0, OTG1_ID         , ANATOP     , 0x401F83F4U, 0x0U), // ANATOP_OTG1_ID
  //( 4, LPI2C   ,  1, SDAS            , LPI2C1     , 0          , 0   ), // LPI2C1_SDAS
  AF( 5, GPIO    ,  1, PIN1            , GPIO1      , 0          , 0   ), // GPIO1_PIN1
  //( 6, EWM     ,  0, EWM_OUT_B       , EWM        , 0          , 0   ), // EWM_EWM_OUT_B
  AF( 7, LPSPI   ,  3, SDO             , LPSPI3     , 0x401F8518U, 0x0U), // LPSPI3_SDO
  //(15, AD      ,  0, B0_01           , AD         ,            ,     ), // AD_B0_01
};

const pin_obj_t pin_AD_B0_01 = PIN(AD_B0_01, D14_SDA, 1, 1, pin_AD_B0_01_af, 0, 0, 0x401F80C0U, 0x401F82B0U);

const pin_af_obj_t pin_AD_B0_02_af[] = {
  //( 0, FLEXCAN ,  2, TX              , FLEXCAN2   , 0          , 0   ), // FLEXCAN2_TX
  //( 1, XBAR    ,  1, XBAR_INOUT16    , XBAR1      , 0x401F864CU, 0x0U), // XBAR1_XBAR_INOUT16
  AF( 2, LPUART  ,  6, TX              , LPUART6    , 0x401F8554U, 0x1U), // LPUART6_TX
  //( 3, USB     ,  0, OTG1_PWR        , USB        , 0          , 0   ), // USB_OTG1_PWR
  AF( 4, PWM     ,  1, PWMX0           , PWM1       , 0          , 0   ), // FLEXPWM1_PWMX0
  AF( 5, GPIO    ,  1, PIN2            , GPIO1      , 0          , 0   ), // GPIO1_PIN2
  //( 6, LPI2C   ,  1, HREQ            , LPI2C1     , 0          , 0   ), // LPI2C1_HREQ
  AF( 7, LPSPI   ,  3, SDI             , LPSPI3     , 0x401F8514U, 0x0U), // LPSPI3_SDI
  //(15, AD      ,  0, B0_02           , AD         ,            ,     ), // AD_B0_02
};

const pin_obj_t pin_AD_B0_02 = PIN(AD_B0_02, D15_SCL, 1, 2, pin_AD_B0_02_af, 0, 0, 0x401F80C4U, 0x401F82B4U);

const pin_af_obj_t pin_AD_B0_03_af[] = {
  //( 0, FLEXCAN ,  2, RX              , FLEXCAN2   , 0x401F8450U, 0x1U), // FLEXCAN2_RX
  //( 1, XBAR    ,  1, XBAR_INOUT17    , XBAR1      , 0x401F862CU, 0x1U), // XBAR1_XBAR_INOUT17
  AF( 2, LPUART  ,  6, RX              , LPUART6    , 0x401F8550U, 0x1U), // LPUART6_RX
  //( 3, USB     ,  0, OTG1_OC         , USB        , 0x401F85D0U, 0x0U), // USB_OTG1_OC
  AF( 4, PWM     ,  1, PWMX1           , PWM1       , 0          , 0   ), // FLEXPWM1_PWMX1
  AF( 5, GPIO    ,  1, PIN3            , GPIO1      , 0          , 0   ), // GPIO1_PIN3
  //( 6, ANATOP  ,  0, 24M_OUT         , ANATOP     , 0          , 0   ), // ANATOP_24M_OUT
  AF( 7, LPSPI   ,  3, PCS0            , LPSPI3     , 0x401F850CU, 0x0U), // LPSPI3_PCS0
  //(15, AD      ,  0, B0_03           , AD         ,            ,     ), // AD_B0_03
};

const pin_obj_t pin_AD_B0_03 = PIN(AD_B0_03, D8, 1, 3, pin_AD_B0_03_af, 0, 0, 0x401F80C8U, 0x401F82B8U);

const pin_af_obj_t pin_AD_B0_09_af[] = {
  //( 0, JTAG    ,  0, TDI             , JTAG       , 0          , 0   ), // JTAG_TDI
  AF( 1, PWM     ,  2, PWMA3           , PWM2       , 0x401F8474U, 0x3U), // FLEXPWM2_PWMA3
  //( 2, ENET    ,  0, RDATA2          , ENET       , 0          , 0   ), // ENET_RDATA2
#if (defined(MICROPY_HW_ENABLE_SAI2) && MICROPY_HW_ENABLE_SAI2)
  AF( 3, SAI     ,  2, TX_DATA         , SAI2       , 0          , 0   ), // SAI2_TX_DATA
#endif
  //( 4, CSI     ,  0, DATA4           , CSI        , 0x401F8408U, 0x1U), // CSI_DATA4
  AF( 5, GPIO    ,  1, PIN9            , GPIO1      , 0          , 0   ), // GPIO1_PIN9
  //( 6, XBAR    ,  1, XBAR_IN21       , XBAR1      , 0x401F8658U, 0x1U), // XBAR1_XBAR_IN21
  AF( 7, GPT     ,  2, CLK             , GPT2       , 0          , 0   ), // GPT2_CLK
  //(15, AD      ,  0, B0_09           , AD         ,            ,     ), // AD_B0_09
};

const pin_obj_t pin_AD_B0_09 = PIN(AD_B0_09, LED, 1, 9, pin_AD_B0_09_af, 0, 0, 0x401F80E0U, 0x401F82D0U);

const pin_af_obj_t pin_AD_B0_10_af[] = {
  //( 0, JTAG    ,  0, TDO             , JTAG       , 0          , 0   ), // JTAG_TDO
  AF( 1, PWM     ,  1, PWMA3           , PWM1       , 0x401F8454U, 0x3U), // FLEXPWM1_PWMA3
  //( 2, ENET    ,  0, CRS             , ENET       , 0          , 0   ), // ENET_CRS
#if (defined(MICROPY_HW_ENABLE_SAI2) && MICROPY_HW_ENABLE_SAI2)
  AF( 3, SAI     ,  2, MCLK            , SAI2       , 0x401F85B0U, 0x1U), // SAI2_MCLK
#endif
  //( 4, CSI     ,  0, DATA3           , CSI        , 0x401F8404U, 0x1U), // CSI_DATA3
  AF( 5, GPIO    ,  1, PIN10           , GPIO1      , 0          , 0   ), // GPIO1_PIN10
  //( 6, XBAR    ,  1, XBAR_IN22       , XBAR1      , 0x401F8638U, 0x1U), // XBAR1_XBAR_IN22
  //( 7, ENET    ,  0, 1588_EVENT0_OUT , ENET       , 0          , 0   ), // ENET_1588_EVENT0_OUT
  //(15, AD      ,  0, B0_10           , AD         ,            ,     ), // AD_B0_10
};

const pin_obj_t pin_AD_B0_10 = PIN(AD_B0_10, D5_PWM1, 1, 10, pin_AD_B0_10_af, 0, 0, 0x401F80E4U, 0x401F82D4U);

const pin_af_obj_t pin_AD_B0_11_af[] = {
  //( 0, JTAG    ,  0, TRSTB           , JTAG       , 0          , 0   ), // JTAG_TRSTB
  AF( 1, PWM     ,  1, PWMB3           , PWM1       , 0x401F8464U, 0x3U), // FLEXPWM1_PWMB3
  //( 2, ENET    ,  0, COL             , ENET       , 0          , 0   ), // ENET_COL
  //( 3, WDOG    ,  1, WDOG_B          , WDOG1      , 0          , 0   ), // WDOG1_WDOG_B
  //( 4, CSI     ,  0, DATA2           , CSI        , 0x401F8400U, 0x1U), // CSI_DATA2
  AF( 5, GPIO    ,  1, PIN11           , GPIO1      , 0          , 0   ), // GPIO1_PIN11
  //( 6, XBAR    ,  1, XBAR_IN23       , XBAR1      , 0x401F863CU, 0x1U), // XBAR1_XBAR_IN23
  //( 7, ENET    ,  0, 1588_EVENT0_IN  , ENET       , 0x401F8444U, 0x1U), // ENET_1588_EVENT0_IN
  //(15, AD      ,  0, B0_11           , AD         ,            ,     ), // AD_B0_11
};

const pin_obj_t pin_AD_B0_11 = PIN(AD_B0_11, D2_INT0, 1, 11, pin_AD_B0_11_af, 0, 0, 0x401F80E8U, 0x401F82D8U);

const pin_af_obj_t pin_AD_B0_12_af[] = {
  AF( 0, LPI2C   ,  4, SCL             , LPI2C4     , 0x401F84E4U, 0x1U), // LPI2C4_SCL
  //( 1, CCM     ,  0, PMIC_RDY        , CCM        , 0x401F83FCU, 0x1U), // CCM_PMIC_RDY
  AF( 2, LPUART  ,  1, TX              , LPUART1    , 0          , 0   ), // LPUART1_TX
  //( 3, WDOG    ,  2, WDOG_B          , WDOG2      , 0          , 0   ), // WDOG2_WDOG_B
  AF( 4, PWM     ,  1, PWMX2           , PWM1       , 0          , 0   ), // FLEXPWM1_PWMX2
  AF( 5, GPIO    ,  1, PIN12           , GPIO1      , 0          , 0   ), // GPIO1_PIN12
  //( 6, ENET    ,  0, 1588_EVENT1_OUT , ENET       , 0          , 0   ), // ENET_1588_EVENT1_OUT
  //( 7, CM      ,  7, NMI             , CM7        , 0x401F8568U, 0   ), // CM7_NMI
  //(15, AD      ,  0, B0_12           , AD         ,            ,     ), // AD_B0_12
};

const pin_obj_t pin_AD_B0_12 = PIN(AD_B0_12, DBG_TXD, 1, 12, pin_AD_B0_12_af, PIN_ADC1, 1, 0x401F80ECU, 0x401F82DCU);

const pin_af_obj_t pin_AD_B0_13_af[] = {
  AF( 0, LPI2C   ,  4, SDA             , LPI2C4     , 0x401F84E8U, 0x1U), // LPI2C4_SDA
  AF( 1, GPT     ,  1, CLK             , GPT1       , 0          , 0   ), // GPT1_CLK
  AF( 2, LPUART  ,  1, RX              , LPUART1    , 0          , 0   ), // LPUART1_RX
  //( 3, EWM     ,  0, EWM_OUT_B       , EWM        , 0          , 0   ), // EWM_EWM_OUT_B
  AF( 4, PWM     ,  1, PWMX3           , PWM1       , 0          , 0   ), // FLEXPWM1_PWMX3
  AF( 5, GPIO    ,  1, PIN13           , GPIO1      , 0          , 0   ), // GPIO1_PIN13
  //( 6, ENET    ,  0, 1588_EVENT1_IN  , ENET       , 0          , 0   ), // ENET_1588_EVENT1_IN
  //( 7, ANATOP  ,  0, 24M_OUT         , ANATOP     , 0          , 0   ), // ANATOP_24M_OUT
  //(15, AD      ,  0, B0_13           , AD         ,            ,     ), // AD_B0_13
};

const pin_obj_t pin_AD_B0_13 = PIN(AD_B0_13, DBG_RXD, 1, 13, pin_AD_B0_13_af, PIN_ADC1, 2, 0x401F80F0U, 0x401F82E0U);

const pin_af_obj_t pin_AD_B1_00_af[] = {
  //( 0, ANATOP  ,  0, OTG2_ID         , ANATOP     , 0x401F83F8U, 0x1U), // ANATOP_OTG2_ID
  AF( 1, TMR     ,  3, TIMER0          , TMR3       , 0x401F857CU, 0x1U), // TMR3_TIMER0
  AF( 2, LPUART  ,  2, CTS_B           , LPUART2    , 0          , 0   ), // LPUART2_CTS_B
  AF( 3, LPI2C   ,  1, SCL             , LPI2C1     , 0x401F84CCU, 0x1U), // LPI2C1_SCL
  //( 4, WDOG    ,  1, WDOG_B          , WDOG1      , 0          , 0   ), // WDOG1_WDOG_B
  AF( 5, GPIO    ,  1, PIN16           , GPIO1      , 0          , 0   ), // GPIO1_PIN16
  //( 6, USDHC   ,  1, WP              , USDHC1     , 0x401F85D8U, 0x2U), // USDHC1_WP
  //( 7, KPP     ,  0, ROW7            , KPP        , 0          , 0   ), // KPP_ROW7
  //(15, AD      ,  0, B1_00           , AD         ,            ,     ), // AD_B1_00
};

const pin_obj_t pin_AD_B1_00 = PIN(AD_B1_00, A5, 1, 16, pin_AD_B1_00_af, PIN_ADC1, 5, 0x401F80FCU, 0x401F82ECU);

const pin_af_obj_t pin_AD_B1_01_af[] = {
  //( 0, USB     ,  0, OTG1_PWR        , USB        , 0          , 0   ), // USB_OTG1_PWR
  AF( 1, TMR     ,  3, TIMER1          , TMR3       , 0x401F8580U, 0x0U), // TMR3_TIMER1
  AF( 2, LPUART  ,  2, RTS_B           , LPUART2    , 0          , 0   ), // LPUART2_RTS_B
  AF( 3, LPI2C   ,  1, SDA             , LPI2C1     , 0x401F84D0U, 0x1U), // LPI2C1_SDA
  //( 4, CCM     ,  0, PMIC_RDY        , CCM        , 0x401F83FCU, 0x2U), // CCM_PMIC_RDY
  AF( 5, GPIO    ,  1, PIN17           , GPIO1      , 0          , 0   ), // GPIO1_PIN17
  //( 6, USDHC   ,  1, VSELECT         , USDHC1     , 0          , 0   ), // USDHC1_VSELECT
  //( 7, KPP     ,  0, COL7            , KPP        , 0          , 0   ), // KPP_COL7
  //(15, AD      ,  0, B1_01           , AD         ,            ,     ), // AD_B1_01
};

const pin_obj_t pin_AD_B1_01 = PIN(AD_B1_01, A4, 1, 17, pin_AD_B1_01_af, PIN_ADC1, 6, 0x401F8100U, 0x401F82F0U);

const pin_af_obj_t pin_AD_B1_02_af[] = {
  //( 0, ANATOP  ,  0, OTG1_ID         , ANATOP     , 0x401F83F4U, 0x1U), // ANATOP_OTG1_ID
  AF( 1, TMR     ,  3, TIMER2          , TMR3       , 0x401F8584U, 0x1U), // TMR3_TIMER2
  AF( 2, LPUART  ,  2, TX              , LPUART2    , 0x401F8530U, 0x1U), // LPUART2_TX
  //( 3, SPDIF   ,  0, OUT             , SPDIF      , 0          , 0   ), // SPDIF_OUT
  //( 4, ENET    ,  0, 1588_EVENT2_OUT , ENET       , 0          , 0   ), // ENET_1588_EVENT2_OUT
  AF( 5, GPIO    ,  1, PIN18           , GPIO1      , 0          , 0   ), // GPIO1_PIN18
  //( 6, USDHC   ,  1, CD_B            , USDHC1     , 0x401F85D4U, 0x1U), // USDHC1_CD_B
  //( 7, KPP     ,  0, ROW6            , KPP        , 0          , 0   ), // KPP_ROW6
  //(15, AD      ,  0, B1_02           , AD         ,            ,     ), // AD_B1_02
};

const pin_obj_t pin_AD_B1_02 = PIN(AD_B1_02, D6_PWM2, 1, 18, pin_AD_B1_02_af, PIN_ADC1, 7, 0x401F8104U, 0x401F82F4U);

const pin_af_obj_t pin_AD_B1_03_af[] = {
  //( 0, USB     ,  0, OTG1_OC         , USB        , 0x401F85D0U, 0x1U), // USB_OTG1_OC
  AF( 1, TMR     ,  3, TIMER3          , TMR3       , 0x401F8588U, 0x1U), // TMR3_TIMER3
  AF( 2, LPUART  ,  2, RX              , LPUART2    , 0x401F852CU, 0x1U), // LPUART2_RX
  //( 3, SPDIF   ,  0, IN              , SPDIF      , 0x401F85C8U, 0x0U), // SPDIF_IN
  //( 4, ENET    ,  0, 1588_EVENT2_IN  , ENET       , 0          , 0   ), // ENET_1588_EVENT2_IN
  AF( 5, GPIO    ,  1, PIN19           , GPIO1      , 0          , 0   ), // GPIO1_PIN19
  //( 6, USDHC   ,  2, CD_B            , USDHC2     , 0x401F85E0U, 0x0U), // USDHC2_CD_B
  //( 7, KPP     ,  0, COL6            , KPP        , 0          , 0   ), // KPP_COL6
  //(15, AD      ,  0, B1_03           , AD         ,            ,     ), // AD_B1_03
};

const pin_obj_t pin_AD_B1_03 = PIN(AD_B1_03, D7_PWM3, 1, 19, pin_AD_B1_03_af, PIN_ADC1, 8, 0x401F8108U, 0x401F82F8U);

const pin_af_obj_t pin_AD_B1_04_af[] = {
  //( 0, FLEXSPI ,  0, B_DATA3         , FLEXSPI    , 0x401F84C4U, 0x1U), // FLEXSPI_B_DATA3
  //( 1, ENET    ,  0, MDC             , ENET       , 0          , 0   ), // ENET_MDC
  AF( 2, LPUART  ,  3, CTS_B           , LPUART3    , 0x401F8534U, 0x1U), // LPUART3_CTS_B
  //( 3, SPDIF   ,  0, SR_CLK          , SPDIF      , 0          , 0   ), // SPDIF_SR_CLK
  //( 4, CSI     ,  0, PIXCLK          , CSI        , 0x401F8424U, 0x0U), // CSI_PIXCLK
  AF( 5, GPIO    ,  1, PIN20           , GPIO1      , 0          , 0   ), // GPIO1_PIN20
  //( 6, USDHC   ,  2, DATA0           , USDHC2     , 0x401F85E8U, 0x1U), // USDHC2_DATA0
  //( 7, KPP     ,  0, ROW5            , KPP        , 0          , 0   ), // KPP_ROW5
  //(15, AD      ,  0, B1_04           , AD         ,            ,     ), // AD_B1_04
};

const pin_obj_t pin_AD_B1_04 = PIN(AD_B1_04, A2, 1, 20, pin_AD_B1_04_af, PIN_ADC1, 9, 0x401F810CU, 0x401F82FCU);

const pin_af_obj_t pin_AD_B1_05_af[] = {
  //( 0, FLEXSPI ,  0, B_DATA2         , FLEXSPI    , 0x401F84C0U, 0x1U), // FLEXSPI_B_DATA2
  //( 1, ENET    ,  0, MDIO            , ENET       , 0x401F8430U, 0x0U), // ENET_MDIO
  AF( 2, LPUART  ,  3, RTS_B           , LPUART3    , 0          , 0   ), // LPUART3_RTS_B
  //( 3, SPDIF   ,  0, OUT             , SPDIF      , 0          , 0   ), // SPDIF_OUT
  //( 4, CSI     ,  0, MCLK            , CSI        , 0          , 0   ), // CSI_MCLK
  AF( 5, GPIO    ,  1, PIN21           , GPIO1      , 0          , 0   ), // GPIO1_PIN21
  //( 6, USDHC   ,  2, DATA1           , USDHC2     , 0x401F85ECU, 0x1U), // USDHC2_DATA1
  //( 7, KPP     ,  0, COL5            , KPP        , 0          , 0   ), // KPP_COL5
  //(15, AD      ,  0, B1_05           , AD         ,            ,     ), // AD_B1_05
};

const pin_obj_t pin_AD_B1_05 = PIN(AD_B1_05, A3, 1, 21, pin_AD_B1_05_af, PIN_ADC1, 10, 0x401F8110U, 0x401F8300U);

const pin_af_obj_t pin_AD_B1_06_af[] = {
  //( 0, FLEXSPI ,  0, B_DATA1         , FLEXSPI    , 0x401F84BCU, 0x1U), // FLEXSPI_B_DATA1
  AF( 1, LPI2C   ,  3, SDA             , LPI2C3     , 0x401F84E0U, 0x2U), // LPI2C3_SDA
  AF( 2, LPUART  ,  3, TX              , LPUART3    , 0x401F853CU, 0x0U), // LPUART3_TX
  //( 3, SPDIF   ,  0, LOCK            , SPDIF      , 0          , 0   ), // SPDIF_LOCK
  //( 4, CSI     ,  0, VSYNC           , CSI        , 0x401F8428U, 0x1U), // CSI_VSYNC
  AF( 5, GPIO    ,  1, PIN22           , GPIO1      , 0          , 0   ), // GPIO1_PIN22
  //( 6, USDHC   ,  2, DATA2           , USDHC2     , 0x401F85F0U, 0x1U), // USDHC2_DATA2
  //( 7, KPP     ,  0, ROW4            , KPP        , 0          , 0   ), // KPP_ROW4
  //(15, AD      ,  0, B1_06           , AD         ,            ,     ), // AD_B1_06
};

const pin_obj_t pin_AD_B1_06 = PIN(AD_B1_06, D1_TX, 1, 22, pin_AD_B1_06_af, PIN_ADC1, 11, 0x401F8114U, 0x401F8304U);

const pin_af_obj_t pin_AD_B1_07_af[] = {
  //( 0, FLEXSPI ,  0, B_DATA0         , FLEXSPI    , 0x401F84B8U, 0x1U), // FLEXSPI_B_DATA0
  AF( 1, LPI2C   ,  3, SCL             , LPI2C3     , 0x401F84DCU, 0x2U), // LPI2C3_SCL
  AF( 2, LPUART  ,  3, RX              , LPUART3    , 0x401F8538U, 0x0U), // LPUART3_RX
  //( 3, SPDIF   ,  0, EXT_CLK         , SPDIF      , 0          , 0   ), // SPDIF_EXT_CLK
  //( 4, CSI     ,  0, HSYNC           , CSI        , 0x401F8420U, 0x1U), // CSI_HSYNC
  AF( 5, GPIO    ,  1, PIN23           , GPIO1      , 0          , 0   ), // GPIO1_PIN23
  //( 6, USDHC   ,  2, DATA3           , USDHC2     , 0x401F85F4U, 0x1U), // USDHC2_DATA3
  //( 7, KPP     ,  0, COL4            , KPP        , 0          , 0   ), // KPP_COL4
  //(15, AD      ,  0, B1_07           , AD         ,            ,     ), // AD_B1_07
};

const pin_obj_t pin_AD_B1_07 = PIN(AD_B1_07, D0_RX, 1, 23, pin_AD_B1_07_af, PIN_ADC1, 12, 0x401F8118U, 0x401F8308U);

const pin_af_obj_t pin_AD_B1_08_af[] = {
  //( 0, FLEXSPI ,  0, A_SS1_B         , FLEXSPI    , 0          , 0   ), // FLEXSPI_A_SS1_B
  AF( 1, PWM     ,  4, PWMA0           , PWM4       , 0x401F8494U, 0x1U), // FLEXPWM4_PWMA0
  //( 2, FLEXCAN ,  1, TX              , FLEXCAN1   , 0          , 0   ), // FLEXCAN1_TX
  //( 3, CCM     ,  0, PMIC_RDY        , CCM        , 0x401F83FCU, 0x3U), // CCM_PMIC_RDY
  //( 4, CSI     ,  0, DATA9           , CSI        , 0x401F841CU, 0x0U), // CSI_DATA9
  AF( 5, GPIO    ,  1, PIN24           , GPIO1      , 0          , 0   ), // GPIO1_PIN24
  //( 6, USDHC   ,  2, CMD             , USDHC2     , 0x401F85E4U, 0x1U), // USDHC2_CMD
  //( 7, KPP     ,  0, ROW3            , KPP        , 0          , 0   ), // KPP_ROW3
  //(15, AD      ,  0, B1_08           , AD         ,            ,     ), // AD_B1_08
};

const pin_obj_t pin_AD_B1_08 = PIN(AD_B1_08, D3_INT1, 1, 24, pin_AD_B1_08_af, PIN_ADC1, 13, 0x401F811CU, 0x401F830CU);

const pin_af_obj_t pin_AD_B1_10_af[] = {
  //( 0, FLEXSPI ,  0, A_DATA3         , FLEXSPI    , 0x401F84B4U, 0x1U), // FLEXSPI_A_DATA3
  //( 1, WDOG    ,  1, WDOG_B          , WDOG1      , 0          , 0   ), // WDOG1_WDOG_B
  AF( 2, LPUART  ,  8, TX              , LPUART8    , 0x401F8564U, 0x1U), // LPUART8_TX
#if (defined(MICROPY_HW_ENABLE_SAI1) && MICROPY_HW_ENABLE_SAI1)
  AF( 3, SAI     ,  1, RX_SYNC         , SAI1       , 0x401F85A4U, 0x1U), // SAI1_RX_SYNC
#endif
  //( 4, CSI     ,  0, DATA7           , CSI        , 0x401F8414U, 0x0U), // CSI_DATA7
  AF( 5, GPIO    ,  1, PIN26           , GPIO1      , 0          , 0   ), // GPIO1_PIN26
  //( 6, USDHC   ,  2, WP              , USDHC2     , 0x401F8608U, 0x1U), // USDHC2_WP
  //( 7, KPP     ,  0, ROW2            , KPP        , 0          , 0   ), // KPP_ROW2
  //(15, AD      ,  0, B1_10           , AD         ,            ,     ), // AD_B1_10
};

const pin_obj_t pin_AD_B1_10 = PIN(AD_B1_10, A0, 1, 26, pin_AD_B1_10_af, PIN_ADC1, 15, 0x401F8124U, 0x401F8314U);

const pin_af_obj_t pin_AD_B1_11_af[] = {
  //( 0, FLEXSPI ,  0, A_DATA2         , FLEXSPI    , 0x401F84B0U, 0x1U), // FLEXSPI_A_DATA2
  //( 1, EWM     ,  0, EWM_OUT_B       , EWM        , 0          , 0   ), // EWM_EWM_OUT_B
  AF( 2, LPUART  ,  8, RX              , LPUART8    , 0x401F8560U, 0x1U), // LPUART8_RX
#if (defined(MICROPY_HW_ENABLE_SAI1) && MICROPY_HW_ENABLE_SAI1)
  AF( 3, SAI     ,  1, RX_BCLK         , SAI1       , 0x401F8590U, 0x1U), // SAI1_RX_BCLK
#endif
  //( 4, CSI     ,  0, DATA6           , CSI        , 0x401F8410U, 0x0U), // CSI_DATA6
  AF( 5, GPIO    ,  1, PIN27           , GPIO1      , 0          , 0   ), // GPIO1_PIN27
  //( 6, USDHC   ,  2, RESET_B         , USDHC2     , 0          , 0   ), // USDHC2_RESET_B
  //( 7, KPP     ,  0, COL2            , KPP        , 0          , 0   ), // KPP_COL2
  //(15, AD      ,  0, B1_11           , AD         ,            ,     ), // AD_B1_11
};

const pin_obj_t pin_AD_B1_11 = PIN(AD_B1_11, A1, 1, 27, pin_AD_B1_11_af, PIN_ADC1, 0, 0x401F8128U, 0x401F8318U);

const pin_af_obj_t pin_SD_B0_00_af[] = {
  //( 0, USDHC   ,  1, CMD             , USDHC1     , 0          , 0   ), // USDHC1_CMD
  AF( 1, PWM     ,  1, PWMA0           , PWM1       , 0x401F8458U, 0x1U), // FLEXPWM1_PWMA0
  AF( 2, LPI2C   ,  3, SCL             , LPI2C3     , 0x401F84DCU, 0x1U), // LPI2C3_SCL
  //( 3, XBAR    ,  1, XBAR_INOUT4     , XBAR1      , 0x401F8614U, 0x1U), // XBAR1_XBAR_INOUT4
  AF( 4, LPSPI   ,  1, SCK             , LPSPI1     , 0x401F84F0U, 0x1U), // LPSPI1_SCK
  AF( 5, GPIO    ,  3, PIN12           , GPIO3      , 0          , 0   ), // GPIO3_PIN12
  //( 6, FLEXSPI ,  0, A_SS1_B         , FLEXSPI    , 0          , 0   ), // FLEXSPI_A_SS1_B
  //( 7, TPSMP   ,  0, HDATA30         , TPSMP      ,            ,     ), // TPSMP_HDATA30
  //(15, SD      ,  0, B0_00           , SD         ,            ,     ), // SD_B0_00
};

const pin_obj_t pin_SD_B0_00 = PIN(SD_B0_00, D13_CK, 3, 12, pin_SD_B0_00_af, 0, 0, 0x401F81BCU, 0x401F83ACU);

const pin_af_obj_t pin_SD_B0_01_af[] = {
  //( 0, USDHC   ,  1, CLK             , USDHC1     , 0          , 0   ), // USDHC1_CLK
  AF( 1, PWM     ,  1, PWMB0           , PWM1       , 0x401F8468U, 0x1U), // FLEXPWM1_PWMB0
  AF( 2, LPI2C   ,  3, SDA             , LPI2C3     , 0x401F84E0U, 0x1U), // LPI2C3_SDA
  //( 3, XBAR    ,  1, XBAR_INOUT5     , XBAR1      , 0x401F8618U, 0x1U), // XBAR1_XBAR_INOUT5
  AF( 4, LPSPI   ,  1, PCS0            , LPSPI1     , 0x401F84ECU, 0x0U), // LPSPI1_PCS0
  AF( 5, GPIO    ,  3, PIN13           , GPIO3      , 0          , 0   ), // GPIO3_PIN13
  //( 6, FLEXSPI ,  0, B_SS1_B         , FLEXSPI    , 0          , 0   ), // FLEXSPI_B_SS1_B
  //( 7, TPSMP   ,  0, HDATA31         , TPSMP      ,            ,     ), // TPSMP_HDATA31
  //(15, SD      ,  0, B0_01           , SD         ,            ,     ), // SD_B0_01
};

const pin_obj_t pin_SD_B0_01 = PIN(SD_B0_01, D10_SS, 3, 13, pin_SD_B0_01_af, 0, 0, 0x401F81C0U, 0x401F83B0U);

const pin_af_obj_t pin_SD_B0_02_af[] = {
  //( 0, USDHC   ,  1, DATA0           , USDHC1     , 0          , 0   ), // USDHC1_DATA0
  AF( 1, PWM     ,  1, PWMA1           , PWM1       , 0x401F845CU, 0x1U), // FLEXPWM1_PWMA1
  AF( 2, LPUART  ,  8, CTS_B           , LPUART8    , 0          , 0   ), // LPUART8_CTS_B
  //( 3, XBAR    ,  1, XBAR_INOUT6     , XBAR1      , 0x401F861CU, 0x1U), // XBAR1_XBAR_INOUT6
  AF( 4, LPSPI   ,  1, SDO             , LPSPI1     , 0x401F84F8U, 0x1U), // LPSPI1_SDO
  AF( 5, GPIO    ,  3, PIN14           , GPIO3      , 0          , 0   ), // GPIO3_PIN14
  //( 6, CCM     ,  0, OUT0            , CCM        ,            ,     ), // CCM_OUT0
  //( 7, OBSERVE ,  0, MUX_OUT0        , OBSERVE    ,            ,     ), // OBSERVE_MUX_OUT0
  //(15, SD      ,  0, B0_02           , SD         ,            ,     ), // SD_B0_02
};

const pin_obj_t pin_SD_B0_02 = PIN(SD_B0_02, D11_MO, 3, 14, pin_SD_B0_02_af, 0, 0, 0x401F81C4U, 0x401F83B4U);

const pin_af_obj_t pin_SD_B0_03_af[] = {
  //( 0, USDHC   ,  1, DATA1           , USDHC1     , 0          , 0   ), // USDHC1_DATA1
  AF( 1, PWM     ,  1, PWMB1           , PWM1       , 0x401F846CU, 0x1U), // FLEXPWM1_PWMB1
  AF( 2, LPUART  ,  8, RTS_B           , LPUART8    , 0          , 0   ), // LPUART8_RTS_B
  //( 3, XBAR    ,  1, XBAR_INOUT7     , XBAR1      , 0x401F8620U, 0x1U), // XBAR1_XBAR_INOUT7
  AF( 4, LPSPI   ,  1, SDI             , LPSPI1     , 0x401F84F4U, 0x1U), // LPSPI1_SDI
  AF( 5, GPIO    ,  3, PIN15           , GPIO3      , 0          , 0   ), // GPIO3_PIN15
  //( 6, CCM     ,  0, OUT1            , CCM        ,            ,     ), // CCM_OUT1
  //( 7, OBSERVE ,  0, MUX_OUT1        , OBSERVE    ,            ,     ), // OBSERVE_MUX_OUT1
  //(15, SD      ,  0, B0_03           , SD         ,            ,     ), // SD_B0_03
};

const pin_obj_t pin_SD_B0_03 = PIN(SD_B0_03, D12_MI, 3, 15, pin_SD_B0_03_af, 0, 0, 0x401F81C8U, 0x401F83B8U);

const pin_af_obj_t pin_WAKEUP_af[] = {
  AF( 5, GPIO    ,  5, PIN0            , GPIO5      , 0          , 0   ), // GPIO5_PIN0
};

const pin_obj_t pin_WAKEUP = PIN(WAKEUP, KEY, 5, 0, pin_WAKEUP_af, 0, 0, 0x400A8000U, 0x400A8018U);

STATIC const mp_rom_map_elem_t pin_cpu_pins_locals_dict_table[] = {
  { MP_ROM_QSTR(MP_QSTR_AD_B0_01), MP_ROM_PTR(&pin_AD_B0_01) },
  { MP_ROM_QSTR(MP_QSTR_AD_B0_02), MP_ROM_PTR(&pin_AD_B0_02) },
  { MP_ROM_QSTR(MP_QSTR_AD_B0_03), MP_ROM_PTR(&pin_AD_B0_03) },
  { MP_ROM_QSTR(MP_QSTR_AD_B0_09), MP_ROM_PTR(&pin_AD_B0_09) },
  { MP_ROM_QSTR(MP_QSTR_AD_B0_10), MP_ROM_PTR(&pin_AD_B0_10) },
  { MP_ROM_QSTR(MP_QSTR_AD_B0_11), MP_ROM_PTR(&pin_AD_B0_11) },
  { MP_ROM_QSTR(MP_QSTR_AD_B0_12), MP_ROM_PTR(&pin_AD_B0_12) },
  { MP_ROM_QSTR(MP_QSTR_AD_B0_13), MP_ROM_PTR(&pin_AD_B0_13) },
  { MP_ROM_QSTR(MP_QSTR_AD_B1_00), MP_ROM_PTR(&pin_AD_B1_00) },
  { MP_ROM_QSTR(MP_QSTR_AD_B1_01), MP_ROM_PTR(&pin_AD_B1_01) },
  { MP_ROM_QSTR(MP_QSTR_AD_B1_02), MP_ROM_PTR(&pin_AD_B1_02) },
  { MP_ROM_QSTR(MP_QSTR_AD_B1_03), MP_ROM_PTR(&pin_AD_B1_03) },
  { MP_ROM_QSTR(MP_QSTR_AD_B1_04), MP_ROM_PTR(&pin_AD_B1_04) },
  { MP_ROM_QSTR(MP_QSTR_AD_B1_05), MP_ROM_PTR(&pin_AD_B1_05) },
  { MP_ROM_QSTR(MP_QSTR_AD_B1_06), MP_ROM_PTR(&pin_AD_B1_06) },
  { MP_ROM_QSTR(MP_QSTR_AD_B1_07), MP_ROM_PTR(&pin_AD_B1_07) },
  { MP_ROM_QSTR(MP_QSTR_AD_B1_08), MP_ROM_PTR(&pin_AD_B1_08) },
  { MP_ROM_QSTR(MP_QSTR_AD_B1_10), MP_ROM_PTR(&pin_AD_B1_10) },
  { MP_ROM_QSTR(MP_QSTR_AD_B1_11), MP_ROM_PTR(&pin_AD_B1_11) },
  { MP_ROM_QSTR(MP_QSTR_SD_B0_00), MP_ROM_PTR(&pin_SD_B0_00) },
  { MP_ROM_QSTR(MP_QSTR_SD_B0_01), MP_ROM_PTR(&pin_SD_B0_01) },
  { MP_ROM_QSTR(MP_QSTR_SD_B0_02), MP_ROM_PTR(&pin_SD_B0_02) },
  { MP_ROM_QSTR(MP_QSTR_SD_B0_03), MP_ROM_PTR(&pin_SD_B0_03) },
  { MP_ROM_QSTR(MP_QSTR_WAKEUP), MP_ROM_PTR(&pin_WAKEUP) },
};
MP_DEFINE_CONST_DICT(pin_cpu_pins_locals_dict, pin_cpu_pins_locals_dict_table);

STATIC const mp_rom_map_elem_t pin_board_pins_locals_dict_table[] = {
  { MP_ROM_QSTR(MP_QSTR_PWM), MP_ROM_PTR(&pin_SD_B0_00) },
  { MP_ROM_QSTR(MP_QSTR_D0_RX), MP_ROM_PTR(&pin_AD_B1_07) },
  { MP_ROM_QSTR(MP_QSTR_D1_TX), MP_ROM_PTR(&pin_AD_B1_06) },
  { MP_ROM_QSTR(MP_QSTR_D2_INT0), MP_ROM_PTR(&pin_AD_B0_11) },
  { MP_ROM_QSTR(MP_QSTR_D3_INT1), MP_ROM_PTR(&pin_AD_B1_08) },
  { MP_ROM_QSTR(MP_QSTR_D4), MP_ROM_PTR(&pin_AD_B0_09) },
  { MP_ROM_QSTR(MP_QSTR_D5_PWM1), MP_ROM_PTR(&pin_AD_B0_10) },
  { MP_ROM_QSTR(MP_QSTR_D6_PWM2), MP_ROM_PTR(&pin_AD_B1_02) },
  { MP_ROM_QSTR(MP_QSTR_D7_PWM3), MP_ROM_PTR(&pin_AD_B1_03) },
  { MP_ROM_QSTR(MP_QSTR_D8), MP_ROM_PTR(&pin_AD_B0_03) },
  { MP_ROM_QSTR(MP_QSTR_D9_PWM4), MP_ROM_PTR(&pin_AD_B0_02) },
  { MP_ROM_QSTR(MP_QSTR_D10_SS), MP_ROM_PTR(&pin_SD_B0_01) },
  { MP_ROM_QSTR(MP_QSTR_D11_MO), MP_ROM_PTR(&pin_SD_B0_02) },
  { MP_ROM_QSTR(MP_QSTR_D12_MI), MP_ROM_PTR(&pin_SD_B0_03) },
  { MP_ROM_QSTR(MP_QSTR_D13_CK), MP_ROM_PTR(&pin_SD_B0_00) },
  { MP_ROM_QSTR(MP_QSTR_D14_SDA), MP_ROM_PTR(&pin_AD_B0_01) },
  { MP_ROM_QSTR(MP_QSTR_D15_SCL), MP_ROM_PTR(&pin_AD_B0_02) },
  { MP_ROM_QSTR(MP_QSTR_A0), MP_ROM_PTR(&pin_AD_B1_10) },
  { MP_ROM_QSTR(MP_QSTR_A1), MP_ROM_PTR(&pin_AD_B1_11) },
  { MP_ROM_QSTR(MP_QSTR_A2), MP_ROM_PTR(&pin_AD_B1_04) },
  { MP_ROM_QSTR(MP_QSTR_A3), MP_ROM_PTR(&pin_AD_B1_05) },
  { MP_ROM_QSTR(MP_QSTR_A4), MP_ROM_PTR(&pin_AD_B1_01) },
  { MP_ROM_QSTR(MP_QSTR_A5), MP_ROM_PTR(&pin_AD_B1_00) },
  { MP_ROM_QSTR(MP_QSTR_LED), MP_ROM_PTR(&pin_AD_B0_09) },
  { MP_ROM_QSTR(MP_QSTR_KEY), MP_ROM_PTR(&pin_WAKEUP) },
  { MP_ROM_QSTR(MP_QSTR_DBG_RXD), MP_ROM_PTR(&pin_AD_B0_13) },
  { MP_ROM_QSTR(MP_QSTR_DBG_TXD), MP_ROM_PTR(&pin_AD_B0_12) },
};
MP_DEFINE_CONST_DICT(pin_board_pins_locals_dict, pin_board_pins_locals_dict_table);

const pin_obj_t * const pin_adc1[] = {
  &pin_AD_B1_11, // 0
  &pin_AD_B0_12, // 1
  &pin_AD_B0_13, // 2
  NULL,    // 3
  NULL,    // 4
  &pin_AD_B1_00, // 5
  &pin_AD_B1_01, // 6
  &pin_AD_B1_02, // 7
  &pin_AD_B1_03, // 8
  &pin_AD_B1_04, // 9
  &pin_AD_B1_05, // 10
  &pin_AD_B1_06, // 11
  &pin_AD_B1_07, // 12
  &pin_AD_B1_08, // 13
  NULL,    // 14
  &pin_AD_B1_10, // 15
  NULL,    // 16
};

const pin_obj_t * const pin_adc2[] = {
  NULL,    // 0
  NULL,    // 1
  NULL,    // 2
  NULL,    // 3
  NULL,    // 4
  NULL,    // 5
  NULL,    // 6
  NULL,    // 7
  NULL,    // 8
  NULL,    // 9
  NULL,    // 10
  NULL,    // 11
  NULL,    // 12
  NULL,    // 13
  NULL,    // 14
  NULL,    // 15
  NULL,    // 16
};
