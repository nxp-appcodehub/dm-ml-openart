/*
 * This file is part of the MicroPython project, http://micropython.org/
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2020 Philipp Ebensberger
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef MICROPY_INCLUDED_MIMXRT_PIN_H
#define MICROPY_INCLUDED_MIMXRT_PIN_H

#include <stdint.h>
#include "py/obj.h"

#include "shared/runtime/mpirq.h"


// ------------------------------------------------------------------------------------------------------------------ //
enum {
    PORT_0 = 0,
    PORT_1,
    PORT_2,
    PORT_3,
    PORT_4,
    PORT_5,
    PORT_6,
    PORT_7,
    PORT_8,
    PORT_9,
};
enum {
    AF_FN_GPIO, 
    AF_FN_LPUART,
    AF_FN_LPI2C,
    AF_FN_LPSPI,
    AF_FN_SAI,
    AF_FN_I2S = AF_FN_SAI,
    AF_FN_GPT,
    AF_FN_TMR,
    AF_FN_PWM,
    AF_FN_SDMMC,
};

enum {
    AF_PIN_TYPE_GPIO_PIN0,
    AF_PIN_TYPE_GPIO_PIN1,
    AF_PIN_TYPE_GPIO_PIN2,
    AF_PIN_TYPE_GPIO_PIN3,
    AF_PIN_TYPE_GPIO_PIN4,
    AF_PIN_TYPE_GPIO_PIN5,
    AF_PIN_TYPE_GPIO_PIN6,
    AF_PIN_TYPE_GPIO_PIN7,
    AF_PIN_TYPE_GPIO_PIN8,
    AF_PIN_TYPE_GPIO_PIN9,
    AF_PIN_TYPE_GPIO_PIN10,
    AF_PIN_TYPE_GPIO_PIN11,
    AF_PIN_TYPE_GPIO_PIN12,
    AF_PIN_TYPE_GPIO_PIN13,
    AF_PIN_TYPE_GPIO_PIN14,
    AF_PIN_TYPE_GPIO_PIN15,  
    AF_PIN_TYPE_GPIO_PIN16,
    AF_PIN_TYPE_GPIO_PIN17,
    AF_PIN_TYPE_GPIO_PIN18,
    AF_PIN_TYPE_GPIO_PIN19,
    AF_PIN_TYPE_GPIO_PIN20,
    AF_PIN_TYPE_GPIO_PIN21,
    AF_PIN_TYPE_GPIO_PIN22,
    AF_PIN_TYPE_GPIO_PIN23,
    AF_PIN_TYPE_GPIO_PIN24,
    AF_PIN_TYPE_GPIO_PIN25,
    AF_PIN_TYPE_GPIO_PIN26,
    AF_PIN_TYPE_GPIO_PIN27,
    AF_PIN_TYPE_GPIO_PIN28,
    AF_PIN_TYPE_GPIO_PIN29,
    AF_PIN_TYPE_GPIO_PIN30,
    AF_PIN_TYPE_GPIO_PIN31,

    AF_PIN_TYPE_TMR_TIMER0 = 0,
    AF_PIN_TYPE_TMR_TIMER1,
    AF_PIN_TYPE_TMR_TIMER2,
    AF_PIN_TYPE_TMR_TIMER3,

    AF_PIN_TYPE_GPT_CLK,
    AF_PIN_TYPE_GPT_CAPTURE1,
    AF_PIN_TYPE_GPT_CAPTURE2,
    AF_PIN_TYPE_GPT_COMPARE1,
    AF_PIN_TYPE_GPT_COMPARE2,
    AF_PIN_TYPE_GPT_COMPARE3,

    AF_PIN_TYPE_LPI2C_SDA = 0,
    AF_PIN_TYPE_LPI2C_SCL,

    AF_PIN_TYPE_LPUART_TX = 0,
    AF_PIN_TYPE_LPUART_RX,
    AF_PIN_TYPE_LPUART_CTS_B,
    AF_PIN_TYPE_LPUART_RTS_B,
    AF_PIN_TYPE_LPUART_CK,

    AF_PIN_TYPE_LPSPI_SDI = 0,
    AF_PIN_TYPE_LPSPI_SDO,
    AF_PIN_TYPE_LPSPI_SCK,
    AF_PIN_TYPE_LPSPI_PCS0,

    AF_PIN_TYPE_PWM_PWMA0 = 0,
    AF_PIN_TYPE_PWM_PWMB0,
    AF_PIN_TYPE_PWM_PWMX0,
    AF_PIN_TYPE_PWM_PWMA1,
    AF_PIN_TYPE_PWM_PWMB1,
    AF_PIN_TYPE_PWM_PWMX1,
    AF_PIN_TYPE_PWM_PWMA2,
    AF_PIN_TYPE_PWM_PWMB2,
    AF_PIN_TYPE_PWM_PWMX2,
    AF_PIN_TYPE_PWM_PWMA3,
    AF_PIN_TYPE_PWM_PWMB3,
    AF_PIN_TYPE_PWM_PWMX3,


    AF_PIN_TYPE_SAI_MCLK = 0,
    AF_PIN_TYPE_SAI_TX_BCLK,
    AF_PIN_TYPE_SAI_TX_SYNC,
    AF_PIN_TYPE_SAI_TX_DATA,
    AF_PIN_TYPE_SAI_TX_DATA0 = AF_PIN_TYPE_SAI_TX_DATA,
    AF_PIN_TYPE_SAI_TX_DATA1,
    AF_PIN_TYPE_SAI_TX_DATA2,  
    AF_PIN_TYPE_SAI_TX_DATA3,
    AF_PIN_TYPE_SAI_RX_BCLK,
    AF_PIN_TYPE_SAI_RX_SYNC,
    AF_PIN_TYPE_SAI_RX_DATA,
    AF_PIN_TYPE_SAI_RX_DATA0 = AF_PIN_TYPE_SAI_RX_DATA,
};

enum {
    PIN_ADC1  = (1 << 1),
    PIN_ADC2  = (1 << 2),
    PIN_ADC3  = (1 << 3),
};
#define IOPAD_OUT_SLEW_FAST		(1<<0)

#define IOPAD_OUT_STRENGTH_0X  (0<<3)  // shut down
#define IOPAD_OUT_STRENGTH_1X  (1<<3)  // weakest
#define IOPAD_OUT_STRENGTH_2X  (2<<3)
#define IOPAD_OUT_STRENGTH_3X  (2<<3) 	
#define IOPAD_OUT_STRENGTH_4X  (4<<3) 
#define IOPAD_OUT_STRENGTH_5X  (5<<3)
#define IOPAD_OUT_STRENGTH_6X  (6<<3) 	
#define IOPAD_OUT_STRENGTH_7X  (7<<3)  // strongest


#define IOPAD_OUT_SPEED_50M		(0<<6)
#define IOPAD_OUT_SPEED_100M	(2<<6)
#define IOPAD_OUT_SPEED_200M	(3<<6)

#define IOPAD_OUT_OD			(IOPAD_OUT_STRENGTH_3X | 1<<11)

#define IOPAD_IN_HIZ			(0<<12)

#define IOPAD_IN_KEEPER			(1<<12 | 0<<13)	// keeper = keep previous (digital) input 

#define IOPAD_IN_PDN_100K		(1<<12 | 1<<13 | 0<<14)
#define IOPAD_IN_PUP_47K		(1<<12 | 1<<13 | 1<<14)
#define IOPAD_IN_PUP_100K		(1<<12 | 1<<13 | 2<<14)
#define IOPAD_IN_PUP_22K		(1<<12 | 1<<13 | 3<<14)
#define IOPAD_IN_HYST			(1<<16)

#define GPIO_MUX_INPUT_FORCE_ON	(1<<4)
#define GPIO_PAD_OUTPUT_MASK	(1<<27)	// if MSB is 1, then this PAD is configured as output (not in reg, just use it in S/W)

#define GPIO_MODE_IN                           ((uint32_t)0x00000000)   /*!< Input Floating Mode                   */
#define GPIO_MODE_OUT_PP                       ((uint32_t)0x00000001)   /*!< Output Push Pull Mode                 */
#define GPIO_MODE_OUT_OD                       ((uint32_t)0x00000011)   /*!< Output Open Drain Mode                */
#define GPIO_MODE_AF_PP                        ((uint32_t)0x00000002)   /*!< Alternate Function Push Pull Mode     */
#define GPIO_MODE_AF_OD                        ((uint32_t)0x00000012)   /*!< Alternate Function Open Drain Mode    */
#define GPIO_MODE_ANALOG                       ((uint32_t)0x00000003)   /*!< Analog Mode  */
#define GPIO_NOPULL                            ((uint32_t)0x00000000)   /*!< No Pull-up or Pull-down activation  */
#define GPIO_PULLUP                            ((uint32_t)0x00000001)   /*!< Pull-up activation                  */
#define GPIO_PULLDOWN                          ((uint32_t)0x00000002)   /*!< Pull-down activation                */

#define  GPIO_MODE_INPUT			(0<<27 | IOPAD_OUT_STRENGTH_0X | IOPAD_OUT_SPEED_50M | IOPAD_IN_KEEPER)
#define	 GPIO_MODE_INPUT_PUP		(0<<27 | IOPAD_OUT_STRENGTH_0X | IOPAD_OUT_SPEED_50M | IOPAD_IN_PUP_22K)
#define	 GPIO_MODE_INPUT_PUP_WEAK	(0<<27 | IOPAD_OUT_STRENGTH_0X | IOPAD_OUT_SPEED_50M | IOPAD_IN_PUP_47K)
#define	 GPIO_MODE_INPUT_PDN		(0<<27 | IOPAD_OUT_STRENGTH_0X | IOPAD_OUT_SPEED_50M | IOPAD_IN_PDN_100K)
#define  GPIO_MODE_OUTPUT_PP        (GPIO_PAD_OUTPUT_MASK | IOPAD_OUT_STRENGTH_6X | IOPAD_OUT_SPEED_100M | IOPAD_IN_KEEPER)
#define  GPIO_MODE_OUTPUT_PP_WEAK   (GPIO_PAD_OUTPUT_MASK | IOPAD_OUT_STRENGTH_1X | IOPAD_OUT_SPEED_100M | IOPAD_IN_KEEPER)


#define  GPIO_MODE_OUTPUT_OD_HIZ 	(GPIO_PAD_OUTPUT_MASK | IOPAD_OUT_OD)
#define  GPIO_MODE_OUTPUT_OD		(GPIO_PAD_OUTPUT_MASK | IOPAD_OUT_OD | IOPAD_IN_KEEPER)
#define  GPIO_MODE_OUTPUT_OD_PUP    (GPIO_PAD_OUTPUT_MASK | IOPAD_OUT_OD | IOPAD_IN_PUP_22K)
#define GPIO_MODE_OUTPUT_OD_KEEP			GPIO_MODE_OUTPUT_OD

#define  GPIO_MODE_IT_RISING                    ((uint32_t)0x10110000U)   /*!< External Interrupt Mode with Rising edge trigger detection          */
#define  GPIO_MODE_IT_FALLING                   ((uint32_t)0x10210000U)   /*!< External Interrupt Mode with Falling edge trigger detection         */
#define  GPIO_MODE_IT_RISING_FALLING            ((uint32_t)0x10310000U)   /*!< External Interrupt Mode with Rising/Falling edge trigger detection  */
 
#define  GPIO_MODE_EVT_RISING                   ((uint32_t)0x10120000U)   /*!< External Event Mode with Rising edge trigger detection               */
#define  GPIO_MODE_EVT_FALLING                  ((uint32_t)0x10220000U)   /*!< External Event Mode with Falling edge trigger detection              */
#define  GPIO_MODE_EVT_RISING_FALLING           ((uint32_t)0x10320000U)   /*!< External Event Mode with Rising/Falling edge trigger detection       */

// >> IOCON settings
#define  GPIO_INVERTER		(1<<7)

// ------------------------------------------------------------------------------------------------------------------ //

typedef struct {
    mp_obj_base_t base;
    qstr name;
    uint8_t idx;
    uint8_t fn;
    uint8_t unit;
    uint8_t af_pin_type;
    // >>> i.mx RT uses 2 levels of muxing (DAISYCHAIN, 'INPUT_SELECT' registers)
    uint32_t inSelReg;
    uint32_t inSelVal;
    // <<<

    union {
        void *reg;
        void *pQTmr;//TMR_Type *pQTmr;
        void *pI2C;//LPI2C_Type *pI2C;
        void *pUART;//LPUART_Type *pUART;
        void *pSPI;//LPSPI_Type *pSPI;
    };
} pin_af_obj_t;

typedef struct {
    mp_obj_base_t base;
    qstr name;  // pad name
    qstr board_name;

    void *gpio;  // gpio instance for pin
    uint32_t port   : 4;
    uint32_t pin    : 5;      // Some ARM processors use 32 bits/PORT
    uint32_t num_af : 4;
    uint32_t adc_channel : 5; // Some ARM processors use 32 bits/PORT
    uint32_t adc_num  : 3;    // 1 bit per ADC
#ifdef SOC_IMXRT1170_SERIES
  uint32_t port_m4;
  uint32_t pin_m4;
#endif	
    uint32_t pin_mask;
    uint32_t afReg;
    uint32_t cfgReg;
    mp_obj_t pin_isr_cb;
    const pin_af_obj_t *af;
} pin_obj_t;

typedef struct {
    const char *name;
    const pin_obj_t *pin;
} pin_named_pin_t;

// ------------------------------------------------------------------------------------------------------------------ //

extern const mp_obj_type_t mpy_pin_type;
extern const mp_obj_type_t pin_af_type;

// ------------------------------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------------------------------ //

void pin_init(void);
uint32_t pin_get_mode(const pin_obj_t *pin);
uint32_t pin_get_af(const pin_obj_t *pin);
const pin_obj_t *pin_find(mp_obj_t user_obj);
const pin_obj_t *pin_find_named_pin(const mp_obj_dict_t *named_pins, mp_obj_t name);
const pin_af_obj_t *pin_find_af(const pin_obj_t *pin, uint8_t fn);
const pin_obj_t *pin_find_af_by_index(const pin_obj_t *pin, mp_uint_t af_idx);
const pin_obj_t *pin_find_af_by_name(const pin_obj_t *pin, const char *name);
void machine_pin_set_mode(const pin_obj_t *pin, uint8_t mode);

#endif // MICROPY_INCLUDED_MIMXRT_PIN_H
