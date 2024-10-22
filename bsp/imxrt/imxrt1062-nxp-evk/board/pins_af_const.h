    /*
 * Copyright (c) 2006-2018, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF0_LPI2C4),  MP_OBJ_NEW_SMALL_INT(GPIO_AF0_LPI2C4) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF0_PWM2),    MP_OBJ_NEW_SMALL_INT(GPIO_AF0_PWM2) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF1_GPT1),    MP_OBJ_NEW_SMALL_INT(GPIO_AF1_GPT1) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF1_LPI2C3),  MP_OBJ_NEW_SMALL_INT(GPIO_AF1_LPI2C3) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF1_PWM1),    MP_OBJ_NEW_SMALL_INT(GPIO_AF1_PWM1) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF1_PWM2),    MP_OBJ_NEW_SMALL_INT(GPIO_AF1_PWM2) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF1_PWM4),    MP_OBJ_NEW_SMALL_INT(GPIO_AF1_PWM4) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF1_TMR3),    MP_OBJ_NEW_SMALL_INT(GPIO_AF1_TMR3) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF2_LPI2C3),  MP_OBJ_NEW_SMALL_INT(GPIO_AF2_LPI2C3) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF2_LPUART1), MP_OBJ_NEW_SMALL_INT(GPIO_AF2_LPUART1) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF2_LPUART2), MP_OBJ_NEW_SMALL_INT(GPIO_AF2_LPUART2) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF2_LPUART3), MP_OBJ_NEW_SMALL_INT(GPIO_AF2_LPUART3) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF2_LPUART6), MP_OBJ_NEW_SMALL_INT(GPIO_AF2_LPUART6) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF2_LPUART8), MP_OBJ_NEW_SMALL_INT(GPIO_AF2_LPUART8) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF3_LPI2C1),  MP_OBJ_NEW_SMALL_INT(GPIO_AF3_LPI2C1) },
#if (defined(MICROPY_HW_ENABLE_SAI1) && MICROPY_HW_ENABLE_SAI1)
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF3_SAI1),    MP_OBJ_NEW_SMALL_INT(GPIO_AF3_SAI1) },
#endif
#if (defined(MICROPY_HW_ENABLE_SAI2) && MICROPY_HW_ENABLE_SAI2)
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF3_SAI2),    MP_OBJ_NEW_SMALL_INT(GPIO_AF3_SAI2) },
#endif
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF4_LPSPI1),  MP_OBJ_NEW_SMALL_INT(GPIO_AF4_LPSPI1) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF4_PWM1),    MP_OBJ_NEW_SMALL_INT(GPIO_AF4_PWM1) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF5_GPIO1),   MP_OBJ_NEW_SMALL_INT(GPIO_AF5_GPIO1) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF5_GPIO3),   MP_OBJ_NEW_SMALL_INT(GPIO_AF5_GPIO3) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF5_GPIO5),   MP_OBJ_NEW_SMALL_INT(GPIO_AF5_GPIO5) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF7_GPT2),    MP_OBJ_NEW_SMALL_INT(GPIO_AF7_GPT2) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_AF7_LPSPI3),  MP_OBJ_NEW_SMALL_INT(GPIO_AF7_LPSPI3) },
