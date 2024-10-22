/*
 * This file is part of the MicroPython project, http://micropython.org/
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2018 SummerGift <zhangyuan@rt-thread.com>
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

#include <stdio.h>
#include <string.h>

#include "py/runtime.h"
#include "py/mphal.h"
#include "extmod/machine_spi.h"
#include "aia_cmm/cfg_mux_mgr.h"
#include "drv_gpio.h"
#ifdef MICROPYTHON_USING_MACHINE_SPI

#ifndef RT_USING_SPI
#error "Please define the RT_USING_SPI on 'rtconfig.h'"
#endif

const mp_obj_type_t machine_hard_spi_type;

typedef struct _machine_hard_spi_obj_t {
    mp_obj_base_t base;
    struct rt_spi_device *spi_device;
} machine_hard_spi_obj_t;

STATIC void machine_hard_spi_print(const mp_print_t *print, mp_obj_t self_in, mp_print_kind_t kind) {
    machine_hard_spi_obj_t *self = (machine_hard_spi_obj_t*)self_in;
    mp_printf(print,"SPI(device port : %s)",self->spi_device->parent.parent.name);
}

mp_obj_t machine_hard_spi_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *all_args) {
    char spi_dev_name[RT_NAME_MAX];
	char spi_bus_name[RT_NAME_MAX];
	
	mp_arg_check_num(n_args, n_kw, 2, 2, false);
	
	int bus = mp_obj_get_int(all_args[0]);
    snprintf(spi_bus_name, sizeof(spi_bus_name), "spi%d", bus);
	
    int ndx = mp_obj_get_int(all_args[0]);
    snprintf(spi_dev_name, sizeof(spi_dev_name), "spi%d%d", bus,ndx);
	
	
    

    // create new hard SPI object
    machine_hard_spi_obj_t *self = m_new_obj(machine_hard_spi_obj_t);
    self->base.type = &machine_hard_spi_type;
    
		
	MuxItem_t mux_pSCK;
    MuxItem_t mux_pMOSI;
	MuxItem_t mux_pMISO;
	MuxItem_t mux_pCS;
	//ndx = ndx /10;// idx 10, 20 30 ,40 defined in drv_spi.c 
	Mux_Take(self->spi_device, "spi", ndx, "SCK", &mux_pSCK);
	if (mux_pSCK.pPinObj == 0)
	{
		mp_printf(&mp_plat_print, "ERROR: SCK not found!\n", spi_dev_name);
        nlr_raise(mp_obj_new_exception_msg_varg(&mp_type_ValueError, "SPI(%s) SCK doesn't exist", spi_dev_name));
	}
	mp_hal_pin_config_alt(mux_pSCK.pPinObj, GPIO_MODE_OUTPUT_PP, AF_FN_LPSPI);//here the GPIO_MODE_OUTPUT_PP is the var in the function IOMUXC_SetPinConfig the last var configvalue
    
	Mux_Take(self->spi_device, "spi", ndx, "SDO", &mux_pMOSI);
	if (mux_pMOSI.pPinObj == 0)
	{
		mp_printf(&mp_plat_print, "ERROR: SDO not found!\n", spi_dev_name);
        nlr_raise(mp_obj_new_exception_msg_varg(&mp_type_ValueError, "SPI(%s) SDO doesn't exist", spi_dev_name));
	}
	mp_hal_pin_config_alt(mux_pMOSI.pPinObj, GPIO_MODE_OUTPUT_PP, AF_FN_LPSPI);
	
    Mux_Take(self->spi_device, "spi", ndx, "SDI", &mux_pMISO);
	if (mux_pMISO.pPinObj == 0)
	{
		mp_printf(&mp_plat_print, "ERROR: SDI not found!\n", spi_dev_name);
        nlr_raise(mp_obj_new_exception_msg_varg(&mp_type_ValueError, "SPI(%s) SDI doesn't exist", spi_dev_name));
	}
	mp_hal_pin_config_alt(mux_pMISO.pPinObj,GPIO_MODE_INPUT, AF_FN_LPSPI);//the before one is MP_HAL_PIN_PULL_UP
	
	Mux_Take(self->spi_device, "spi", ndx, "PCS0", &mux_pCS);
	if (mux_pCS.pPinObj == 0)
	{
		mp_printf(&mp_plat_print, "ERROR: PCS0 not found!\n", spi_dev_name);
        nlr_raise(mp_obj_new_exception_msg_varg(&mp_type_ValueError, "SPI(%s) PCS0 doesn't exist", spi_dev_name));
	}
	mp_hal_pin_config_alt(mux_pCS.pPinObj,GPIO_MODE_OUTPUT_PP, AF_FN_GPIO);//the before one is MP_HAL_PIN_PULL_UP
	rt_hw_spi_device_attach(spi_bus_name,spi_dev_name,GET_PIN(mux_pCS.pPinObj->port,mux_pCS.pPinObj->pin));
	
	struct rt_spi_device *rt_spi_device = (struct rt_spi_device *) rt_device_find(spi_dev_name);
    if (rt_spi_device == RT_NULL || rt_spi_device->parent.type != RT_Device_Class_SPIDevice) {
        mp_printf(&mp_plat_print, "ERROR: SPI device %s not found!\n", spi_dev_name);
        nlr_raise(mp_obj_new_exception_msg_varg(&mp_type_ValueError, "SPI(%s) doesn't exist", spi_dev_name));
    }
	self->spi_device = rt_spi_device;
	
	struct rt_spi_configuration cfg;
    cfg.data_width = 8;
    cfg.mode = RT_SPI_MASTER | RT_SPI_MODE_0 | RT_SPI_MSB ;
    cfg.max_hz = 20000;
    rt_spi_configure(self->spi_device, &cfg);
    return (mp_obj_t) self;
}

//SPI.init( baudrate=100000, polarity=0, phase=0, bits=8, firstbit=SPI.MSB/LSB )
STATIC void machine_hard_spi_init(mp_obj_base_t *self_in, size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    machine_hard_spi_obj_t *self = (machine_hard_spi_obj_t*)self_in;
    rt_uint8_t mode = 0;
    int baudrate = mp_obj_get_int(pos_args[0]);
    int polarity = mp_obj_get_int(pos_args[1]);
    int phase = mp_obj_get_int(pos_args[2]);
    int bits = mp_obj_get_int(pos_args[3]);
    int firstbit = mp_obj_get_int(pos_args[4]);

    if(!polarity && !phase)
    {
        mode = RT_SPI_MODE_0;
    }

    if(!polarity && phase)
    {
        mode = RT_SPI_MODE_1;
    }

    if(polarity && !phase)
    {
        mode = RT_SPI_MODE_2;
    }

    if(polarity && phase)
    {
        mode = RT_SPI_MODE_3;
    }

    if(firstbit)
    {
        mode |= RT_SPI_MSB;
    } else {
        mode |= RT_SPI_LSB;
    }

    /* config spi */
    {
        struct rt_spi_configuration cfg;
        cfg.data_width = bits;
        cfg.mode = mode;
        cfg.max_hz = baudrate;
        rt_spi_configure(self->spi_device, &cfg);
    }
}

STATIC void machine_hard_spi_transfer(mp_obj_base_t *self_in, size_t len, const uint8_t *src, uint8_t *dest) {
    machine_hard_spi_obj_t *self = (machine_hard_spi_obj_t*)self_in;

    if (src && dest) {
        rt_spi_send_then_recv(self->spi_device, src, len, dest, len);
    } else if (src) {
        rt_spi_send(self->spi_device, src, len);
    } else {
        rt_spi_recv(self->spi_device, dest, len);
    }
}

STATIC const mp_machine_spi_p_t machine_hard_spi_p = {
    .init = machine_hard_spi_init,
    .deinit = NULL,
    .transfer = machine_hard_spi_transfer,
};

const mp_obj_type_t machine_hard_spi_type = {
    { &mp_type_type },
    .name = MP_QSTR_SPI,
    .print = machine_hard_spi_print,
    .make_new = machine_hard_spi_make_new,
    .protocol = &machine_hard_spi_p,
    .locals_dict = (mp_obj_t)&mp_machine_spi_locals_dict,
};

#endif // MICROPYTHON_USING_MACHINE_SPI

