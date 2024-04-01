/*
 * Copyright (c) 2006-2018, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       		Notes
 * 2020-04-13     Tony.Zhang(NXP)	
 *
 */

#include <rtthread.h>
#include <rtdevice.h>
#ifdef SOC_IMXRT1170_SERIES
#else
#include "fsl_csi_camera_adapter.h"
#include "fsl_camera.h"
#include "fsl_camera_receiver.h"
#include "fsl_camera_device.h"
#include "fsl_flexio_camera.h"
#include "fsl_ov7725.h"
#endif
#include "fsl_elcdif.h"
#include "fsl_edma.h"
#include "fsl_dmamux.h"
#include "fsl_cache.h"
#include "fsl_gpio.h"
#include "fsl_clock.h"
#include "fsl_csi.h"
#include "py/mpprint.h"
#include "irq.h"
#include "sensor.h"
#include "drv_camera.h"
#include "ov9650.h"
#include "ov2640.h"
#include "ov5640.h"
#include "ov5640_regs.h"
#include "ov7725.h"
#include "ov7725_regs.h"
#include "mt9v034.h"
#include "framebuffer.h"


#define cam_echo(...) //mp_printf(MP_PYTHON_PRINTER, __VA_ARGS__)
#define cam_err(...) mp_printf(MP_PYTHON_PRINTER, __VA_ARGS__)

#if defined(__CC_ARM) || defined(__CLANG_ARM)
//	extern unsigned int Image$$MPY_SENSOR_BUFER_START$$Base;
//	extern unsigned int Image$$MPY_SENSOR_BUFER_START$$Limit;
#elif defined(__ICCARM__)
	extern unsigned int MPY_SENSOR_BUFER_START$$Limit[];
#elif defined(__GNUC__)

#endif


enum{
	RT_CAMERA_DEVICE_INIT = 0,
	RT_CAMERA_DEVICE_SUSPEND,
	RT_CAMERA_DEVICE_RUNING,
}RT_CAMERA_DEVICE_ST;


#define FB_MEM_HEAD_SIZE   (2*sizeof(uint32_t))

struct imxrt_camera
{
    char *name;
    
    
	struct rt_camera_device *rtt_device;

    uint32_t flexio_base ;
    FLEXIO_CAMERA_Type FlexioCameraDevice;
    flexio_camera_config_t FlexioCameraConfig;
    edma_config_t edma_config;
    edma_handle_t g_EDMA_Handle;
    edma_transfer_config_t transferConfig;
    IRQn_Type irqn;
	sensor_t	sensor;
    GPIO_Type * vSyncgpio;
	uint8_t     vSyncpin;
    rt_base_t    vSyncRTTpin;
	uint32_t LineCount;
	uint32_t LineBytes;
	
	
	uint16_t	sensor_id;
	uint8_t     sensor_addr;
	GPIO_Type * sensor_pwn_io;
	uint8_t     sensor_pwn_io_pin;
	GPIO_Type *  sensor_rst_io;
	uint8_t sensor_rst_io_pin;
	char 		*sensor_bus_name;
	uint32_t	fb_buffer_start;
	uint32_t	fb_buffer_end;
};

const uint16_t supported_sensors[3][2] = {
	{0x21,OV_CHIP_ID},//ov7725
	{0x48,0x00},//mt9m114
	{OV5640_SLV_ADDR>>1,OV5640_CHIP_ID}
};

static struct imxrt_camera *pCam = NULL;
#define CAM_NUM		1
#ifndef BSP_SENSOR2_BUS_NAME
#define BSP_SENSOR2_BUS_NAME "CAM2_I2C"
#endif
static struct imxrt_camera cams[1] = {
	{
		.name = "camera0",
		.sensor_id = 0,
		.sensor_addr = 0x42U,
		
		.sensor_bus_name = BSP_SENSOR2_BUS_NAME,
	}

};


const int resolution[][2] = {
    {0,    0   },
    // C/SIF Resolutions
    {88,   72  },    /* QQCIF     */
    {176,  144 },    /* QCIF      */
    {352,  288 },    /* CIF       */
    {88,   60  },    /* QQSIF     */
    {176,  120 },    /* QSIF      */
    {352,  240 },    /* SIF       */
    // VGA Resolutions
    {40,   30  },    /* QQQQVGA   */
    {80,   60  },    /* QQQVGA    */
    {160,  120 },    /* QQVGA     */
    {320,  240 },    /* QVGA      */
    {640,  480 },    /* VGA       */
    {60,   40  },    /* HQQQVGA   */
    {120,  80  },    /* HQQVGA    */
    {240,  160 },    /* HQVGA     */
    // FFT Resolutions
    {64,   32  },    /* 64x32     */
    {64,   64  },    /* 64x64     */
    {128,  64  },    /* 128x64    */
    {128,  128 },    /* 128x64    */
    // Other
    {128,  160 },    /* LCD       */
    {128,  160 },    /* QQVGA2    */
    {720,  480 },    /* WVGA      */
    {752,  480 },    /* WVGA2     */
    {800,  600 },    /* SVGA      */
    {1024, 768 },    /* XGA       */
    {1280, 1024},    /* SXGA      */
    {1600, 1200},    /* UXGA      */
	{1280, 720 },    /* HD        */ 
    {1920, 1080},    /* FHD       */ 
    {2560, 1440},    /* QHD       */
    {2048, 1536},    /* QXGA      */
    {2560, 1600},    /* WQXGA     */
    {2592, 1944},    /* WQXGA2    */
};

#define FLEXIO_CAMERA_DMA_CHN           0u
#define FLEXIO_CAMERA_DMA_MUX_SRC       (kDmaRequestMuxFlexIO1Request0Request1 & 0xFF)

#define FXIO_SHFT_COUNT     1u
#define DMA_CHN             0u
#define DMA_MUX_SRC         kDmaRequestMuxFlexIO1Request0Request1

#define FRAMEBUFFER_SIZE (640*480*2)
#define FRAMEBUFFER_COUNT 3
#define FRAGBUF_LOC  __attribute__((section(".dmaFramebuffer")))
extern char Image$$CAM_FRAME_BUFFER$$Base;
FRAGBUF_LOC static uint32_t *s_dmaFragBufs[2];	// max supported line length, XRGB8888(*4)
uint32_t pFlexioCameraFrameBuffer = 0;
static struct rt_event frame_event;
#define EVENT_CSI	(1<<0)

static uint32_t g_csi_int_en = 0;
static uint32_t isToStopCSICommand = 0;
static rt_timer_t g_csi_to = NULL;
static uint32_t g_csi_to_ticks = 1500;
bool csi_calc_first = 0;
//==================================================================================================
#ifdef RT_USING_LCD
extern void LCDMonitor_Update(uint32_t fbNdx, uint8_t isGray,uint32_t wndH, uint32_t wndW, uint32_t pixels_addr);
#endif

void imx_cam_csi_start_frame(struct imxrt_camera *cam);

void imx_cam_sensor_io_init(GPIO_Type *base, uint32_t pin)
{
    gpio_pin_config_t config = {
        kGPIO_DigitalOutput, 0,
    };
	
	GPIO_PinInit(base, pin, &config);	
}

static void imx_cam_sensor_io_set(GPIO_Type *base, uint32_t pin,bool pullUp)
{
    if (pullUp)
    {
        GPIO_PinWrite(base, pin, 1);
    }
    else
    {
        GPIO_PinWrite(base, pin, 0);
    }
}

uint8_t imx_cam_sensor_scan(struct rt_i2c_bus_device *i2c_bus)
{
	struct rt_i2c_msg msgs;
	rt_uint8_t buf[3] = {0};
	for (uint8_t addr=0x08;addr<=0x77;addr++)
	{
		buf[0] = 0x0; //cmd
		msgs.addr = addr;
		msgs.flags = RT_I2C_WR;
		msgs.buf = buf;
		msgs.len = 1;
		
		if (rt_i2c_transfer(i2c_bus, &msgs, 1) != 0)
		{
			return addr ;
		}
	}
	
	return 0;
}

int imx_cam_sensor_read_reg(struct rt_i2c_bus_device *i2c_bus,rt_uint16_t addr, rt_uint8_t reg, rt_uint8_t *data)
{
	struct rt_i2c_msg msgs[2];
	rt_uint8_t buf[3];
	
	
	buf[0] = reg; //cmd
    msgs[0].addr = addr;
    msgs[0].flags = RT_I2C_WR;
    msgs[0].buf = buf;
    msgs[0].len = 1;
	
	msgs[1].addr = addr;
    msgs[1].flags = RT_I2C_RD|RT_I2C_NO_START;
    msgs[1].buf = data;
    msgs[1].len = 1;
	
//	count = rt_i2c_master_send(i2c_bus,addr,0,buf,1);
//	count = rt_i2c_master_recv(i2c_bus,addr,RT_I2C_NO_START,data,1);
    
	
    if (rt_i2c_transfer(i2c_bus, msgs, 2) != 0)
    {
        return 0 ;
    }
    else
    {
		cam_err("[%s] error\r\n",__func__);
        return -1;
    }
	
}

int imx_cam_sensor_write_reg(struct rt_i2c_bus_device *i2c_bus,rt_uint16_t addr, rt_uint8_t reg, rt_uint8_t data)
{
	struct rt_i2c_msg msgs;
	rt_uint8_t buf[3];

	buf[0] = reg; //cmd
	buf[1] = data;

	msgs.addr = addr;
	msgs.flags = RT_I2C_WR;
	msgs.buf = buf;
	msgs.len = 2;

	
	if (rt_i2c_transfer(i2c_bus, &msgs, 1) == 1)
	{
		return 0;
	}
	else
	{
		cam_err("[%s] error\r\n",__func__);
		return -1;
	}
}

int imx_cam_sensor_readb2_reg(struct rt_i2c_bus_device *i2c_bus,rt_uint8_t addr, rt_uint16_t reg, rt_uint8_t *data)
{
	struct rt_i2c_msg msgs[2];
	rt_uint8_t buf[3];
	
	
	buf[0] = reg >> 8; //cmd
	buf[1] = reg & 0xff;
    msgs[0].addr = addr;
    msgs[0].flags = RT_I2C_WR;
    msgs[0].buf = buf;
    msgs[0].len = 2;
	
	msgs[1].addr = addr;
    msgs[1].flags = RT_I2C_RD|RT_I2C_NO_START;
    msgs[1].buf = data;
    msgs[1].len = 1;
		
    if (rt_i2c_transfer(i2c_bus, msgs, 2) != 0)
    {
        return 0 ;
    }
    else
    {
		cam_err("[%s] error\r\n",__func__);
        return -1;
    }
	
}

int imx_cam_sensor_writeb2_reg(struct rt_i2c_bus_device *i2c_bus,rt_uint16_t addr, rt_uint16_t reg, rt_uint8_t data)
{
	struct rt_i2c_msg msgs;
	rt_uint8_t buf[3];

	buf[0] = reg >> 8; //cmd
	buf[1] = reg & 0xff;
	buf[2] = data;

	msgs.addr = addr;
	msgs.flags = RT_I2C_WR;
	msgs.buf = buf;
	msgs.len = 3;

	
	if (rt_i2c_transfer(i2c_bus, &msgs, 1) == 1)
	{
		return 0;
	}
	else
	{
		cam_err("[%s] error\r\n",__func__);
		return -1;
	}
}

int imx_cam_sensor_cambus_writews(sensor_t *sensor, uint8_t slv_addr, uint16_t reg_addr, uint8_t *reg_data, uint8_t size)
{
	struct rt_i2c_msg msgs;
	rt_uint8_t *buf;

	struct rt_i2c_bus_device *i2c_bus = (struct rt_i2c_bus_device *)sensor->i2c_bus;
	buf = (rt_uint8_t*)rt_malloc(size+2);
	buf[0] = reg_addr & 0xff;
	buf[1] = reg_addr >> 8; //cmd
	
	memcpy(buf+2 , reg_data,size);

	msgs.addr = slv_addr;
	msgs.flags = RT_I2C_WR;
	msgs.buf = buf;
	msgs.len = 2+size;

	
	if (rt_i2c_transfer(i2c_bus, &msgs, 1) == 1)
	{
		rt_free(buf);
		return 0;
	}
	else
	{
		rt_free(buf);
		cam_err("[%s] error\r\n",__func__);
		return -1;
	}
}

int imx_cam_sensor_cambus_readws(sensor_t *sensor, uint8_t slv_addr, uint16_t reg_addr, uint8_t *reg_data, uint8_t size)
{
	struct rt_i2c_msg msgs[2];
	rt_uint8_t buf[4];
	struct rt_i2c_bus_device *i2c_bus = (struct rt_i2c_bus_device *)sensor->i2c_bus;
	
	buf[0] = reg_addr & 0x00ff; //cmd
	buf[1] = reg_addr >> 8; //cmd
    msgs[0].addr = slv_addr;
    msgs[0].flags = RT_I2C_WR;
    msgs[0].buf = buf;
    msgs[0].len = 2;
	
	msgs[1].addr = slv_addr;
    msgs[1].flags = RT_I2C_RD;
    msgs[1].buf = reg_data;
    msgs[1].len = size;
	
    
    if (rt_i2c_transfer(i2c_bus, msgs, 2) != 0)
    {
        return 0 ;
    }
    else
    {
		cam_err("[%s] error\r\n",__func__);
        return -1;
    }
}

int imx_cam_sensor_cambus_writeb(sensor_t *sensor, uint8_t slv_addr, uint8_t reg_addr, uint8_t reg_data)
{
	struct rt_i2c_bus_device *i2c_bus = (struct rt_i2c_bus_device *)sensor->i2c_bus;
	
	return imx_cam_sensor_write_reg(i2c_bus,slv_addr,reg_addr,reg_data);
}
int imx_cam_sensor_cambus_readb(sensor_t *sensor, uint8_t slv_addr, uint8_t reg_addr, uint8_t *reg_data)
{
	struct rt_i2c_bus_device *i2c_bus = (struct rt_i2c_bus_device *)sensor->i2c_bus;
	
	return imx_cam_sensor_read_reg(i2c_bus,slv_addr,reg_addr,reg_data);
}

int imx_cam_sensor_cambus_writeb2(sensor_t *sensor, uint8_t slv_addr, uint16_t reg_addr, uint8_t reg_data)
{
	struct rt_i2c_bus_device *i2c_bus = (struct rt_i2c_bus_device *)sensor->i2c_bus;
	
	return imx_cam_sensor_writeb2_reg(i2c_bus,slv_addr,reg_addr,reg_data);
}
int imx_cam_sensor_cambus_readb2(sensor_t *sensor, uint8_t slv_addr, uint16_t reg_addr, uint8_t *reg_data)
{
	struct rt_i2c_bus_device *i2c_bus = (struct rt_i2c_bus_device *)sensor->i2c_bus;
	
	return imx_cam_sensor_readb2_reg(i2c_bus,slv_addr,reg_addr,reg_data);
}

int imx_cam_sensor_cambus_readw(sensor_t *sensor, uint8_t slv_addr, uint8_t reg_addr, uint16_t *reg_data)
{
	struct rt_i2c_msg msgs[2];
	rt_uint8_t buf[3];
	struct rt_i2c_bus_device *i2c_bus = (struct rt_i2c_bus_device *)sensor->i2c_bus;
	
	buf[0] = reg_addr; //cmd
    msgs[0].addr = slv_addr;
    msgs[0].flags = RT_I2C_WR;
    msgs[0].buf = buf;
    msgs[0].len = 1;
	
	msgs[1].addr = reg_addr;
    msgs[1].flags = RT_I2C_RD;
    msgs[1].buf = (uint8_t *)reg_data;
    msgs[1].len = 1;
	
    
    if (rt_i2c_transfer(i2c_bus, msgs, 2) != 0)
    {
        return 0 ;
    }
    else
    {
		cam_err("[%s] error\r\n",__func__);
        return -1;
    }
}

int imx_cam_sensor_cambus_writew(sensor_t *sensor, uint8_t slv_addr, uint8_t reg_addr, uint16_t reg_data)
{
	struct rt_i2c_msg msgs;
	rt_uint8_t buf[3];
	struct rt_i2c_bus_device *i2c_bus = (struct rt_i2c_bus_device *)sensor->i2c_bus;
	
	buf[0] = reg_addr; //cmd
	buf[1] = reg_data & 0x00ff;
	buf[2] = reg_data >> 8;
	
	msgs.addr = slv_addr;
	msgs.flags = RT_I2C_WR;
	msgs.buf = buf;
	msgs.len = 3;

	
	if (rt_i2c_transfer(i2c_bus, &msgs, 1) == 1)
	{
		return 0;
	}
	else
	{
		cam_err("[%s] error\r\n",__func__);
		return -1;
	}
}

int imx_cam_sensor_cambus_readw2(sensor_t *sensor, uint8_t slv_addr, uint16_t reg_addr, uint16_t *reg_data)
{
	struct rt_i2c_msg msgs[2];
	rt_uint8_t buf[4];
	struct rt_i2c_bus_device *i2c_bus = (struct rt_i2c_bus_device *)sensor->i2c_bus;
	
	buf[0] = reg_addr & 0x00ff; //cmd
	buf[1] = reg_addr >> 8; //cmd
    msgs[0].addr = slv_addr;
    msgs[0].flags = RT_I2C_WR;
    msgs[0].buf = buf;
    msgs[0].len = 2;
	
	msgs[1].addr = slv_addr;
    msgs[1].flags = RT_I2C_RD;
    msgs[1].buf = (uint8_t *)(buf+2);
    msgs[1].len = 2;
	
    
    if (rt_i2c_transfer(i2c_bus, msgs, 2) != 0)
    {
		*reg_data = buf[3] | (buf[2] << 8);
        return 0 ;
    }
    else
    {
		cam_err("[%s] error\r\n",__func__);
        return -1;
    }
}


int imxrt_camera_set_framerate(struct imxrt_camera *cam,int framerate)
{
    if (cam->sensor.framerate == framerate) {
       /* no change */
        return 0;
    }
#ifdef 	SOC_IMXRT1170_SERIES
#else		
	if ((framerate & 0x80000000) && (cam->sensor_id == OV7725_ID))
		CCM->CSCDR3 = framerate & (0x1F<<9);
#endif
    /* call the sensor specific function */
    if (cam->sensor.set_framerate == NULL
        || cam->sensor.set_framerate(&cam->sensor, framerate) != 0) {
        /* operation not supported */
        return -1;
    }

    /* set the frame rate */
    cam->sensor.framerate = framerate;

    return 0;
}

void imx_cam_flexio_mclk_init()
{
	flexio_timer_config_t timerConfig;

    timerConfig.triggerSelect = 0u;
    timerConfig.triggerPolarity = kFLEXIO_TimerTriggerPolarityActiveHigh;
    timerConfig.triggerSource = kFLEXIO_TimerTriggerSourceInternal;
    timerConfig.pinConfig = kFLEXIO_PinConfigOutput;
    timerConfig.pinSelect = 12;
    timerConfig.pinPolarity = kFLEXIO_PinActiveHigh;
    timerConfig.timerMode = kFLEXIO_TimerModeDual8BitPWM;
    timerConfig.timerOutput = kFLEXIO_TimerOutputZeroNotAffectedByReset;
    timerConfig.timerDecrement = kFLEXIO_TimerDecSrcOnFlexIOClockShiftTimerOutput;
    timerConfig.timerReset = kFLEXIO_TimerResetNever;
    timerConfig.timerDisable = kFLEXIO_TimerDisableNever;
    timerConfig.timerEnable = kFLEXIO_TimerEnabledAlways;
    timerConfig.timerStop = kFLEXIO_TimerStopBitDisabled;
    timerConfig.timerStart = kFLEXIO_TimerStartBitDisabled;
    timerConfig.timerCompare = 0x201; /* 120MHz clock source generates 24MHz clock.*/

    FLEXIO_SetTimerConfig(FLEXIO1, 1u, &timerConfig);
}

void imx_cam_flexio_hclk_init()
{
	 flexio_timer_config_t timerConfig;

    timerConfig.triggerSelect = FLEXIO_TIMER_TRIGGER_SEL_PININPUT(8);;
    timerConfig.triggerPolarity = kFLEXIO_TimerTriggerPolarityActiveLow;
    timerConfig.triggerSource = kFLEXIO_TimerTriggerSourceInternal;
    timerConfig.pinConfig = kFLEXIO_PinConfigOutputDisabled;
    timerConfig.pinSelect = 8;
    timerConfig.pinPolarity = kFLEXIO_PinActiveHigh;
    timerConfig.timerMode = kFLEXIO_TimerModeDual8BitBaudBit;
    timerConfig.timerOutput = kFLEXIO_TimerOutputZeroNotAffectedByReset;
    timerConfig.timerDecrement = kFLEXIO_TimerDecSrcOnPinInputShiftPinInput;
    timerConfig.timerReset = kFLEXIO_TimerResetOnTimerTriggerRisingEdge;
    timerConfig.timerDisable = kFLEXIO_TimerDisableOnTimerCompare;
    timerConfig.timerEnable = kFLEXIO_TimerEnableOnTriggerRisingEdge;
    timerConfig.timerStop = kFLEXIO_TimerStopBitDisabled;
    timerConfig.timerStart = kFLEXIO_TimerStartBitDisabled;
    timerConfig.timerCompare = 0x0001;

    FLEXIO_SetTimerConfig(FLEXIO1, 2u, &timerConfig);
    FLEXIO_EnableTimerStatusInterrupts(FLEXIO1, 1<<2u);
}

void imx_cam_flexio_init(struct imxrt_camera *cam, bool isWindowing)
{
	cam->FlexioCameraDevice.flexioBase = FLEXIO1;                 
    cam->FlexioCameraDevice.datPinStartIdx = 0;     
    cam->FlexioCameraDevice.pclkPinIdx = 10;          
    cam->FlexioCameraDevice.hrefPinIdx = 8;          
    cam->FlexioCameraDevice.shifterStartIdx = 0;   
    cam->FlexioCameraDevice.shifterCount = 8;        
    cam->FlexioCameraDevice.timerIdx = 0;                        

    CLOCK_EnableClock(kCLOCK_Flexio1);
    FLEXIO_Reset(FLEXIO1);

    FLEXIO_CAMERA_GetDefaultConfig(&cam->FlexioCameraConfig);
    FLEXIO_CAMERA_Init(&cam->FlexioCameraDevice, &cam->FlexioCameraConfig);
    FLEXIO_CAMERA_ClearStatusFlags(&cam->FlexioCameraDevice, kFLEXIO_CAMERA_RxDataRegFullFlag | kFLEXIO_CAMERA_RxErrorFlag);
	
	imx_cam_flexio_mclk_init();
	
	if(isWindowing)
		imx_cam_flexio_hclk_init();
	
    FLEXIO_CAMERA_Enable(&cam->FlexioCameraDevice, true);
	
	
}


extern void BOARD_FlexIOCam_pwd_io_set(int value);
extern void BORAD_FlexIOCam_rst_io_set(int value);
extern void BORAD_FlexIOCam_get_vsyncio(uint32_t *base, uint8_t *pin,rt_base_t *rttPin);
bool imx_cam_sensor_init(struct imxrt_camera *cam)
{
	struct rt_i2c_bus_device *i2c_bus;
	

	
	i2c_bus = rt_i2c_bus_device_find((const char *)cam->sensor_bus_name);
	if(i2c_bus == RT_NULL)
	{
		cam_err("[%s]driver can not find %s bus\r\n",__func__,cam->sensor_bus_name);
		return false;
	}
	
	CLOCK_SetMux(kCLOCK_Flexio1Mux, 3);
    CLOCK_SetDiv(kCLOCK_Flexio1PreDiv, 1);
    CLOCK_SetDiv(kCLOCK_Flexio1Div, 3);
	
    BORAD_FlexIOCam_get_vsyncio((uint32_t *)(&cam->vSyncgpio),&cam->vSyncpin,&cam->vSyncRTTpin);
	imx_cam_flexio_init(cam, 0);
	cam_err("csi root clk:%d\r\n",CLOCK_GetClockRootFreq(kCLOCK_Flexio1ClkRoot));
	for (int i=0; i< 3;i++)
	{
		cam->sensor.i2c_bus = (uint32_t *)i2c_bus;
		cam->sensor.cambus_readb = imx_cam_sensor_cambus_readb;
		cam->sensor.cambus_writeb = imx_cam_sensor_cambus_writeb;
		cam->sensor.cambus_readw = imx_cam_sensor_cambus_readw;
		cam->sensor.cambus_writew = imx_cam_sensor_cambus_writew;
		cam->sensor.cambus_readb2 = imx_cam_sensor_cambus_readb2;
		cam->sensor.cambus_writeb2 = imx_cam_sensor_cambus_writeb2;
		cam->sensor.cambus_writews = imx_cam_sensor_cambus_writews;
		cam->sensor.cambus_readws = imx_cam_sensor_cambus_readws;
		cam->sensor_addr = supported_sensors[i][0];
		cam->sensor.slv_addr = cam->sensor_addr;

		
		//reset camera
		//power down to low
		BOARD_FlexIOCam_pwd_io_set(0);
		//reset 
		BORAD_FlexIOCam_rst_io_set(0);
		rt_thread_delay(50);
		BORAD_FlexIOCam_rst_io_set(1);
		rt_thread_delay(50);

		//read 2 bytes first for mt9m114
		uint16_t sensor_id_s = 0;
		uint8_t sensor_id = 0;
		if(imx_cam_sensor_cambus_readw2(&cam->sensor,cam->sensor_addr,supported_sensors[i][1], &sensor_id_s) == 0)
		{//two bytes addr, two bytes value
			if(sensor_id_s == MT9M114_ID)
			{
				cam_err("Camera Device id:0x%x\r\n",sensor_id_s);
				mt9m114_init(&cam->sensor);
				cam->sensor_id = MT9M114_ID;
				return true;
			}
		}
		
		if (imx_cam_sensor_readb2_reg(i2c_bus,cam->sensor_addr,supported_sensors[i][1], &sensor_id) == 0)
		{//two byte addr, one bytes value
			if(sensor_id == OV5640_ID)
			{
				cam_err("Camera Device id:0x%x\r\n",sensor_id);
				#ifdef SENSOR_OV5640
				ov5640_init(&cam->sensor);
				#endif
				cam->sensor_id = OV5640_ID;
				return true;
			}
		}
		
		if (imx_cam_sensor_read_reg(i2c_bus,cam->sensor_addr,OV_CHIP_ID, &sensor_id) == 0)
		{//one byte addr. one byte value
			switch(sensor_id)
			{

				case OV9650_ID:
					cam_err("Camera Device id:0x%x\r\n",sensor_id);
					#ifdef SENSOR_OV9650
					ov9650_init(&cam->sensor);
					#endif
					cam->sensor_id = OV9650_ID;
					return true;
				case OV2640_ID:
					cam_err("Camera Device id:0x%x\r\n",sensor_id);
					#ifdef SENSOR_OV2640
					ov2640_init(&cam->sensor);
					#endif
					cam->sensor_id = OV2640_ID;
					return true;
				case OV7725_ID:
					cam_err("Camera Device id:0x%x\r\n",sensor_id);
					#ifdef SENSOR_OV7725
					ov7725_init(&cam->sensor);
					cam->sensor_id = OV7725_ID;
					imxrt_camera_set_framerate(cam,0x80000000 | (2<<9|(8-1)<<11));
					#endif
					
					return true;
				default:
					
					break;
			}
		
		}
	}
	cam_err("[%s] sensor id:0x%2x not support\r\n",__func__,cam->sensor_id);
	return false;
}



void imx_cam_reset(struct imxrt_camera *cam)
{
	mutex_init0(&JPEG_FB()->lock);
	//sensor init
	imx_cam_sensor_init(cam);
	
	csi_calc_first = 0;
	cam->sensor.isWindowing = 0;
	cam->sensor.wndH = cam->sensor.fb_h;
	cam->sensor.wndW = cam->sensor.fb_w;
	cam->sensor.wndX = cam->sensor.wndY = 0;	
	
	cam->sensor.sde          = 0xFF;
    cam->sensor.pixformat    = 0xFF;
    cam->sensor.framesize    = 0xFF;
    cam->sensor.framerate    = 0xFF;
    cam->sensor.gainceiling  = 0xFF;


    // Call sensor-specific reset function; in the moment,we use our init function and defaults regs
    if (cam->sensor.reset)
		cam->sensor.reset(&cam->sensor);
}



#if defined(__CC_ARM)
#define ARMCC_ASM_FUNC	__asm
ARMCC_ASM_FUNC __attribute__((section(".ram_code"))) uint32_t ExtractYFromYuv(uint32_t dmaBase, uint32_t datBase, uint32_t _128bitUnitCnt) {
	push	{r4-r7, lr}
10
	LDMIA	R0!, {r3-r6}
	// schedule code carefully to allow dual-issue on Cortex-M7
	bfi		r7, r3, #0, #8	// Y0
	bfi		ip, r5, #0, #8	// Y4
	lsr		r3,	r3,	#16
	lsr		r5,	r5,	#16
	bfi		r7, r3, #8, #8	// Y1
	bfi		ip, r5, #8, #8  // Y5
	bfi		r7, r4, #16, #8 // Y2
	bfi		ip, r6, #16, #8 // Y6
	lsr		r4,	r4,	#16
	lsr		r6,	r6,	#16
	bfi		r7, r4, #24, #8 // Y3
	bfi		ip, r6, #24, #8	// Y7
	STMIA	r1!, {r7, ip}
	
	subs	r2,	#1
	bne		%b10
	mov		r0,	r1
	pop		{r4-r7, pc}
}
#elif defined(__CLANG_ARM)
__attribute__((section(".ram_code")))
__attribute__((naked))  uint32_t ExtractYFromYuv(uint32_t dmaBase, uint32_t datBase, uint32_t _128bitUnitCnt) {
		__asm volatile (
		"	push	{r1-r7, ip, lr}  \n "
		"10:  \n "
		"	ldmia	r0!, {r3-r6}  \n "
			// schedule code carefully to allow dual-issue on Cortex-M7
		"	bfi		r7, r3, #0, #8  \n "	// Y0
		"	bfi		ip, r5, #0, #8  \n "	// Y4
		"	lsr		r3,	r3,	#16  \n "
		"	lsr		r5,	r5,	#16  \n "
		"	bfi		r7, r3, #8, #8  \n "	// Y1
		"	bfi		ip, r5, #8, #8  \n "  // Y5
		"	bfi		r7, r4, #16, #8  \n " // Y2
		"	bfi		ip, r6, #16, #8  \n " // Y6
		"	lsr		r4,	r4,	#16  \n "
		"	lsr		r6,	r6,	#16  \n "
		"	bfi		r7, r4, #24, #8  \n " // Y3
		"	bfi		ip, r6, #24, #8  \n "	// Y7
		"	stmia	r1!, {r7, ip}  \n "	
		"	subs	r2,	#1  \n "
		"	bne		10b  \n "
		"	mov		r0,	r1  \n "
		"	pop		{r1-r7, ip, pc}  \n "		
	);
}
#else
__attribute__((naked))
__attribute__((section(".ram_code"))) uint32_t ExtractYFromYuv(uint32_t dmaBase, uint32_t datBase, uint32_t _128bitUnitCnt) {
	__asm volatile (
		"	push	{r1-r7, ip, lr}  \n "
		"10:  \n "
		"	ldmia	r0!, {r3-r6}  \n "
			// schedule code carefully to allow dual-issue on Cortex-M7
		"	bfi		r7, r3, #0, #8  \n "	// Y0
		"	bfi		ip, r5, #0, #8  \n "	// Y4
		"	lsr		r3,	r3,	#16  \n "
		"	lsr		r5,	r5,	#16  \n "
		"	bfi		r7, r3, #8, #8  \n "	// Y1
		"	bfi		ip, r5, #8, #8  \n "  // Y5
		"	bfi		r7, r4, #16, #8  \n " // Y2
		"	bfi		ip, r6, #16, #8  \n " // Y6
		"	lsr		r4,	r4,	#16  \n "
		"	lsr		r6,	r6,	#16  \n "
		"	bfi		r7, r4, #24, #8  \n " // Y3
		"	bfi		ip, r6, #24, #8  \n "	// Y7
		"	stmia	r1!, {r7, ip}  \n "	
		"	subs	r2,	#1  \n "
		"	bne		10b  \n "
		"	mov		r0,	r1  \n "
		"	pop		{r1-r7, ip, pc}  \n "		
	);
}

#endif

__attribute__((section(".ram_code")))
__attribute__((naked)) void rgb32Torgb565(uint32_t *src, uint16_t *dst, uint32_t len){
	__asm volatile(
		"push {r3-r8, lr}\n"
		"loop: \n"
		"mov r8, #0\n"
		"ldrd r3,r4, [r0], #8\n"
	
		"ubfx r5, r3, #0, #8\n" //r1
		"lsr r5, #3\n"	
		"ubfx r6, r3, #8, #8\n" //g1
		"lsr r6, #2\n"
		"orr r8, r8, r5\n"
		"ubfx r7, r3, #16, #8\n" //b1
		"orr r8, r8, r6, LSL #5\n"	
		"lsr r7, #3\n"

	
		"ubfx r5, r4, #0, #8\n" //r1
		"orr r8, r8, r7, LSL #11\n"
		"lsr r5, #3\n"	
		"ubfx r6, r4, #8, #8\n" //g1
		"lsr r6, #2\n"
		"orr r8, r8, r5, LSL #16\n"
		"ubfx r7, r4, #16, #8\n" //b1
		"orr r8, r8, r6, LSL #21\n"	
		"lsr r7, #3\n"
		"orr r8, r8, r7, LSL #27\n"
			
		"rev16 r8, r8\n"
		"subs r2, 8\n"
		"str r8, [r1], #4\n"
		"bne loop\n"
		"pop {r3-r8, pc}\n"
		);
}
void rgb32Torgb565_c(uint32_t *src, uint16_t *dst, uint32_t len){
	uint8_t r,g,b;
	for(int i=0;i<len;i+=4){
		uint32_t col = *(src++);
		r = (col>>16) & 0xff;
		g = (col>>8) & 0xff;
		b = col & 0xff;
		*dst++ = ((r >> 3) << 11) | ((g>>2)<<5) | ((b>>3));	
	}
}

__attribute__((naked)) void copy2mem(void* dst, void* src, uint32_t len){
	__asm volatile(
		"push {r3-r4, lr}\n"
		"loop1: \n"
		"ldrd r3, r4, [r1], #8\n"
		"strd r3, r4, [r0], #8\n"
		"subs r2, #8\n"
		"bne loop1\n"
		"pop {r3-r4, lr}\n"
		);
}

__attribute__((naked)) static void unaligned_memcpy_rev16(uint16_t *dst, uint16_t *src, uint32_t u64Cnt){
	__asm volatile(
		"	push {r3, r4} \n"
		"loop2: \n"
		"	ldrd r3, r4, [r1], #8 \n"
		"	rev16 r3, r3 \n"
		"	rev16 r4, r4 \n"
		"	strd r3, r4, [r0], #8 \n"
		"	subs r2, #1 \n"
		"	bne loop2 \n"
		"	pop {r3, r4} \n"
		"	bx lr ");
	
}

static uint64_t rt_tick_get_us(){
	uint64_t tick = rt_tick_get();
	uint64_t us = 1000000 * (SysTick->LOAD - SysTick->VAL) / (SystemCoreClock);
	us += tick * 1000;
	return us;
}
uint32_t g_start,g_s1,g_e1, g_csi_end;
uint64_t g_csi_start, g_exit;
static int sync_found = 0;


void flexio_camera_dma_cb(edma_handle_t *handle, void *param, bool transferDone, uint32_t tcds)
{
	struct imxrt_camera *cam = (struct imxrt_camera *)param;
    vbuffer_t *buffer = framebuffer_get_tail(FB_NO_FLAGS);
        
    if (!buffer) 
    {
        framebuffer_flush_buffers();
        buffer = framebuffer_get_tail(FB_NO_FLAGS);
    }

    if(!buffer)
        return;
/*
    if (pCam->s_irq.isGray || 
			(pCam->sensor.isWindowing &&  lineNdx >= pCam->sensor.wndY && lineNdx - pCam->sensor.wndY <= pCam->sensor.wndH) )
    {

        dmaBase += pCam->sensor.wndX * 2 * pCam->s_irq.linePerFrag;	// apply line window offset
        if (pCam->s_irq.isGray) {
            
            pCam->s_irq.datCurBase = ExtractYFromYuv(s_dmaFragBufs, buffer->data, (cam->sensor.framebytes) >> 3);
        
        } else {
           memcpy(buffer->data,s_dmaFragBufs,cam->sensor.framebytes);
        }
    }
    else
*/
    {
        
        memcpy(buffer->data,(void*)pFlexioCameraFrameBuffer,cam->sensor.framebytes);
    }

    rt_event_send(&frame_event, EVENT_CSI);
}



void FLEXIO_CameraFrameEnd(struct imxrt_camera *cam)
{
    FLEXIO_CAMERA_ClearStatusFlags(&cam->FlexioCameraDevice,
                                   kFLEXIO_CAMERA_RxDataRegFullFlag | kFLEXIO_CAMERA_RxErrorFlag);

    
}

void FLEXIO_CameraLineStart(struct imxrt_camera *cam)
{
    if(cam->LineCount < cam->sensor.fb_h)
    {
        DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].DADDR = (uint32_t)pFlexioCameraFrameBuffer + cam->LineBytes * cam->LineCount;
        DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].CITER_ELINKNO = (cam->LineBytes / 32u);
        DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].BITER_ELINKNO = (cam->LineBytes / 32u);
        DMA0->SERQ = DMA_SERQ_SERQ(FLEXIO_CAMERA_DMA_CHN);
        cam->LineCount++;
    }
}

void FLEXIO_CameraFrameStart(struct imxrt_camera *cam)
{
    if(pFlexioCameraFrameBuffer == (uint32_t)s_dmaFragBufs[1])
    {
        pFlexioCameraFrameBuffer = (uint32_t)s_dmaFragBufs[0];
    }
    else
    {
        pFlexioCameraFrameBuffer = (uint32_t)s_dmaFragBufs[1];
    }
	
	DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].DADDR = (uint32_t)pFlexioCameraFrameBuffer;
	DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].CITER_ELINKNO = (cam->sensor.framebytes / 32u);
	DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].BITER_ELINKNO = (cam->sensor.framebytes / 32u);
	DMA0->SERQ = DMA_SERQ_SERQ(FLEXIO_CAMERA_DMA_CHN);
    cam->LineCount = 0;
    //FLEXIO_CameraLineStart(cam);
}

void FLEXIO1_IRQHandler(void)
{
	if (pCam == NULL)
		return;
	
    FLEXIO_ClearTimerStatusFlags(FLEXIO1, 1<<2u);
    FLEXIO_CameraLineStart(pCam);
    __DSB();
}

void vSync_isr_handler(void *arg)
{
	struct imxrt_camera *cam = (struct imxrt_camera *)arg;
	
	#if 1
	
    FLEXIO_CameraFrameEnd(cam);

    vbuffer_t *buffer = framebuffer_get_tail(FB_NO_FLAGS);
        
    if (!buffer) 
    {
        framebuffer_flush_buffers();
        buffer = framebuffer_get_tail(FB_NO_FLAGS);
    }

    if(!buffer)
        return;
	
	if(pFlexioCameraFrameBuffer > 0) {
		memcpy(buffer->data,(void*)pFlexioCameraFrameBuffer,cam->sensor.framebytes);
		rt_event_send(&frame_event, EVENT_CSI);
	}
    FLEXIO_CameraFrameStart(cam);	
	#else
	
	
	int dma_chn = cam->g_EDMA_Handle.channel;
	
	if(pFlexioCameraFrameBuffer == (uint32_t)s_dmaFragBufs[1])
    {
        pFlexioCameraFrameBuffer = (uint32_t)s_dmaFragBufs[0];
    }
    else
    {
        pFlexioCameraFrameBuffer = (uint32_t)s_dmaFragBufs[1];
    }
	
    // void dma_restart(uint8 *dest_addr)
    DMA0->TCD[dma_chn].DADDR = (uint32_t)(pFlexioCameraFrameBuffer);
    // flexio_flag_clear();
    FLEXIO_CAMERA_ClearStatusFlags(&cam->FlexioCameraDevice, kFLEXIO_CAMERA_RxDataRegFullFlag | kFLEXIO_CAMERA_RxErrorFlag);
    DMA0->SERQ = DMA_SERQ_SERQ(dma_chn);
	
	#endif
    __DSB();
}

static void configDMA(struct imxrt_camera *cam)
{
#if 1
	 /* Configure DMA TCD */
    DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].SADDR = FLEXIO_CAMERA_GetRxBufferAddress(&cam->FlexioCameraDevice);
    DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].SOFF = 0u;
    DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].ATTR = DMA_ATTR_SMOD(0u) |
                                            DMA_ATTR_SSIZE(8u) |
                                            DMA_ATTR_DMOD(0u) |
                                            DMA_ATTR_DSIZE(8u);
    DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].NBYTES_MLNO = 32u;
    DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].SLAST = 0u;
    DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].DOFF = 32u;
    DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].DLAST_SGA = 0;
    DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].CSR = 0u;
    DMA0->TCD[FLEXIO_CAMERA_DMA_CHN].CSR |= DMA_CSR_DREQ_MASK;

    /* Configure DMA MUX Source */
    DMAMUX->CHCFG[FLEXIO_CAMERA_DMA_CHN] = DMAMUX->CHCFG[FLEXIO_CAMERA_DMA_CHN] &
                                            (~DMAMUX_CHCFG_SOURCE_MASK) | 
                                            DMAMUX_CHCFG_SOURCE(FLEXIO_CAMERA_DMA_MUX_SRC);
    /* Enable DMA channel. */
    DMAMUX->CHCFG[FLEXIO_CAMERA_DMA_CHN] |= DMAMUX_CHCFG_ENBL_MASK;
#else	
    int dma_chn = DMA_CHN;  // Default
	edma_modulo_t s_addr_modulo;
	
	DMAMUX_Deinit(DMAMUX);
	DMAMUX_Init(DMAMUX);
	DMAMUX_SetSource(DMAMUX, DMA_CHN, DMA_MUX_SRC);
	DMAMUX_EnableChannel(DMAMUX, DMA_CHN);
	
	EDMA_CreateHandle(&cam->g_EDMA_Handle, DMA0, dma_chn);
	EDMA_SetCallback(&cam->g_EDMA_Handle, flexio_camera_dma_cb, NULL);
	EDMA_PrepareTransfer(&cam->transferConfig, 
							(void *)FLEXIO_CAMERA_GetRxBufferAddress(&cam->FlexioCameraDevice), 
							1,
							(void *)(s_dmaFragBufs), 
							4,
							1*FXIO_SHFT_COUNT,
							cam->LineBytes,
							kEDMA_MemoryToMemory);

	EDMA_SubmitTransfer(&cam->g_EDMA_Handle, &cam->transferConfig);

	switch(4*FXIO_SHFT_COUNT)
	{
		case 4:     s_addr_modulo = kEDMA_Modulo4bytes;break;
		case 8:     s_addr_modulo = kEDMA_Modulo8bytes;break;
		case 16:    s_addr_modulo = kEDMA_Modulo16bytes;break;
		case 32:    s_addr_modulo = kEDMA_Modulo32bytes;break;
		default:assert(0); 
	}

	EDMA_SetModulo(DMA0, dma_chn, s_addr_modulo, kEDMA_ModuloDisable);
	EDMA_StartTransfer(&cam->g_EDMA_Handle);
    
#endif    
	// flexio_enable_rxdma()
	FLEXIO_CAMERA_EnableRxDMA(&cam->FlexioCameraDevice, true);
	
	NVIC_SetPriority(DMA0_DMA16_IRQn, IRQ_PRI_CSI);            
}

void imx_cam_csi_start_frame(struct imxrt_camera *cam)
{//flexio	
	uint32_t start = rt_tick_get_us();
	edma_config_t edmaConfig;
	
	pCam = cam;
	
	DMAMUX_Init(DMAMUX);
    EDMA_GetDefaultConfig(&edmaConfig);
    edmaConfig.enableDebugMode = true;
    EDMA_Init(DMA0, &edmaConfig);
	
    imx_cam_flexio_init(cam, 0);
	
    configDMA(cam);

    // 设置场中断
    rt_pin_mode(cam->vSyncRTTpin, PIN_MODE_INPUT_PULLUP);
    rt_pin_attach_irq(cam->vSyncRTTpin, PIN_IRQ_MODE_FALLING, vSync_isr_handler, (void*)cam);
    rt_pin_irq_enable(cam->vSyncRTTpin, PIN_IRQ_ENABLE);

    NVIC_SetPriority(FLEXIO1_IRQn, IRQ_PRI_CSI); 
	EnableIRQ(FLEXIO1_IRQn);
	
}

void imx_cam_start_snapshot(struct imxrt_camera *cam)
{
	pCam = cam;
	//start csi
	uint8_t bpp=2;
	int16_t w,h;
	uint32_t size;
	
	w = cam->sensor.isWindowing ? cam->sensor.fb_w : resolution[cam->sensor.framesize][0];
	h = cam->sensor.isWindowing ? cam->sensor.fb_h : resolution[cam->sensor.framesize][1];
	switch (cam->sensor.pixformat) {
        case PIXFORMAT_GRAYSCALE:
            bpp = 1;
            break;
        case PIXFORMAT_YUV422:
        case PIXFORMAT_RGB565:
            bpp = 2;
            break;
        case PIXFORMAT_BAYER:
            bpp = 3;
            break;
        case PIXFORMAT_JPEG:
            // Read the number of data items transferred
            // MAIN_FB()->bpp = (MAX_XFER_SIZE - __HAL_DMA_GET_COUNTER(&DMAHandle))*4;
            break;
		default:
			bpp = 2;
			break;
    }
	
	size = w * h * bpp;
	cam->LineBytes = w * bpp;
    cam->sensor.framebytes = size;
	framebuffer_set_buffers(3);
	imx_cam_csi_start_frame(cam);
}



void imx_cam_sensor_set_contrast(struct imxrt_camera *cam, uint32_t level)
{
	if (cam->sensor.set_contrast != NULL)
		cam->sensor.set_contrast(&cam->sensor,level);
}

void imx_cam_sensor_set_gainceiling(struct imxrt_camera *cam, gainceiling_t gainceiling)
{
	if (cam->sensor.set_gainceiling != NULL && !cam->sensor.set_gainceiling(&cam->sensor, gainceiling))
		cam->sensor.gainceiling = gainceiling;
}

int imx_cam_sensor_set_framesize(struct imxrt_camera *cam, framesize_t framesize)
{
	if(cam->sensor.set_framesize == NULL || cam->sensor.set_framesize(&cam->sensor,framesize) != 0)
		return -1;
	
	cam->sensor.framesize = framesize;
	
	cam->sensor.fb_w = resolution[framesize][0];
	cam->sensor.fb_h = resolution[framesize][1];
	
	cam->sensor.wndX = 0; cam->sensor.wndY = 0 ; cam->sensor.wndW = cam->sensor.fb_w ; cam->sensor.wndH = cam->sensor.fb_h;

	return 0;
}

void imx_cam_sensor_set_pixformat(struct imxrt_camera *cam, pixformat_t pixformat)
{
	if (cam->sensor.set_pixformat == NULL || cam->sensor.set_pixformat(&cam->sensor,pixformat) != 0)
		return;
	if (cam->sensor.pixformat == pixformat)
		return;
	
	cam->sensor.pixformat = pixformat;
}

void imx_cam_csi_stop(struct rt_camera_device *cam)
{
//flexio stop
}

static rt_err_t imx_cam_camera_control(struct rt_camera_device *cam, rt_uint32_t cmd, rt_uint32_t parameter)
{
	struct imxrt_camera *imx_cam = (struct imxrt_camera *)cam->imx_cam;
	switch(cmd)
	{
		case RT_DRV_CAM_CMD_RESET:
			cam->omv_tid = rt_thread_self();
			if(cam->status == RT_CAMERA_DEVICE_RUNING)
			{
				imx_cam_csi_stop(cam);
			}
			imx_cam_reset(imx_cam);
			pCam = (struct imxrt_camera *)cam->imx_cam;
			cam->status = RT_CAMERA_DEVICE_INIT;
			break;
		case RT_DRV_CAM_CMD_SET_FRAMERATE:
			return imxrt_camera_set_framerate(imx_cam, parameter);
		    break;
		case RT_DRV_CAM_CMD_SET_CONTRAST:
			imx_cam_sensor_set_contrast(imx_cam, parameter);
			break;
		case RT_DRV_CAM_CMD_SET_GAINCEILING:
			imx_cam_sensor_set_gainceiling(imx_cam, parameter);
			break;
		case RT_DRV_CAM_CMD_SET_FRAMESIZE:
			return imx_cam_sensor_set_framesize(imx_cam, parameter);
		case RT_DRV_CAM_CMD_SET_PIXFORMAT:
			imx_cam_sensor_set_pixformat(imx_cam, parameter);
			break;
		case RT_DRV_CAM_CMD_SNAPSHOT:
		{
			if(cam->status != RT_CAMERA_DEVICE_RUNING)
			{
				//rt_sem_init(cam->sem,"cam", 0, RT_IPC_FLAG_FIFO);
				imx_cam_start_snapshot((struct imxrt_camera *)cam->imx_cam);
				cam->status = RT_CAMERA_DEVICE_RUNING;
				
			}
			//rest total count 
			//#warning "Enable this, and can not begin twice"
			break;
			
		}
		case RT_DRV_CAM_CMD_SHUTDOWN:
			//imx_cam_csi_stop(cam);
			//rt_event_control(&frame_event,RT_IPC_CMD_RESET,0);
			break;
	}
	
	return RT_EOK;
}

static rt_size_t imx_cam_get_frame_jpeg(struct rt_camera_device *cam, void *frame_ptr)
{
	return 0;
}

#define PRODUCT_PRIO   (RT_MAIN_THREAD_PRIORITY-6)
#define CONSUME_PRIO   (RT_MAIN_THREAD_PRIORITY-5)

// when we use the lvgl, the omv is as a plug-in
// but now we provide a way to pause the lvgl, and 
// the omv concour the panel
static bool lvgl_running = false;
void set_lvgl_running(bool enable){
	lvgl_running = enable;
}

static void framebuffer_update_lcd()
{
#ifdef RT_USING_LCD	
	image_t main_fb_src;
    framebuffer_init_image(&main_fb_src);
    image_t *src = &main_fb_src;
	
	static uint8_t fbIdx = 0;
	
	LCDMonitor_Update(fbIdx++,pCam->s_irq.isGray, pCam->sensor.wndH, pCam->sensor.wndW, (uint32_t)src->data);
#endif	
}

static rt_size_t imx_cam_get_frame(struct rt_camera_device *cam, image_t * image)
{
	static uint32_t ls_prevTick = 0;
	uint32_t s_minProcessTicks = 10;
	
	struct imxrt_camera *imx_cam = (struct imxrt_camera *)cam->imx_cam;
	register rt_ubase_t temp;
	
	if(JPEG_FB()->enabled)
		framebuffer_update_jpeg_buffer();
	
	if(!lvgl_running)
		framebuffer_update_lcd();
	
	uint32_t diff = rt_tick_get() - ls_prevTick;
	if((JPEG_FB()->enabled)&&(diff < s_minProcessTicks)){
			rt_thread_mdelay(s_minProcessTicks - diff);
	}

	if((pCam->sensor.isWindowing))
	{
		uint32_t win_h = pCam->sensor.wndH;
		uint32_t win_w = pCam->sensor.wndW;
		uint32_t win_x = pCam->sensor.wndX;
		uint32_t win_y = pCam->sensor.wndY;
		// handle the windowing 
		MAIN_FB()->w = win_w;
		MAIN_FB()->h = win_h;
	}else{
		MAIN_FB()->w = pCam->sensor.fb_w;
		MAIN_FB()->h = pCam->sensor.fb_h;		
	}
	MAIN_FB()->pixfmt = pCam->sensor.pixformat;

	vbuffer_t *buffer = NULL;
	while(1)
	{
		if (rt_event_recv(&frame_event, EVENT_CSI,RT_EVENT_FLAG_OR | RT_EVENT_FLAG_CLEAR,
                      RT_WAITING_FOREVER, NULL) == RT_EOK)
		{
			temp = rt_hw_interrupt_disable();
			buffer = framebuffer_get_head(FB_NO_FLAGS);
			
			if(buffer)
			{
				 rt_hw_interrupt_enable(temp);
			 break;
			}
			rt_hw_interrupt_enable(temp);
		}
	}
	
	framebuffer_init_image(image);
	
//	if(!lvgl_running){
//		#ifdef RT_USING_LCD
//		static uint8_t fbIdx = 0;
//		LCDMonitor_Update(fbIdx++,pCam->s_irq.isGray, pCam->sensor.wndH, pCam->sensor.wndW, (uint32_t)buffer->data);
//		#endif
//		
//		//if(JPEG_FB()->enabled)
//		//	framebuffer_update_jpeg_buffer();
//	}
	
	ls_prevTick = rt_tick_get();
	return 0;

}

static const struct rt_camera_device_ops imxrt_cam_ops =
{
    .get_frame_jpeg = imx_cam_get_frame_jpeg,
	.get_frame = imx_cam_get_frame,
	.camera_control = imx_cam_camera_control,
};

//csi timeout handler
void csi_to_handler(void* paramter)
{
	return;
	//struct rt_camera_device *cam = (struct rt_camera_device *)paramter;
	//imx_cam_csi_stop(cam);
	//rt_event_control(&frame_event,RT_IPC_CMD_RESET,0);
}

void camera_clear_by_omv_ide(void)
{
	if (pCam) {
		imx_cam_csi_stop(pCam->rtt_device);
		rt_event_control(&frame_event,RT_IPC_CMD_RESET,0);
		set_lvgl_running(0);
	}
}


static void imxrt_cam_device_init(struct imxrt_camera *cam)
{
	struct rt_camera_device *device = NULL;
	rt_err_t ret;
	
	device = (struct rt_camera_device *)rt_malloc(sizeof(struct rt_camera_device));
	if (device == NULL)
	{
		cam_err("malloc failed in %s\r\n",__func__);
		return;
	}
	cam_echo("Camera device: %s init\r\n",cam->name);
	device->imx_cam = (uint32_t *)cam;
	cam->rtt_device = device;
	device->status = RT_CAMERA_DEVICE_INIT;
	device->ops = &imxrt_cam_ops;
	
	//g_csi_to = rt_timer_create("cam",csi_to_handler,(void*)cam,g_csi_to_ticks,RT_TIMER_FLAG_ONE_SHOT);
	ret = rt_event_init(&frame_event, "event", RT_IPC_FLAG_FIFO);
    if (ret != RT_EOK)
    {
        rt_kprintf("init event failed.\n");
    }

	device->status = RT_CAMERA_DEVICE_SUSPEND;
}

struct rt_camera_device * imxrt_camera_device_find(char *name)
{
	int i;
	
	for(i = 0; i < CAM_NUM;i++)
	{
		if(strcmp(name,cams[i].name) == 0)
		{
			return cams[i].rtt_device;
		}
	}
	
	return NULL;
}

bool sensor_is_detected(struct rt_camera_device *sensor)
{
	struct imxrt_camera *imx_cam = (struct imxrt_camera *)sensor->imx_cam;
	
	return imx_cam_sensor_init(imx_cam);
}

int imxrt_camera_width(struct rt_camera_device *sensor)
{
	struct imxrt_camera *imx_cam = (struct imxrt_camera *)sensor->imx_cam;
	return imx_cam->sensor.fb_w;
}
int imxrt_camera_height(struct rt_camera_device *sensor)
{
	struct imxrt_camera *imx_cam = (struct imxrt_camera *)sensor->imx_cam;
	return imx_cam->sensor.fb_h;
}
int imxrt_camera_chip_id(struct rt_camera_device *sensor)
{
	return 0;
}
int imxrt_camera_pixformat(struct rt_camera_device *sensor)
{
	struct imxrt_camera *imx_cam = (struct imxrt_camera *)sensor->imx_cam;
	return imx_cam->sensor.pixformat;
}
int imxrt_camera_framesize(struct rt_camera_device *sensor)
{
	struct imxrt_camera *imx_cam = (struct imxrt_camera *)sensor->imx_cam;
	return imx_cam->sensor.framesize;
}

int imxrt_camera_framerate(struct rt_camera_device *sensor)
{
	struct imxrt_camera *imx_cam = (struct imxrt_camera *)sensor->imx_cam;
	return imx_cam->sensor.framerate;
}
int imxrt_camera_set_windowing(struct rt_camera_device *sensor, int x,int y, int w,int h)
{
	struct imxrt_camera *imx_cam = (struct imxrt_camera *)sensor->imx_cam;
	sensor_t *g_pcur_sensor = &imx_cam->sensor;
	w = (w + 7) & ~7 , x = (x + 7) & ~7;
	if (x >= g_pcur_sensor->fb_w - 8)
		x = g_pcur_sensor->fb_w - 8;
	if (y >= g_pcur_sensor->fb_h - 1)
		y = g_pcur_sensor->fb_h - 1;
	if (x + w > g_pcur_sensor->fb_w)
		w = g_pcur_sensor->fb_w - x;
	if (y + h > g_pcur_sensor->fb_h)
		h = g_pcur_sensor->fb_h - y;

	g_pcur_sensor->isWindowing = (w < g_pcur_sensor->fb_w && h < g_pcur_sensor->fb_h) ? 1 : 0;
	if(g_pcur_sensor->isWindowing){
		g_pcur_sensor->wndX = x ; g_pcur_sensor->wndY = y ; g_pcur_sensor->wndW = w ; g_pcur_sensor->wndH = h;
		
		csi_calc_first = 0;
	}
	return 0;
}
int imxrt_camera_set_auto_gain(struct rt_camera_device *sensor,int enable, float gain_db, float gain_db_ceiling)
{
	return 0;
}
int imxrt_camera_sensor_get_gain_db(struct rt_camera_device *sensor,float *gain_db)
{
	return 0;
}
int imxrt_camera_set_auto_exposure(struct rt_camera_device *sensor,int enable, int exposure_us)
{
	return 0;
}
int imxrt_camera_get_exposure_us(struct rt_camera_device *sensor, int *us)
{
	return 0;
}
int imxrt_camera_set_auto_whitebal(struct rt_camera_device *sensor,int enable, float r_gain_db, float g_gain_db, float b_gain_db)
{
	return 0;
}
int imxrt_camera_get_rgb_gain_db(struct rt_camera_device *sensor,float *r_gain_db, float *g_gain_db, float *b_gain_db)
{
	return 0;
}
int imxrt_camera_set_lens_correction(struct rt_camera_device *sensor,int enable, int radi, int coef)
{
	return 0;
}
uint16_t * imxrt_camera_get_color_palette(struct rt_camera_device *sensor)
{
	return 0;
}
int imxrt_camera_write_reg(struct rt_camera_device *sensor,uint16_t reg_addr, uint16_t reg_data)
{
	return 0;
}
int imxrt_camera_read_reg(struct rt_camera_device *sensor,uint16_t reg_addr)
{
	return 0;
}

int imxrt_camera_ioctl(struct rt_camera_device *sensor,int request, ... /* arg */)
{
    int ret = -1;
	
	struct imxrt_camera *imx_cam = (struct imxrt_camera *)sensor->imx_cam;
    if (imx_cam->sensor.ioctl != NULL) {
        va_list ap;
        va_start(ap, request);
        /* call the sensor specific function */
        ret = imx_cam->sensor.ioctl(&imx_cam->sensor, request, ap);
        va_end(ap);
    }
    return ret;
}

void reset_displaymix(){
   
}
int rt_camera_init(void)
{
	int i;
	
	s_dmaFragBufs[0] = (uint32_t *)&Image$$CAM_FRAME_BUFFER$$Base;
	s_dmaFragBufs[1] = s_dmaFragBufs[0] + FRAMEBUFFER_SIZE/4;
	for(i = 0; i < CAM_NUM;i++)
	{
		imxrt_cam_device_init(&cams[i]);
	}
		
	return 0;
}


INIT_DEVICE_EXPORT(rt_camera_init);

const char *sensor_strerror(int error)
{
    static const char *sensor_errors[] = {
        "No error.",
        "Sensor control failed.",
        "The requested operation is not supported by the image sensor.",
        "Failed to detect the image sensor or image sensor is detached.",
        "The detected image sensor is not supported.",
        "Failed to initialize the image sensor.",
        "Failed to initialize the image sensor clock.",
        "Failed to initialize the image sensor DMA.",
        "Failed to initialize the image sensor DCMI.",
        "An low level I/O error has occurred.",
        "Frame capture has failed.",
        "Frame capture has timed out.",
        "Frame size is not supported or is not set.",
        "Pixel format is not supported or is not set.",
        "Window is not supported or is not set.",
        "Frame rate is not supported or is not set.",
        "An invalid argument is used.",
        "The requested operation is not supported on the current pixel format.",
        "Frame buffer error.",
        "Frame buffer overflow, try reducing the frame size.",
        "JPEG frame buffer overflow.",
    };

    // Sensor errors are negative.
    error = ((error < 0) ? (error * -1) : error);

    if (error > (sizeof(sensor_errors) / sizeof(sensor_errors[0]))) {
        return "Unknown error.";
    } else {
        return sensor_errors[error];
    }
}

int sensor_set_framebuffers(int count)
{
    // Flush previous frame.
    framebuffer_update_jpeg_buffer();

    return framebuffer_set_buffers(count);
}