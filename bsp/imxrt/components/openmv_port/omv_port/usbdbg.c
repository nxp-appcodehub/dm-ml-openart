/*
 * This file is part of the OpenMV project.
 *
 * Copyright (c) 2013-2021 Ibrahim Abdelkader <iabdalkader@openmv.io>
 * Copyright (c) 2013-2021 Kwabena W. Agyeman <kwagyeman@openmv.io>
 *
 * This work is licensed under the MIT license, see the file LICENSE for details.
 *
 * USB debugger.
 */
#include <string.h>
#include <stdio.h>
#include "py/nlr.h"
#include "py/gc.h"
#include "py/mphal.h"
#include "py/obj.h"
#include "py/runtime.h"
#include "pendsv.h"

#include "imlib.h"
#include "sensor.h"

#include "framebuffer.h"
#include "virtual_com.h"
#include "usbdbg.h"
#include "omv_boardconfig.h"
#include "py_image.h"
#include "drv_camera.h"
#include "pendsv.h"
static int xfer_bytes;
static int xfer_length;
static enum usbdbg_cmd cmd;
volatile uint8_t s_UsbDbgIsToRunScript;
static volatile bool script_ready;
static volatile bool script_running;
static volatile bool irq_enabled;
static vstr_t script_buf;
static mp_obj_t mp_const_ide_interrupt = MP_OBJ_NULL;
volatile uint8_t g_omvIdeConnecting;
extern const char *ffs_strerror(FRESULT res);
extern void VCOM_Omv_Start(bool bStart);
void usbdbg_init()
{
    cmd = USBDBG_NONE;
    script_ready=false;
    script_running=false;
    irq_enabled = false;
    vstr_init(&script_buf, 32);
	MP_STATE_PORT(script_buffer) = script_buf.buf;
    MP_STATE_PORT(mp_const_ide_interrupt) = mp_obj_new_exception_msg(&mp_type_Exception, MP_ERROR_TEXT("IDE interrupt"));
	VCOM_Omv_Start(true);
}

__WEAK void Hook_OnUsbDbgScriptExec(void) {}
__WEAK int sensor_get_id(void) {return 1;}

void usbdbg_wait_for_command(uint32_t timeout)
{
    for (mp_uint_t ticks = mp_hal_ticks_ms(); ((mp_hal_ticks_ms() - ticks) < timeout) && (cmd != USBDBG_NONE); );
}

bool usbdbg_script_ready()
{
    return script_ready;
}

void usbdbg_set_script_ready(bool ready)
{
	script_ready = ready;
}

vstr_t *usbdbg_get_script()
{
    return &script_buf;
}

void usbdbg_set_script_running(bool running)
{
    script_running = running;
	s_UsbDbgIsToRunScript = running;
}

bool usbdbg_script_running()
{
	return script_running;
}

inline void usbdbg_set_irq_enabled(bool enabled)
{
    if (enabled) {
        NVIC_EnableIRQ(USB_OTG1_IRQn);
    } else {
        NVIC_DisableIRQ(USB_OTG1_IRQn);
    }
    __DSB(); __ISB();
    irq_enabled = enabled;
}

bool usbdbg_get_irq_enabled()
{
    return irq_enabled;
}

void usbdbg_data_in(void *buffer, int length)
{
    switch (cmd) {
        case USBDBG_FW_VERSION: {
			g_omvIdeConnecting = 1;
			VCOM_FlushTxBuffer();
            uint32_t *ver_buf = buffer;
            ver_buf[0] = FIRMWARE_VERSION_MAJOR;
            ver_buf[1] = FIRMWARE_VERSION_MINOR;
            ver_buf[2] = 0; //FIRMWARE_VERSION_PATCH;
            cmd = USBDBG_NONE;
            break;
        }

        case USBDBG_TX_BUF_LEN: {
			uint32_t *p = (uint32_t*)buffer;
            p[0] = VCOM_OmvGetLogTxLen();
            cmd = USBDBG_NONE;
            break;
        }

        case USBDBG_SENSOR_ID: {
            int sensor_id = sensor_get_id();
            memcpy(buffer, &sensor_id, 4);
            cmd = USBDBG_NONE;
            break;
        }

        case USBDBG_TX_BUF: {
			int n = VCOM_OmvReadLogTxBlk(buffer, length);
			if (n < 0) {
				cmd = USBDBG_NONE;
			} else if (n == 0) {
				cmd = USBDBG_NONE;
			} else {
				xfer_bytes += n;
				if (xfer_bytes == xfer_length) {
					cmd = USBDBG_NONE;
				}
			}
            break;
        }

        case USBDBG_FRAME_SIZE:
			g_omvIdeConnecting = 0;
		#ifdef OMV_MPY_ONLY
			((uint32_t*)buffer)[0] = 0;
			((uint32_t*)buffer)[1] = 0;
			((uint32_t*)buffer)[2] = 0; // MAIN_FB()->w * MAIN_FB()->h * MAIN_FB()->bpp;			
		#else
			#ifdef DUMP_RAW
				((uint32_t*)buffer)[0] = MAIN_FB()->w;
				((uint32_t*)buffer)[1] = MAIN_FB()->h;
				((uint32_t*)buffer)[2] = MAIN_FB()->bpp; // MAIN_FB()->w * MAIN_FB()->h * MAIN_FB()->bpp;		
			#else
				// Return 0 if FB is locked or not ready.
				((uint32_t*)buffer)[0] = 0;
				// Try to lock FB. If header size == 0 frame is not ready
				if (mutex_try_lock_alternate(&JPEG_FB()->lock, MUTEX_TID_IDE)) {
					// If header size == 0 frame is not ready
					if (JPEG_FB()->size == 0) {
						// unlock FB
						mutex_unlock(&JPEG_FB()->lock, MUTEX_TID_IDE);
					} else {
						// Return header w, h and size/bpp
						((uint32_t*)buffer)[0] = JPEG_FB()->w;
						((uint32_t*)buffer)[1] = JPEG_FB()->h;
						((uint32_t*)buffer)[2] = JPEG_FB()->size;
					}
				}
			#endif
		#endif
            cmd = USBDBG_NONE;
            break;

        case USBDBG_FRAME_DUMP:
            if (xfer_bytes < xfer_length) {
                memcpy(buffer, JPEG_FB()->pixels+xfer_bytes, length);
                xfer_bytes += length;
                if (xfer_bytes == xfer_length) {
                    cmd = USBDBG_NONE;
                    JPEG_FB()->w = 0; JPEG_FB()->h = 0; JPEG_FB()->size = 0;
                    mutex_unlock(&JPEG_FB()->lock, MUTEX_TID_IDE);
                }
            }
            break;

        case USBDBG_ARCH_STR: {
			#if 0
            unsigned int uid[3] = {
            #if (OMV_UNIQUE_ID_SIZE == 2)
                0U,
            #else
                *((unsigned int *) (OMV_UNIQUE_ID_ADDR + 8)),
            #endif
                *((unsigned int *) (OMV_UNIQUE_ID_ADDR + 4)),
                *((unsigned int *) (OMV_UNIQUE_ID_ADDR + 0)),
            };
			#endif
            snprintf((char *) buffer, 64, "%s [%s:%08X%08X%08X]",
                    OMV_ARCH_STR/*OpenMV i.MX RT1050/60 port*/, OMV_BOARD_TYPE /* M7 */,
                    0x35383236 /*0x4B4E4854*/,
                    0x3436510f /*0x564D4F20*/,
                    0x0041001E /*0x434C4C20*/);	
			
            cmd = USBDBG_NONE;
            break;
        }

        case USBDBG_SCRIPT_RUNNING: {
            uint32_t *buf = buffer;
             // rocky: though may not run yet, set running flag in case openmv ide get not running
             // flag before script get run.
            buf[0] = (uint32_t) (s_UsbDbgIsToRunScript || script_running);
            cmd = USBDBG_NONE;
            break;
        }
        default: /* error */
            break;
    }
}

extern int py_image_descriptor_from_roi(image_t *image, const char *path, rectangle_t *roi);

void usbdbg_try_run_script(void)
{
    if (!s_UsbDbgIsToRunScript)
        return;
    // Disable IDE IRQ (re-enabled by pyexec or main).
    usbdbg_set_irq_enabled(false);
    s_UsbDbgIsToRunScript = 0;
    // Clear interrupt traceback
    mp_obj_exception_clear_traceback( MP_STATE_PORT(mp_const_ide_interrupt));
    // Interrupt running REPL
    // Note: setting pendsv explicitly here because the VM is probably
    // waiting in REPL and the soft interrupt flag will not be checked.
    // PRINTF("nlr jumping to execute script\r\n");
    
    pendsv_nlr_jump_hard( MP_STATE_PORT(mp_const_ide_interrupt));
}
extern uint8_t g_isMainDotPyRunning;
void usbdbg_data_out(void *buffer, int length)
{
    switch (cmd) {
        case USBDBG_FB_ENABLE: {
            uint32_t enable = *((int32_t*)buffer);
            JPEG_FB()->enabled = enable;
            if (enable == 0) {
                // When disabling framebuffer, the IDE might still be holding FB lock.
                // If the IDE is not the current lock owner, this operation is ignored.
                mutex_unlock(&JPEG_FB()->lock, MUTEX_TID_IDE);
            }
            cmd = USBDBG_NONE;
            break;
        }

        case USBDBG_SCRIPT_EXEC:
            // check if GC is locked before allocating memory for vstr. If GC was locked
            // at least once before the script is fully uploaded xfer_bytes will be less
            // than the total length (xfer_length) and the script will Not be executed.
            if (!script_running && !gc_is_locked()) {
                vstr_add_strn(&script_buf, buffer, length);
                MP_STATE_PORT(script_buffer) = script_buf.buf;
                Hook_OnUsbDbgScriptExec();
                xfer_bytes += length;
                if (xfer_bytes == xfer_length) {
                    s_UsbDbgIsToRunScript = 1;
                    script_ready = true;

                    // Set script running flag
                    script_running = true;

                    // Disable IDE IRQ (re-enabled by pyexec or main).
                    usbdbg_set_irq_enabled(false);

                    // Clear interrupt traceback
                    mp_obj_exception_clear_traceback( MP_STATE_PORT(mp_const_ide_interrupt));

                    // Remove the BASEPRI masking (if any)
                    __set_BASEPRI(0);

                    // Interrupt running REPL
                    // Note: setting pendsv explicitly here because the VM is probably
                    // waiting in REPL and the soft interrupt flag will not be checked.
// in case main.py is running just after system reset, notify the VM to stop it.
					
                    if (g_isMainDotPyRunning)
                        pendsv_nlr_jump( MP_STATE_PORT(mp_const_ide_interrupt));

                }
            }
            break;

        case USBDBG_TEMPLATE_SAVE: {
            #if defined(IMLIB_ENABLE_IMAGE_FILE_IO)
            image_t image;
            framebuffer_init_image(&image);

            // null terminate the path
            length = (length == 64) ? 63:length;
            ((char*)buffer)[length] = 0;

            rectangle_t *roi = (rectangle_t*)buffer;
            char *path = (char*)buffer+sizeof(rectangle_t);

            imlib_save_image(&image, path, roi, 50);

            // raise a flash IRQ to flush image
            //NVIC->STIR = FLASH_IRQn;
            #endif  //IMLIB_ENABLE_IMAGE_FILE_IO
            break;
        }

        case USBDBG_DESCRIPTOR_SAVE: {
            #if defined(IMLIB_ENABLE_IMAGE_FILE_IO)\
                && defined(IMLIB_ENABLE_KEYPOINTS)
            image_t image;
            framebuffer_init_image(&image);

            // null terminate the path
            length = (length == 64) ? 63:length;
            ((char*)buffer)[length] = 0;

            rectangle_t *roi = (rectangle_t*)buffer;
            char *path = (char*)buffer+sizeof(rectangle_t);

            py_image_descriptor_from_roi(&image, path, roi);
            #endif  //IMLIB_ENABLE_IMAGE_FILE_IO && IMLIB_ENABLE_KEYPOINTS
            cmd = USBDBG_NONE;
            break;
        }

        case USBDBG_ATTR_WRITE: {
            /* write sensor attribute */
            int32_t attr= *((int32_t*)buffer);
            int32_t val = *((int32_t*)buffer+1);
			struct rt_camera_device *sensor = imxrt_camera_device_find(SENSOR_NAME);
            switch (attr) {
               case ATTR_CONTRAST:
					sensor->ops->camera_control(sensor, RT_DRV_CAM_CMD_SET_CONTRAST, val);
                    break;
                case ATTR_BRIGHTNESS:
					sensor->ops->camera_control(sensor, RT_DRV_CAM_CMD_SETBRIGHTNESS, val);
                    break;
                case ATTR_SATURATION:
					sensor->ops->camera_control(sensor, RT_DRV_CAM_CMD_SETSATURATION, val);
                    break;
                case ATTR_GAINCEILING:
					sensor->ops->camera_control(sensor, RT_DRV_CAM_CMD_SET_GAINCEILING, val);
                    break;
                default:
                    break;
            }
            cmd = USBDBG_NONE;
            break;
        }
        default: /* error */
            break;
    }
}

void usbdbg_stop_script(void) {
    // Set script running flag
    
    script_running = false;
    script_ready = false;


    // interrupt running code by raising an exception
    // pendsv_kbd_intr();
    //mp_obj_exception_clear_traceback(mp_const_ide_interrupt);
     
}


void usbdbg_control(void *buffer, uint8_t request, uint32_t length)
{
    cmd = (enum usbdbg_cmd) request;
    switch (cmd) {
        case USBDBG_FW_VERSION:
            xfer_bytes = 0;
            xfer_length = length;
            break;

        case USBDBG_FRAME_SIZE:
            xfer_bytes = 0;
            xfer_length = length;
            break;

        case USBDBG_FRAME_DUMP:
            xfer_bytes = 0;
            xfer_length = length;
            break;

        case USBDBG_ARCH_STR:
            xfer_bytes = 0;
            xfer_length = length;
            break;

        case USBDBG_SCRIPT_EXEC:
            xfer_bytes = 0;
            xfer_length = length;
            vstr_reset(&script_buf);
            break;

        case USBDBG_SCRIPT_STOP:
            if (script_running) {
                // Set script running flag
                script_running = false;

                // Disable IDE IRQ (re-enabled by pyexec or main).
                usbdbg_set_irq_enabled(false);
				usbdbg_stop_script();
                // interrupt running code by raising an exception
                mp_obj_exception_clear_traceback( MP_STATE_PORT(mp_const_ide_interrupt));

                // Remove the BASEPRI masking (if any)
                __set_BASEPRI(0);

                pendsv_nlr_jump( MP_STATE_PORT(mp_const_ide_interrupt));
            }
            cmd = USBDBG_NONE;
            break;

        case USBDBG_SCRIPT_SAVE:
            // TODO: save running script
            cmd = USBDBG_NONE;
            break;

        case USBDBG_SCRIPT_RUNNING:
            xfer_bytes = 0;
            xfer_length =length;
            break;

        case USBDBG_TEMPLATE_SAVE:
        case USBDBG_DESCRIPTOR_SAVE:
            /* save template */
            xfer_bytes = 0;
            xfer_length =length;
            break;

        case USBDBG_ATTR_WRITE:
            xfer_bytes = 0;
            xfer_length =length;
            break;

        case USBDBG_SYS_RESET:
            NVIC_SystemReset();
            break;

        case USBDBG_SYS_RESET_TO_BL:{
            #if defined(MICROPY_RESET_TO_BOOTLOADER)
            MICROPY_RESET_TO_BOOTLOADER();
            #else
            NVIC_SystemReset();
            #endif
            break;
        }

        case USBDBG_FB_ENABLE: {
            xfer_bytes = 0;
            xfer_length = length;
            break;
        }

        case USBDBG_TX_BUF:
        case USBDBG_TX_BUF_LEN:
            xfer_bytes = 0;
            xfer_length = length;
            break;

        case USBDBG_SENSOR_ID:
            xfer_bytes = 0;
            xfer_length = length;
            break;

        default: /* error */
            cmd = USBDBG_NONE;
            break;
    }
}

void usbdbg_connect(void)
{
	// slow down the sensor to avoid tearing effect, when executing from FlexSPI
	// sensor_set_framerate(2<<9 | 2<<11);
}
void usbdbg_disconnect(void) {
	#ifndef OMV_MPY_ONLY
	JPEG_FB()->enabled = 0;
	mutex_unlock(&JPEG_FB()->lock, MUTEX_TID_IDE);
	#endif
	// sensor_set_framerate(2<<9 | 1<<11);
}


