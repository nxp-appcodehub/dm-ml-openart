/*
 * This file is part of the OpenMV project.
 *
 * Copyright (c) 2013-2019 Ibrahim Abdelkader <iabdalkader@openmv.io>
 * Copyright (c) 2013-2019 Kwabena W. Agyeman <kwagyeman@openmv.io>
 *
 * This work is licensed under the MIT license, see the file LICENSE for details.
 *
 * main function.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <rtthread.h>
#include "dfs_fs.h"
#include <py/compile.h>
#include <py/runtime.h>
#include <py/repl.h>
#include <py/gc.h>
#include <py/mperrno.h>
#include <py/stackctrl.h>
#include <py/frozenmod.h>
#include <shared/readline/readline.h>
#include <shared/runtime/pyexec.h>
#include "mpgetcharport.h"
#include "mpputsnport.h"
#include "drv_usb_omv.h"
#include "framebuffer.h"
#include "pendsv.h"
#include "omv_boardconfig.h"
#include "usbdbg.h"

volatile uint8_t g_isMainDotPyRunning;
#if defined(__CC_ARM) || defined (__CLANG_ARM)
	#define STACK_SIZE	(0x2000)
	extern unsigned int Image$$RTT_MPY_THREAD_STACK$$Base;
	extern unsigned int Image$$MPY_HEAP_START$$Base;
	uint32_t _thread_stack_start = (uint32_t) &Image$$RTT_MPY_THREAD_STACK$$Base;
	uint32_t _heap_start = (uint32_t) &Image$$MPY_HEAP_START$$Base;
#elif defined(__ICCARM__)
	extern unsigned int RTT_MPY_THREAD_STACK$$Limit[];
	uint32_t _thread_stack_start = (uint32_t)RTT_MPY_THREAD_STACK$$Limit;	// we put HEAP at the last of DATA region, so we can assume mpy heap start here
#else
	extern int RTT_MPY_THREAD_STACK;
	extern int MPY_HEAP_START;
	uint32_t _thread_stack_start = (uint32_t) &RTT_MPY_THREAD_STACK;
	uint32_t _heap_start = (uint32_t) &MPY_HEAP_START;

#endif

__WEAK bool usbdbg_script_ready()
{
	return false;
}

__WEAK void camera_clear_by_omv_ide()
{

}
#if defined(MICROPYTHON_USING_LVGL)
#include "../../lv_binding_micropython/lvgl/lvgl.h"
#endif

void post_processing(bool exception){
	#if defined(SOC_MIMXRT1176DVMMA) && defined(RT_USING_LCD)
		extern void reset_displaymix();
		 /*
		 * Reset the displaymix, otherwise during debugging, the
		 * debugger may not reset the display, then the behavior
		 * is not right.
		 */
		reset_displaymix();
	
		// inform the LCD flush driver, send it a sinal
		extern bool ide_signal;
		ide_signal = true;
	#endif
		camera_clear_by_omv_ide();	
		//NVIC_ClearPendingIRQ(LCDIF1_IRQn);
	#if defined(MICROPYTHON_USING_LVGL)
	#if defined(__CC_ARM) || defined (__CLANG_ARM)
		extern char Load$$ER_LVGL_ZI_DATA$$Base[];
		extern char Load$$ER_LVGL_ZI_DATA$$Limit[];
		extern char Image$$ER_LVGL_ZI_DATA$$Base[];
		extern char Image$$ER_LVGL_ZI_DATA$$ZI$$Base[];
		extern char Image$$ER_LVGL_ZI_DATA$$ZI$$Limit[]; // must use the xx$$ZI$$Limit, general, the xx$$Limit, means the end of the non-zi area
		
		#define LVGL_ZI Image$$ER_LVGL_ZI_DATA$$ZI$$Base 
		#define LVGL_ZI_END Image$$ER_LVGL_ZI_DATA$$ZI$$Limit
		#define LVGL_ZI_LEN (LVGL_ZI_END - LVGL_ZI)
	
		#define LVGL_RW_SRC Load$$ER_LVGL_ZI_DATA$$Base 
		#define LVGL_RW_DST Image$$ER_LVGL_ZI_DATA$$Base
		#define LVGL_RW_LEN (Load$$ER_LVGL_ZI_DATA$$Limit - LVGL_RW_SRC)
	#else
		extern char _lv_data_start;
		extern char _lv_data_end;
		extern char __RAMCODE_ROM_END;

		extern char _lv_bss_start;
		extern char _lv_bss_end;

		#define LVGL_ZI &_lv_bss_start 
		#define LVGL_ZI_END &_lv_bss_end
		#define LVGL_ZI_LEN (LVGL_ZI_END - LVGL_ZI)
	
		#define LVGL_RW_SRC __RAMCODE_ROM_END 
		#define LVGL_RW_DST _lv_data_start
		#define LVGL_RW_LEN (_lv_data_end - _lv_data_start)
	#endif	
		// when running the lvgl, and an except occur, we need to release their resources 
		// do this before memset, in case obj_created will be zero inited after
		if(exception){
			extern lv_obj_t *obj_created;
			if(obj_created) {
				lv_obj_del(obj_created);
			}
			// deinit the lvgl
			lv_deinit();
			
		}
		
		#if defined(__CC_ARM) || defined (__CLANG_ARM) || defined (__GNUC__)
		// must zero-init the ZI data inside the LVGL lib, in case a wrong initial value, 
		// default, the lvgl's ZI data only init once by scatter loader after boot, when 
		// we do soft reset, need to do this manually. both the RW data.
		// 1. static theme_styles_t * styles; in lv_theme_material, if not NULL, will be 
		// re-use, NULL will allocate a new one
		// 2. gc roots, memset by the gc_deinit(), remember that
		memset(LVGL_ZI, 0x00, LVGL_ZI_LEN);
		memcpy(LVGL_RW_DST, LVGL_RW_SRC, LVGL_RW_LEN);
		#endif
		// free the mem inside the lv_helper_nxp.c
		extern void mp_free_buf_out();
		mp_free_buf_out();
		
	#endif
		fb_alloc_init0();
		//a rude way to do the env-clean after omv quit
		// NVIC_SystemReset();
		
		// deinit all the timer 
		if(exception){
			#define TO_STRING(x) #x
			#define TIMER_NAME(x) TO_STRING(gpt##x)
			#define TIMER_MAX (2)
			for(int i=0;i<TIMER_MAX;i++){
				char* timer_name;
				switch(i){
					case 0:
						timer_name = TIMER_NAME(1);
						break;
					case 1:
						timer_name = TIMER_NAME(2);
						break;
					default:
						break;
				}
				rt_device_t timer = rt_device_find(timer_name);
				// if opened then close
				if(timer && timer->open_flag){
					rt_device_close(timer);
				}
			}
		}
}
#define THREAD_STACK_NO_SYNC   4096
#define THREAD_STACK_WITH_SYNC 8192

static void *stack_top = RT_NULL;
static char *heap = RT_NULL;
extern void pin_init(void);

void print_execute(void * ret_val)
{
	if (ret_val == NULL)
		return;
	
	if (mp_obj_is_subclass_fast(MP_OBJ_FROM_PTR(((mp_obj_base_t *)ret_val)->type), MP_OBJ_FROM_PTR(&mp_type_SystemExit))) {
            // at the moment, the value of SystemExit is unused
        } else {
            mp_obj_print_exception(&mp_plat_print, MP_OBJ_FROM_PTR(ret_val));
        }
}

void omv_main_thread_entry(void *parameter)
{
	int stack_dummy;
    int stack_size_check;
    stack_top = (void *)&stack_dummy;

	rt_kprintf("Start OpenMV Thread\n");

omv_restart:	
	mp_getchar_init();
    mp_putsn_init();
#if defined(MICROPYTHON_USING_FILE_SYNC_VIA_IDE)
    stack_size_check = THREAD_STACK_WITH_SYNC;
#else
    stack_size_check = THREAD_STACK_NO_SYNC;
#endif	
	
	if (rt_thread_self()->stack_size < stack_size_check) 
    {
        mp_printf(&mp_plat_print, "The stack (%.*s) size for executing MicroPython must be >= %d\n", RT_NAME_MAX, rt_thread_self()->name, stack_size_check);
    }
	
#if MICROPY_PY_THREAD
    mp_thread_init(rt_thread_self()->stack_addr, ((rt_uint32_t)stack_top - (rt_uint32_t)rt_thread_self()->stack_addr) / 4);
#endif

    mp_stack_set_top(stack_top);
    // Make MicroPython's stack limit somewhat smaller than full stack available
    mp_stack_set_limit(rt_thread_self()->stack_size - 1024);

	gc_init((void*)_heap_start, (void*)(_heap_start + MICROPY_HEAP_SIZE));
	
	/* MicroPython initialization */
    mp_init();

    /* system path initialization */
    mp_obj_list_init(mp_sys_path, 0);
    mp_obj_list_append(mp_sys_path, MP_OBJ_NEW_QSTR(MP_QSTR_)); // current dir (or base dir of the script)
    mp_obj_list_append(mp_sys_path, mp_obj_new_str(MICROPY_PY_PATH_FIRST, strlen(MICROPY_PY_PATH_FIRST)));
    mp_obj_list_append(mp_sys_path, mp_obj_new_str(MICROPY_PY_PATH_SECOND, strlen(MICROPY_PY_PATH_SECOND)));
    
	// delay to check sdcard
	int time_delay = 50;
	while(time_delay--) {rt_thread_delay(10);  if(mmcsd_is_present()) break;}
	if(rt_device_find("sd0") != RT_NULL)
	{
		mp_obj_list_append(mp_sys_path, mp_obj_new_str("/sd", strlen("/sd")));
	}
#ifdef BSP_USING_SPIFLASH_PARTITION	
	else
	{
		mp_obj_list_append(mp_sys_path, mp_obj_new_str("/flash", strlen("/flash")));
	}
#endif	
	mp_obj_list_init(mp_sys_argv, 0);
	imlib_init_all();
    readline_init0();
	pin_init();
	ticks_init();
	fb_alloc_init0();
    file_buffer_init0();
	framebuffer_init0();
	usbdbg_init();
	
	//run cmm_load.py
	char *cmm_path = "/sd/cmm_load.py";
	mp_import_stat_t stat = mp_import_stat(cmm_path);
	if (stat != MP_IMPORT_STAT_FILE)
	{
		cmm_path = "/flash/cmm_load.py";
		stat = mp_import_stat(cmm_path);
	}
	nlr_buf_t nlr;
	nlr.ret_val = NULL;
    if (stat == MP_IMPORT_STAT_FILE) {
        
		mp_printf(&mp_plat_print, "Found and execute %s!\r\n",cmm_path);
		int exit_return = nlr_push(&nlr);
        if (exit_return == 0) {
            int ret = pyexec_file(cmm_path,true);
            if (ret & PYEXEC_FORCED_EXIT) {
				mp_printf(&mp_plat_print, "cmm_load.py error!\r\n");
                ret = 1;
            }
            if (!ret) {
            }
            nlr_pop();
        }
        else {           
        }
    }
	
	print_execute(nlr.ret_val);
	
#ifndef MICROPYTHON_USING_UOS
#error "MICROPYTHON_USING_UOS not enable, mp_import_stat() always return failed"
#endif	
	// rocky: pyb's main uses different method to access file system from omv
	char *main_path = "/sd/main.py";
	stat = mp_import_stat(main_path);
	if (stat != MP_IMPORT_STAT_FILE)
	{
		main_path = "/flash/main.py";
		stat = mp_import_stat(main_path);
	}
	nlr.ret_val = NULL;
	if (stat == MP_IMPORT_STAT_FILE) {
		
		if (nlr_push(&nlr) == 0) {
			g_isMainDotPyRunning = 1;
			mp_printf(&mp_plat_print, "Found and execute %s!\r\n",main_path);
			int ret = pyexec_file(main_path,true);
			g_isMainDotPyRunning = 0;
			if (ret & PYEXEC_FORCED_EXIT) {
				ret = 1;
			}
			if (!ret) {
				
			}
			nlr_pop();
		}
		else {
			g_isMainDotPyRunning = 0;	
			fb_free_all();      
			fb_alloc_init0();
			post_processing(true);
		}
		
		mp_printf(&mp_plat_print, "Exit from main.py!\r\n");
	}
	print_execute(nlr.ret_val);
			
RunREPL:
	
    while (!usbdbg_script_ready()) {
        nlr.ret_val = NULL;

        if (nlr_push(&nlr) == 0) {
		
            // enable IDE interrupt
            usbdbg_set_irq_enabled(true);
			g_isMainDotPyRunning = 1;
            // run REPL
            if (pyexec_mode_kind == PYEXEC_MODE_RAW_REPL) {
                if (pyexec_raw_repl() != 0) {
                    break;
                }
            } else {
                if (pyexec_friendly_repl() != 0) {
					
                    break;
                }
            }

            nlr_pop();
        }
    }
	mp_printf(&mp_plat_print, "Exit from repy!\r\n");
    if (usbdbg_script_ready()) {
        nlr.ret_val = NULL;
		mp_printf(&mp_plat_print, "script ready!\r\n");
        // execute the script
        if (nlr_push(&nlr) == 0) {
	
			usbdbg_set_irq_enabled(true);
			pyexec_str((vstr_t *)usbdbg_get_script(),true);

            nlr_pop();
			usbdbg_set_script_running(false);
			post_processing(false);
        } else {
            mp_obj_print_exception(&mp_plat_print, (mp_obj_t)nlr.ret_val);
            usbdbg_stop_script();
			post_processing(true);
			
            //goto RunREPL;   // rocky: script is stopped, waiting for next script
        }

    }

	usbdbg_set_irq_enabled(true);
	gc_sweep_all();
    mp_deinit();
#if MICROPY_PY_THREAD
    mp_thread_deinit();
#endif
    rt_free(heap);
    mp_putsn_deinit();
    mp_getchar_deinit();
	rt_thread_mdelay(500);
	goto omv_restart;
	
	return;
}

void _sys_exit(int return_code)
{
	NVIC_SystemReset();
}	


struct rt_thread omv_thread;
void omv_main()
{
	rt_err_t result;
    
	pendsv_init();
    result = rt_thread_init(&omv_thread, "omv_main", omv_main_thread_entry, RT_NULL,
                            (void*)_thread_stack_start, NXP_MICROPYTHON_THREAD_STACK_SIZE, 6, 20);
    RT_ASSERT(result == RT_EOK);
	rt_thread_startup(&omv_thread);
}

static void omv(uint8_t argc, char **argv) {
    omv_main();
}
#if 1 //def NXP_OMV_AUTO_START
INIT_APP_EXPORT(omv_main);
#endif
MSH_CMD_EXPORT(omv, OpenMV: `execute python script);

#if defined (__GNUC__)
void __cxa_pure_virtual(void)
{
    rt_kprintf("Illegal to call a pure virtual function.\n");
}
#endif
