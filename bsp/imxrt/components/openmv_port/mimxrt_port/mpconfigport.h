/*
 * This file is part of the MicroPython project, http://micropython.org/
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019 Damien P. George
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

// Options controlling how MicroPython is built, overriding defaults in py/mpconfig.h

// Board specific definitions
#include "rtthread.h"
#include "mpconfigboard.h"
#include "fsl_common.h"
#ifndef SEEK_SET
#define	SEEK_SET	0	/* set file offset to offset */
#endif
#ifndef SEEK_CUR
#define	SEEK_CUR	1	/* set file offset to current plus offset */
#endif
#ifndef SEEK_END
#define	SEEK_END	2	/* set file offset to EOF plus offset */
#endif

#ifndef errno
#define errno    (-1)
#endif
uint32_t trng_random_u32(void);

//#define MICROPY_DEBUG_VERBOSE 1
//#define MICROPY_DEBUG_PRINTERS      (1)


//ulab
#ifdef CONFIG_ULAB_HAS_DTYPE_OBJECT
#define ULAB_HAS_DTYPE_OBJECT	(1)
#else
#define ULAB_HAS_DTYPE_OBJECT	(0)
#endif
#ifdef CONFIG_NDARRAY_HAS_DTYPE
#define NDARRAY_HAS_DTYPE	(1)
#else
#define NDARRAY_HAS_DTYPE	(0)
#endif
#ifdef CONFIG_ULAB_HAS_SCIPY
#define ULAB_HAS_SCIPY	(1)
#else
#define ULAB_HAS_SCIPY	(0)
#endif
#ifdef CONFIG_ULAB_HAS_UTILS_MODULE
#define ULAB_HAS_UTILS_MODULE	(1)
#else
#define ULAB_HAS_UTILS_MODULE	(0)
#endif

#define MICROPY_USE_INTERNAL_ERRNO    (1)
#include "py/mperrno.h"
#define ULAB_MAX_DIMS   4

#if defined(PKG_MICROPYTHON_HEAP_SIZE)
#define MICROPY_HEAP_SIZE PKG_MICROPYTHON_HEAP_SIZE
#else
#define MICROPY_HEAP_SIZE (8 * 1024)
#endif

// Memory allocation policies
#define MICROPY_GC_STACK_ENTRY_TYPE         uint16_t
#define MICROPY_GC_ALLOC_THRESHOLD          (0)
#define MICROPY_ALLOC_PARSE_CHUNK_INIT      (32)
#define MICROPY_ALLOC_PATH_MAX              (256)

// MicroPython emitters
#define MICROPY_PERSISTENT_CODE_LOAD        (1)
#define MICROPY_EMIT_THUMB                  (1)
#define MICROPY_EMIT_INLINE_THUMB           (1)

// You can disable the built-in MicroPython compiler by setting the following
// config option to 0.  If you do this then you won't get a REPL prompt, but you
// will still be able to execute pre-compiled scripts, compiled with mpy-cross.
#define MICROPY_ENABLE_COMPILER     (1)
#define MICROPY_COMP_MODULE_CONST   (1)
#define MICROPY_COMP_TRIPLE_TUPLE_ASSIGN (1)
#define MICROPY_COMP_RETURN_IF_EXPR (1)
#ifndef MICROPY_QSTR_BYTES_IN_HASH
#define MICROPY_QSTR_BYTES_IN_HASH  (2)
#endif

// Optimisations
#ifndef MICROPY_OPT_COMPUTED_GOTO
#ifdef NXP_USING_OPENMV
#define MICROPY_OPT_COMPUTED_GOTO (1)
#else
#define MICROPY_OPT_COMPUTED_GOTO (0)
#endif
#endif

#define MICROPY_EMIT_X64            (0)

#define MICROPY_COMP_CONST          (0)
#define MICROPY_COMP_DOUBLE_TUPLE_ASSIGN (0)
#define MICROPY_MEM_STATS           (0)


#define MICROPY_OPT_LOAD_ATTR_FAST_PATH     (1)
#define MICROPY_OPT_MAP_LOOKUP_CACHE        (1)

// Python internal features
#define MICROPY_READER_POSIX                  (1)
#define MICROPY_ENABLE_GC                   (1)
#define MICROPY_ENABLE_FINALISER            (1)
#define MICROPY_STACK_CHECK                 (1)
#define MICROPY_ENABLE_EMERGENCY_EXCEPTION_BUF  (1)
#define MICROPY_KBD_EXCEPTION               (1)
#define MICROPY_HELPER_REPL                 (1)
#define MICROPY_REPL_AUTO_INDENT            (1)
#define MICROPY_LONGINT_IMPL                (MICROPY_LONGINT_IMPL_MPZ)
#undef MICROPY_FLOAT_IMPL
#define MICROPY_FLOAT_IMPL          (MICROPY_FLOAT_IMPL_FLOAT)

#define MICROPY_BUILTIN_METHOD_CHECK_SELF_ARG (1)

#define MICROPY_ENABLE_SOURCE_LINE          (1)
#define MICROPY_STREAMS_NON_BLOCK           (1)
#define MICROPY_MODULE_BUILTIN_INIT         (1)
#define MICROPY_MODULE_WEAK_LINKS           (1)
#define MICROPY_CAN_OVERRIDE_BUILTINS       (1)
#define MICROPY_ENABLE_SCHEDULER            (1)
#define MICROPY_SCHEDULER_DEPTH             (8)
#define MICROPY_VFS                         (1)
#define MICROPY_VFS_POSIX                   (1)
#define MICROPY_PY_UOS_VFS                  (1)
#define MICROPY_MODULE_FROZEN_MPY           (1)
#define MODULE_ULAB_ENABLED                 (1)
//#define MICROPY_QSTR_EXTRA_POOL             mp_qstr_frozen_const_pool

// Control over Python builtins
#define MICROPY_PY_FUNCTION_ATTRS           (1)
#define MICROPY_PY_DESCRIPTORS              (1)
#define MICROPY_PY_DELATTR_SETATTR          (1)
#define MICROPY_PY_FSTRINGS                 (1)
#define MICROPY_PY_BUILTINS_STR_UNICODE     (1)
#define MICROPY_PY_BUILTINS_STR_CENTER      (1)
#define MICROPY_PY_BUILTINS_STR_PARTITION   (1)
#define MICROPY_PY_BUILTINS_STR_SPLITLINES  (1)
#define MICROPY_PY_BUILTINS_BYTEARRAY (1)
#define MICROPY_PY_BUILTINS_MEMORYVIEW      (1)
#define MICROPY_PY_BUILTINS_SLICE_ATTRS     (1)
#define MICROPY_PY_BUILTINS_SLICE_INDICES   (1)
#define MICROPY_PY_BUILTINS_FROZENSET       (1)
#define MICROPY_PY_BUILTINS_FLOAT           (1)
#define MICROPY_PY_BUILTINS_ROUND_INT       (1)
#define MICROPY_PY_ALL_SPECIAL_METHODS      (1)
#define MICROPY_PY_REVERSE_SPECIAL_METHODS  (1)
#define MICROPY_PY_BUILTINS_COMPILE         (1)
#define MICROPY_PY_BUILTINS_INPUT           (1)
#define MICROPY_PY_BUILTINS_POW3            (1)
#define MICROPY_PY_BUILTINS_HELP            (1)
#define MICROPY_PY_BUILTINS_HELP_MODULES    (1)
#define MICROPY_PY_BUILTINS_HELP_TEXT       mimxrt_help_text
#define MICROPY_PY_MICROPYTHON_MEM_INFO     (1)
#define MICROPY_PY_ARRAY_SLICE_ASSIGN       (1)
#define MICROPY_PY_COLLECTIONS_DEQUE        (1)
#define MICROPY_PY_COLLECTIONS_ORDEREDDICT  (1)
#define MICROPY_PY_MATH_SPECIAL_FUNCTIONS   (1)
#define MICROPY_PY_MATH_FACTORIAL           (1)
#define MICROPY_PY_MATH_ISCLOSE             (1)
#define MICROPY_PY_CMATH                    (1)
#define MICROPY_PY_SYS                      (1)
#define MICROPY_PY_IO_IOBASE                (1)
#define MICROPY_PY_IO_FILEIO                (1)
#define MICROPY_PY_SYS_MAXSIZE              (1)
#define MICROPY_PY_SYS_PLATFORM             "mimxrt"
#define MICROPY_PY_SYS_STDFILES             (1)
#define MICROPY_PY_SYS_STDIO_BUFFER         (1)
#define MICROPY_PY_UERRNO                   (1)
#define MICROPY_PY_PATH_FIRST               "/libs/mpy"
#define MICROPY_PY_PATH_SECOND              "/scripts"
#define MICROPY_MAIN_PY_PATH                "/main.py"

#define MICROPY_PY_SD_PATH				    "/sd"
#define MICROPY_PY_FLASH_PATH				"/flash"
#define MICROPY_MODULE_FROZEN               (0)
// Extended modules
#define MICROPY_EPOCH_IS_1970               (1)
#define MICROPY_PY_UASYNCIO                 (1)
#define MICROPY_PY_UCTYPES                  (1)
#define MICROPY_PY_UZLIB                    (1)
#define MICROPY_PY_UJSON                    (1)
#define MICROPY_PY_URE                      (1)
#define MICROPY_PY_URE_MATCH_GROUPS         (1)
#define MICROPY_PY_URE_MATCH_SPAN_START_END (1)
#define MICROPY_PY_URE_SUB                  (1)
#define MICROPY_PY_USSL_FINALISER           (MICROPY_PY_USSL)
#define MICROPY_PY_UHASHLIB                 (1)
#define MICROPY_PY_UBINASCII                (1)
#define MICROPY_PY_UBINASCII_CRC32          (1)
#define MICROPY_PY_UTIME_MP_HAL             (1)
#define MICROPY_PY_OS_DUPTERM               (3)
#define MICROPY_PY_URANDOM                  (1)
#define MICROPY_PY_URANDOM_EXTRA_FUNCS      (1)
#define MICROPY_PY_URANDOM_SEED_INIT_FUNC   (trng_random_u32())
#define MICROPY_PY_USELECT                  (1)
#define MICROPY_PY_MACHINE                  (1)
#define MICROPY_PY_MACHINE_PIN_MAKE_NEW     mp_pin_make_new
#define MICROPY_PY_MACHINE_BITSTREAM        (1)
#define MICROPY_PY_MACHINE_PULSE            (1)
#define MICROPY_PY_MACHINE_PWM              (1)
#define MICROPY_PY_MACHINE_I2C              (1)
#define MICROPY_PY_MACHINE_I2C_MAKE_NEW machine_hard_i2c_make_new
#define MICROPY_PY_MACHINE_SOFTI2C          (1)
#define MICROPY_PY_MACHINE_SPI              (1)
#define MICROPY_PY_MACHINE_SPI_MAKE_NEW machine_hard_spi_make_new

#define MICROPY_PY_MACHINE_SOFTSPI          (1)
#define MICROPY_PY_FRAMEBUF                 (0)
#define MICROPY_PY_ONEWIRE                  (0)
#define MICROPY_PY_UPLATFORM                (0)
#define MICROPY_PY_SENSOR                   (1)
#define MICROPY_PY_UTIME                   	(1)
// fatfs configuration used in ffconf.h
#define MICROPY_FATFS_ENABLE_LFN            (1)
#define MICROPY_FATFS_RPATH                 (2)
#define MICROPY_FATFS_MAX_SS                (4096)
#define MICROPY_FATFS_LFN_CODE_PAGE         437 /* 1=SFN/ANSI 437=LFN/U.S.(OEM) */
#define USE_DEVICE_MODE
// If MICROPY_PY_LWIP is defined, add network support
#if MICROPY_PY_LWIP

#define MICROPY_PY_NETWORK                  (1)
#define MICROPY_PY_USOCKET                  (1)
#define MICROPY_PY_UWEBSOCKET               (1)
#define MICROPY_PY_WEBREPL                  (1)
#define MICROPY_PY_UHASHLIB_SHA1            (1)
#define MICROPY_PY_LWIP_SOCK_RAW            (1)
#define MICROPY_HW_ETH_MDC                  (1)

// Prevent the "LWIP task" from running.
#define MICROPY_PY_LWIP_ENTER   MICROPY_PY_PENDSV_ENTER
#define MICROPY_PY_LWIP_REENTER MICROPY_PY_PENDSV_REENTER
#define MICROPY_PY_LWIP_EXIT    MICROPY_PY_PENDSV_EXIT

#endif
/*****************************************************************************/
/* Hardware Module                                                           */
#ifdef MICROPYTHON_USING_MACHINE_UART
#define MICROPY_PY_MACHINE_UART      (1)
#endif

#ifdef MICROPYTHON_USING_MACHINE_ADC
#define MICROPY_PY_MACHINE_ADC       (1)
#endif

#ifdef MICROPYTHON_USING_MACHINE_PWM
#define MICROPY_PY_MACHINE_PWM       (1)
#endif

#ifdef MICROPYTHON_USING_MACHINE_LCD
#define MICROPY_PY_MACHINE_LCD       (1)
#endif

#ifdef MICROPYTHON_USING_MACHINE_RTC
#define MICROPY_PY_MACHINE_RTC       (1)
#endif

#ifdef MICROPYTHON_USING_MACHINE_WDT
#define MICROPY_PY_MACHINE_WDT       (1)
#endif

#ifdef MICROPYTHON_USING_MACHINE_TIMER
#define MICROPY_PY_MACHINE_TIMER     (1)
#endif

// For regular code that wants to prevent "background tasks" from running.
// These background tasks (LWIP, Bluetooth) run in PENDSV context.
// TODO: Check for the settings of the STM32 port in irq.h
#define MICROPY_PY_PENDSV_ENTER   uint32_t atomic_state = disable_irq();
#define MICROPY_PY_PENDSV_REENTER atomic_state = disable_irq();
#define MICROPY_PY_PENDSV_EXIT    enable_irq(atomic_state);

// Use VfsLfs2's types for fileio/textio
#define mp_type_fileio mp_type_vfs_posix_fileio
#define mp_type_textio mp_type_vfs_posix_textio

// Use VFS's functions for import stat and builtin open
#define mp_import_stat mp_posix_import_stat


// Hooks to add builtins

__attribute__((always_inline)) static inline void enable_irq(uint32_t state) {
    __set_PRIMASK(state);
}

__attribute__((always_inline)) static inline uint32_t disable_irq(void) {
    uint32_t state = __get_PRIMASK();
    __disable_irq();
    return state;
}

static inline uint32_t raise_irq_pri(uint32_t pri) {
    uint32_t basepri = __get_BASEPRI();
    // If non-zero, the processor does not process any exception with a
    // priority value greater than or equal to BASEPRI.
    // When writing to BASEPRI_MAX the write goes to BASEPRI only if either:
    //   - Rn is non-zero and the current BASEPRI value is 0
    //   - Rn is non-zero and less than the current BASEPRI value
    pri <<= (8 - __NVIC_PRIO_BITS);
    __ASM volatile ("msr basepri_max, %0" : : "r" (pri) : "memory");
    return basepri;
}

// "basepri" should be the value returned from raise_irq_pri
static inline void restore_irq_pri(uint32_t basepri) {
    __set_BASEPRI(basepri);
}

#define MICROPY_BEGIN_ATOMIC_SECTION()     disable_irq()
#define MICROPY_END_ATOMIC_SECTION(state)  enable_irq(state)




extern const struct _mp_obj_module_t mp_module_machine;
extern const struct _mp_obj_module_t mp_module_mimxrt;
extern const struct _mp_obj_module_t mp_module_uos;
extern const struct _mp_obj_module_t mp_module_usocket;
extern const struct _mp_obj_module_t mp_module_network;
extern const struct _mp_obj_module_t pyb_module;
extern const struct _mp_obj_fun_builtin_fixed_t machine_soft_reset_obj;
extern const struct _mp_obj_module_t mp_module_time;

#define MICROPY_PORT_BUILTINS \
    { MP_ROM_QSTR(MP_QSTR_open), MP_ROM_PTR(&mp_builtin_open_obj) },\
    { MP_ROM_QSTR(MP_QSTR_exit), MP_ROM_PTR(&machine_soft_reset_obj) }, \
    { MP_ROM_QSTR(MP_QSTR_quit), MP_ROM_PTR(&machine_soft_reset_obj) }, \

#if MICROPY_PY_NETWORK
#define NETWORK_BUILTIN_MODULE              { MP_ROM_QSTR(MP_QSTR_network), MP_ROM_PTR(&mp_module_network) },
#else
#define NETWORK_BUILTIN_MODULE
#endif

#if MICROPY_PY_USOCKET && MICROPY_PY_LWIP
// usocket implementation provided by lwIP
#define SOCKET_BUILTIN_MODULE               { MP_ROM_QSTR(MP_QSTR_usocket), MP_ROM_PTR(&mp_module_lwip) },
#elif MICROPY_PY_USOCKET
// usocket implementation provided by skeleton wrapper
#define SOCKET_BUILTIN_MODULE               { MP_ROM_QSTR(MP_QSTR_usocket), MP_ROM_PTR(&mp_module_usocket) },
#else
// no usocket module
#define SOCKET_BUILTIN_MODULE
#endif

#if MICROPY_SSL_MBEDTLS
#define MICROPY_PORT_ROOT_POINTER_MBEDTLS void **mbedtls_memory;
#else
#define MICROPY_PORT_ROOT_POINTER_MBEDTLS
#endif

#if defined(MICROPY_HW_ETH_MDC)
extern const struct _mp_obj_type_t network_lan_type;
#define MICROPY_HW_NIC_ETH                  { MP_ROM_QSTR(MP_QSTR_LAN), MP_ROM_PTR(&network_lan_type) },
#else
#define MICROPY_HW_NIC_ETH
#endif

#ifdef MICROPYTHON_USING_LVGL
#ifndef MICROPY_INCLUDED_PY_MPSTATE_H
#define MICROPY_INCLUDED_PY_MPSTATE_H
#include "../lv_binding_micropython/lvgl/src/lv_misc/lv_gc.h"  // very important to include the lv_gc.h like this, in case of inter-
#undef MICROPY_INCLUDED_PY_MPSTATE_H 
#else 
#include "../lv_binding_micropython/lvgl/src/lv_misc/lv_gc.h"
#endif

extern const struct _mp_obj_module_t mp_module_lodepng;
extern const struct _mp_obj_module_t mp_module_lvgl;
extern const struct _mp_obj_module_t mp_module_lvgl_helper;

#define MICROPY_PY_LVGL_DEF \
	{ MP_OBJ_NEW_QSTR(MP_QSTR_lvgl), MP_ROM_PTR(&mp_module_lvgl)} , \
	{ MP_OBJ_NEW_QSTR(MP_QSTR_lvgl_helper), MP_ROM_PTR(&mp_module_lvgl_helper)}, \
    { MP_OBJ_NEW_QSTR(MP_QSTR_lodepng), MP_ROM_PTR(&mp_module_lodepng)}, 
    
#define LVGL_ROOT_DATA \
	LV_ROOTS \
	void *mp_lv_user_data; \
	
#else
#define MICROPY_PY_LVGL_DEF
#define LVGL_ROOT_DATA
#endif

#if MICROPY_PY_RTTHREAD
#define RTTHREAD_PORT_BUILTIN_MODULES { MP_ROM_QSTR(MP_QSTR_rtthread), MP_ROM_PTR(&mp_module_rtthread) },
#else
#define RTTHREAD_PORT_BUILTIN_MODULES
#endif /* MICROPY_PY_RTTHREAD */


#if MICROPY_PY_UOS_VFS
extern const struct _mp_obj_module_t mp_module_uos_vfs;
#define MICROPY_PY_UOS_DEF { MP_ROM_QSTR(MP_QSTR_uos), MP_ROM_PTR(&mp_module_uos_vfs) },
#else
#define MICROPY_PY_UOS_DEF { MP_ROM_QSTR(MP_QSTR_uos), MP_ROM_PTR(&mp_module_os) },
#endif

#define MICROPY_PORT_BUILTIN_MODULES \
    { MP_ROM_QSTR(MP_QSTR_machine), MP_ROM_PTR(&mp_module_machine) }, \
    { MP_ROM_QSTR(MP_QSTR_mimxrt), (mp_obj_t)&mp_module_mimxrt }, \
    { MP_ROM_QSTR(MP_QSTR_pyb), MP_ROM_PTR(&pyb_module) },\
    { MP_ROM_QSTR(MP_QSTR_time), MP_ROM_PTR(&mp_module_time) }, \
    { MP_ROM_QSTR(MP_QSTR_utime), MP_ROM_PTR(&mp_module_time) }, \
    SOCKET_BUILTIN_MODULE \
    NETWORK_BUILTIN_MODULE \
    MICROPY_PY_UOS_DEF \
    MICROPY_PY_LVGL_DEF \

#define MICROPY_PORT_NETWORK_INTERFACES \
    MICROPY_HW_NIC_ETH  \

#define MICROPY_HW_PIT_NUM_CHANNELS 3

#define MICROPY_PORT_ROOT_POINTERS \
    void *script_buffer; \
    mp_obj_t mp_const_ide_interrupt;\
    LVGL_ROOT_DATA \
    mp_obj_t pyb_hid_report_desc; \
    const char *readline_hist[8]; \
    struct _machine_timer_obj_t *timer_table[MICROPY_HW_PIT_NUM_CHANNELS]; \
    void *machine_pin_irq_objects[MICROPY_HW_NUM_PIN_IRQS]; \
    /* list of registered NICs */ \
    mp_obj_list_t mod_network_nic_list; \
    /* root pointers for sub-systems */ \
    MICROPY_PORT_ROOT_POINTER_MBEDTLS \
    mp_obj_t pin_class_mapper; \
    mp_obj_t pin_class_map_dict; \
     void* pvPortRoots[16];
     
	
#define MP_STATE_PORT MP_STATE_VM


// Miscellaneous settings

#ifdef NXP_USING_OPENMV	

extern void usbdbg_try_run_script();
#define OS_POLL_HOOK \
    do { \
        usbdbg_try_run_script(); \
        MP_THREAD_GIL_EXIT(); \
        MP_THREAD_GIL_ENTER(); \
    }while(0);\

#else

#define OS_POLL_HOOK \
    do { \
        MP_THREAD_GIL_EXIT(); \
        MP_THREAD_GIL_ENTER(); \
    }while(0);\
	
#endif

#define MICROPY_EVENT_POLL_HOOK \
    do { \
        extern void mp_handle_pending(bool); \
        mp_handle_pending(true); \
        OS_POLL_HOOK \
    } while (0);

#define MICROPY_MAKE_POINTER_CALLABLE(p) ((void *)((mp_uint_t)(p) | 1))

#define MP_HAL_CLEAN_DCACHE(addr, size) \
    (SCB_CleanDCache_by_Addr((uint32_t *)((uint32_t)addr & ~0x1f), \
    ((uint32_t)((uint8_t *)addr + size + 0x1f) & ~0x1f) - ((uint32_t)addr & ~0x1f)))


#define MP_SSIZE_MAX (0x7fffffff)
typedef int mp_int_t; // must be pointer size
typedef unsigned mp_uint_t; // must be pointer size
typedef long mp_off_t;
//typedef int ssize_t;
// Need an implementation of the log2 function which is not a macro.
#define MP_NEED_LOG2 (0)

// Need to provide a declaration/definition of alloca()
#include <alloca.h>
