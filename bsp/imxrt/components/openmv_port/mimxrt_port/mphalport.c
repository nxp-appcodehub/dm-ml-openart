/*
 * This file is part of the MicroPython project, http://micropython.org/
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2018 Armink (armink.ztl@gmail.com)
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

#include <rtthread.h>
#include <py/mpconfig.h>
#include <py/runtime.h>
#include "mphalport.h"
#include "mpgetcharport.h"
#include "mpputsnport.h"
#include "drivers/pin.h"
#include "fsl_snvs_lp.h"
#ifdef NXP_USING_OPENMV
#include "drv_usb_omv.h"
#endif
#include "fsl_clock.h"
#include "fsl_gpt.h"

#include "shared/timeutils/timeutils.h"

// General purpose timer for keeping microsecond and millisecond tick values.
#define GPTx GPT2
#define GPTx_IRQn GPT2_IRQn
#define GPTx_IRQHandler GPT2_IRQHandler

static uint32_t ticks_us64_upper;
static uint32_t ticks_ms_upper;

const char rtthread_help_text[] =
"Welcome to MicroPython on RT-Thread!\n"
"\n"
"Control commands:\n"
"  CTRL-A        -- on a blank line, enter raw REPL mode\n"
"  CTRL-B        -- on a blank line, enter normal REPL mode\n"
"  CTRL-C        -- interrupt a running program\n"
"  CTRL-D        -- on a blank line, do a soft reset of the board\n"
"  CTRL-E        -- on a blank line, enter paste mode\n"
"\n"
"For further help on a specific object, type help(obj)\n"
;

int mp_hal_stdin_rx_chr(void) {
    char ch;
    while (1) {
        ch = mp_getchar();
        if (ch != (char)0xFF) {
            break;
        }
        MICROPY_EVENT_POLL_HOOK;
        rt_thread_delay(1);
    }
    return ch;
}

// Send string of given length
void mp_hal_stdout_tx_strn(const char *str, mp_uint_t len) {
    mp_putsn(str, len);
#ifdef NXP_USING_OPENMV	
	if (usb_vcp_is_enabled()) {
		// rocky: if send through VCP, on windows, MUST open the port,
		// otherwise, a buffer on windows will finally overrun, and host 
		// will no longer accept data from us!
		usb_vcp_send_strn(str, len);
	}
#endif	
}


void mp_hal_stdout_tx_strn_stream(const char *str, size_t len) {
    mp_putsn_stream(str, len);
#ifdef NXP_USING_OPENMV		
    if (usb_vcp_is_enabled()) {
		// rocky: if send through VCP, on windows, MUST open the port,
		// otherwise, a buffer on windows will finally overrun, and host 
		// will no longer accept data from us!		
        usb_vcp_send_strn_cooked(str, len);
    } else {

    }
#endif
}

void ticks_init(void) {
    ticks_us64_upper = 0;
    ticks_ms_upper = 0;

    gpt_config_t config;
    config.clockSource = kGPT_ClockSource_Osc;
    config.divider = 24; // XTAL is 24MHz
    config.enableFreeRun = true;
    config.enableRunInWait = true;
    config.enableRunInStop = true;
    config.enableRunInDoze = true;
    config.enableRunInDbg = false;
    config.enableMode = true;
    GPT_Init(GPTx, &config);

    GPT_EnableInterrupts(GPTx, kGPT_RollOverFlagInterruptEnable);
    NVIC_SetPriority(GPTx_IRQn, 0); // highest priority
    NVIC_EnableIRQ(GPTx_IRQn);

    GPT_StartTimer(GPTx);
    #ifdef NDEBUG
    mp_hal_ticks_cpu_enable();
    #endif
}

void GPTx_IRQHandler(void) {
    if (GPT_GetStatusFlags(GPTx, kGPT_OutputCompare1Flag)) {
        GPT_ClearStatusFlags(GPTx, kGPT_OutputCompare1Flag);
        GPT_DisableInterrupts(GPTx, kGPT_OutputCompare1InterruptEnable);
        __SEV();
    }
    if (GPT_GetStatusFlags(GPTx, kGPT_RollOverFlag)) {
        GPT_ClearStatusFlags(GPTx, kGPT_RollOverFlag);
        ++ticks_us64_upper;
        if (++ticks_ms_upper >= 1000) {
            // Wrap upper counter at a multiple of 1000 so that when mp_hal_ticks_ms()
            // wraps due to overflow it wraps smoothly.
            ticks_ms_upper = 0;
        }
    }
}

static void ticks_wake_after_us32(uint32_t us) {
    if (us < 2) {
        // Delay too short to guarantee that we won't miss it when setting the OCR below.
        __SEV();
    } else {
        // Disable IRQs so setting the OCR is done without any interruption.
        uint32_t irq_state = DisableGlobalIRQ();
        GPT_EnableInterrupts(GPTx, kGPT_OutputCompare1InterruptEnable);
        uint32_t oc = GPT_GetCurrentTimerCount(GPTx) + us;
        GPT_SetOutputCompareValue(GPTx, kGPT_OutputCompare_Channel1, oc);
        EnableGlobalIRQ(irq_state);
    }
}

static uint64_t ticks_us64_with(uint32_t *upper_ptr) {
    uint32_t irq_state = DisableGlobalIRQ();
    uint32_t lower = GPT_GetCurrentTimerCount(GPTx);
    uint32_t upper = *upper_ptr;
    uint32_t overflow = GPT_GetStatusFlags(GPTx, kGPT_RollOverFlag);
    EnableGlobalIRQ(irq_state);
    if (overflow && lower < 0x80000000) {
        // The timer counter overflowed before reading it but the IRQ handler
        // has not yet been called, so perform the IRQ arithmetic now.
        ++upper;
    }
    return (uint64_t)upper << 32 | (uint64_t)lower;
}

uint32_t ticks_us32(void) {
    return GPT_GetCurrentTimerCount(GPTx);
}

uint64_t ticks_us64(void) {
    return ticks_us64_with(&ticks_us64_upper);
}

uint32_t ticks_ms32(void) {
    // This will return a value that only has the lower 32-bits valid.
    return ticks_us64_with(&ticks_ms_upper) / 1000;
}

mp_uint_t mp_hal_ticks_us(void) {
    return ticks_us32();
}

mp_uint_t mp_hal_ticks_ms(void) {
    return ticks_ms32();
}

static void mp_hal_ticks_cpu_enable(void) {
    DWT->LAR = 0xc5acce55;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

mp_uint_t mp_hal_ticks_cpu(void) {
    if (!(DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk)) {
        mp_hal_ticks_cpu_enable();
    }
    return DWT->CYCCNT;
}

void mp_hal_delay_us(mp_uint_t us) {
    rt_tick_t t0 = rt_tick_get(), t1, dt;
    uint64_t dtick = us * RT_TICK_PER_SECOND / 1000000L;
    while (1) {
        t1 = rt_tick_get();
        dt = t1 - t0;
        if (dt >= dtick) {
            break;
        }
        mp_handle_pending(true);
    }
}

uint64_t mp_hal_time_ns(void) {
    snvs_lp_srtc_datetime_t t;
    SNVS_LP_SRTC_GetDatetime(SNVS, &t);
    uint64_t s = timeutils_seconds_since_epoch(t.year, t.month, t.day, t.hour, t.minute, t.second);
    return s * 1000000000ULL;
}

void mp_hal_delay_ms(mp_uint_t ms) {
    rt_tick_t t0 = rt_tick_get(), t1, dt;
    uint64_t dtick = ms * RT_TICK_PER_SECOND / 1000L;
    while (1) {
        t1 = rt_tick_get();
        dt = t1 - t0;
        if (dt >= dtick) {
            break;
        }
        mp_handle_pending(true);
        rt_thread_delay(1);
    }
}
#if MICROPY_VFS_POSIX || MICROPY_VFS_POSIX_FILE

#endif