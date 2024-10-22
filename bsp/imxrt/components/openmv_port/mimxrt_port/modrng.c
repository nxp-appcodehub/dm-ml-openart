/*
 * This file is part of the MicroPython project, http://micropython.org/
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2013-2018 Damien P. George
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
#include "modrng.h"
#include "board.h"
#include <stdio.h>
#include "py/nlr.h"
#include "py/runtime.h"
#include "py/mphal.h"
#define TRNG0 TRNG
#define TRNG_EXAMPLE_RANDOM_NUMBER 1
#include <stdlib.h>
#ifdef MICROPY_HW_ENABLE_RNG
#ifdef SOC_IMXRT1170_SERIES
#include "fsl_caam.h"
uint8_t s_isRngInited;
uint32_t s_seed;
caam_handle_t caamHandle;
caam_config_t caamConfig;
uint32_t rng_get(void) {

	int status;
    CAAM_Type *base = CAAM;

    uint32_t data[TRNG_EXAMPLE_RANDOM_NUMBER];
	if (s_isRngInited == 0) {
	
		CAAM_GetDefaultConfig(&caamConfig);

		status = CAAM_Init(base, &caamConfig);
		caamHandle.jobRing = kCAAM_JobRing0;
		status = CAAM_RNG_GetRandomData(base,&caamHandle,0,
										data,sizeof(data),0,NULL);
		if (status == kStatus_Success)
		{
			s_seed = data[0];
			srand(s_seed);
		}
	}	
	return rand();
}

// Return a 30-bit hardware generated random number.
STATIC mp_obj_t pyb_rng_getnum(void) {
	return mp_obj_new_int(rng_get() >> 2);
}

MP_DEFINE_CONST_FUN_OBJ_0(pyb_rng_getnum_obj, pyb_rng_getnum);

static bool initialized = false;

STATIC void trng_start(void) {
	CAAM_Type *base = CAAM;
    int status;
    if (!initialized) {
        CAAM_GetDefaultConfig(&caamConfig);

		status = CAAM_Init(base, &caamConfig);
		caamHandle.jobRing = kCAAM_JobRing0;
        initialized = true;
    }
}

uint32_t trng_random_u32(void) {
    uint32_t rngval;
	int status;
    trng_start();
    status = CAAM_RNG_GetRandomData(CAAM,&caamHandle,0,
										&rngval,sizeof(rngval),0,NULL);
    return rngval;
}
#else
#include "fsl_trng.h"
uint8_t s_isRngInited;
uint32_t s_seed;
uint32_t rng_get(void) {
    uint32_t i;
    trng_config_t trngConfig;
    status_t status;
    uint32_t data[TRNG_EXAMPLE_RANDOM_NUMBER];
	if (s_isRngInited == 0) {
	
		TRNG_GetDefaultConfig(&trngConfig);
		/* Set sample mode of the TRNG ring oscillator to Von Neumann, for better random data.
		 * It is optional.*/
		trngConfig.sampleMode = kTRNG_SampleModeVonNeumann;
		/* Initialize TRNG */
		status = TRNG_Init(TRNG0, &trngConfig);
		s_isRngInited = 1;
		status = TRNG_GetRandomData(TRNG0, data, sizeof(data));
		if (status == kStatus_Success)
		{
			s_seed = data[0];
			srand(s_seed);
		}
	}	
	return rand();
}

// Return a 30-bit hardware generated random number.
STATIC mp_obj_t pyb_rng_getnum(void) {
	return mp_obj_new_int(rng_get() >> 2);
}

MP_DEFINE_CONST_FUN_OBJ_0(pyb_rng_getnum_obj, pyb_rng_getnum);

static bool initialized = false;

STATIC void trng_start(void) {
    trng_config_t trngConfig;

    if (!initialized) {
        TRNG_GetDefaultConfig(&trngConfig);
        trngConfig.sampleMode = kTRNG_SampleModeVonNeumann;

        TRNG_Init(TRNG, &trngConfig);
        initialized = true;
    }
}

uint32_t trng_random_u32(void) {
    uint32_t rngval;

    trng_start();
    TRNG_GetRandomData(TRNG, (uint8_t *)&rngval, 4);
    return rngval;
}
#endif
#endif // MICROPY_HW_ENABLE_RNG
