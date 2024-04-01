/*
 * Copyright (c) 2006-2018, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-04-29     tyustli      first version
 */

#if (MICROPY_PY_AUDIO)
    extern const struct _mp_obj_module_t audio_module;
    #define MODULE_DEF_MP_QSTR_AUDIO { MP_ROM_QSTR(MP_QSTR_audio), MP_ROM_PTR(&audio_module) },
#else
    #define MODULE_DEF_MP_QSTR_AUDIO
#endif

#if (MICROPY_PY_BUZZER)
    extern const struct _mp_obj_module_t buzzer_module;
    #define MODULE_DEF_MP_QSTR_BUZZER { MP_ROM_QSTR(MP_QSTR_buzzer), MP_ROM_PTR(&buzzer_module) },
#else
    #define MODULE_DEF_MP_QSTR_BUZZER
#endif

#if (1)
    extern const struct _mp_obj_module_t example_user_cmodule;
    #define MODULE_DEF_MP_QSTR_CEXAMPLE { MP_ROM_QSTR(MP_QSTR_cexample), MP_ROM_PTR(&example_user_cmodule) },
#else
    #define MODULE_DEF_MP_QSTR_CEXAMPLE
#endif

#if (0)
    extern const struct _mp_obj_module_t cpufreq_module;
    #define MODULE_DEF_MP_QSTR_CPUFREQ { MP_ROM_QSTR(MP_QSTR_cpufreq), MP_ROM_PTR(&cpufreq_module) },
#else
    #define MODULE_DEF_MP_QSTR_CPUFREQ
#endif

#if (0) //def NXP_USING_OPENMV
    extern const struct _mp_obj_module_t fir_module;
    #define MODULE_DEF_MP_QSTR_FIR { MP_ROM_QSTR(MP_QSTR_fir), MP_ROM_PTR(&fir_module) },
#else
    #define MODULE_DEF_MP_QSTR_FIR
#endif

#ifdef NXP_USING_OPENMV
    extern const struct _mp_obj_module_t gif_module;
    #define MODULE_DEF_MP_QSTR_GIF { MP_ROM_QSTR(MP_QSTR_gif), MP_ROM_PTR(&gif_module) },
#else
    #define MODULE_DEF_MP_QSTR_GIF
#endif

#ifdef NXP_USING_OPENMV
    extern const struct _mp_obj_module_t image_module;
    #define MODULE_DEF_MP_QSTR_IMAGE { MP_ROM_QSTR(MP_QSTR_image), MP_ROM_PTR(&image_module) },
#else
    #define MODULE_DEF_MP_QSTR_IMAGE
#endif

#if (MICROPY_PY_IMU)
    extern const struct _mp_obj_module_t imu_module;
    #define MODULE_DEF_MP_QSTR_IMU { MP_ROM_QSTR(MP_QSTR_imu), MP_ROM_PTR(&imu_module) },
#else
    #define MODULE_DEF_MP_QSTR_IMU
#endif

#if (MICROPY_PY_LCD)
    extern const struct _mp_obj_module_t lcd_module;
    #define MODULE_DEF_MP_QSTR_LCD { MP_ROM_QSTR(MP_QSTR_lcd), MP_ROM_PTR(&lcd_module) },
#else
    #define MODULE_DEF_MP_QSTR_LCD
#endif

#if (MICROPY_PY_MICRO_SPEECH)
    extern const struct _mp_obj_module_t micro_speech_module;
    #define MODULE_DEF_MP_QSTR_MICRO_SPEECH { MP_ROM_QSTR(MP_QSTR_micro_speech), MP_ROM_PTR(&micro_speech_module) },
#else
    #define MODULE_DEF_MP_QSTR_MICRO_SPEECH
#endif

#ifdef NXP_USING_OPENMV
    extern const struct _mp_obj_module_t mjpeg_module;
    #define MODULE_DEF_MP_QSTR_MJPEG { MP_ROM_QSTR(MP_QSTR_mjpeg), MP_ROM_PTR(&mjpeg_module) },
#else
    #define MODULE_DEF_MP_QSTR_MJPEG
#endif

#ifdef NXP_USING_OPENMV
    extern const struct _mp_obj_module_t omv_module;
    #define MODULE_DEF_MP_QSTR_OMV { MP_ROM_QSTR(MP_QSTR_omv), MP_ROM_PTR(&omv_module) },
#else
    #define MODULE_DEF_MP_QSTR_OMV
#endif

#if (MICROPY_PY_SENSOR)
    extern const struct _mp_obj_module_t sensor_module;
    #define MODULE_DEF_MP_QSTR_SENSOR { MP_ROM_QSTR(MP_QSTR_sensor), MP_ROM_PTR(&sensor_module) },
#else
    #define MODULE_DEF_MP_QSTR_SENSOR
#endif

#ifdef NXP_USING_OPENMV
    extern const struct _mp_obj_module_t tf_module;
    #define MODULE_DEF_MP_QSTR_TF { MP_ROM_QSTR(MP_QSTR_tf), MP_ROM_PTR(&tf_module) },
	
	extern const struct _mp_obj_module_t g_cmm_module;
	#define MODULE_DEF_MP_QSTR_CMM { MP_OBJ_NEW_QSTR(MP_QSTR_cmm), MP_ROM_PTR(&g_cmm_module) },
#else
    #define MODULE_DEF_MP_QSTR_TF
	
	#define MODULE_DEF_MP_QSTR_CMM
#endif

#if (OMV_PY_TOF)
    extern const struct _mp_obj_module_t tof_module;
    #define MODULE_DEF_MP_QSTR_TOF { MP_ROM_QSTR(MP_QSTR_tof), MP_ROM_PTR(&tof_module) },
#else
    #define MODULE_DEF_MP_QSTR_TOF
#endif

#if (MICROPY_PY_TV)
    extern const struct _mp_obj_module_t tv_module;
    #define MODULE_DEF_MP_QSTR_TV { MP_ROM_QSTR(MP_QSTR_tv), MP_ROM_PTR(&tv_module) },
#else
    #define MODULE_DEF_MP_QSTR_TV
#endif

#if (MICROPY_PY_ARRAY)
    extern const struct _mp_obj_module_t mp_module_uarray;
    #define MODULE_DEF_MP_QSTR_UARRAY { MP_ROM_QSTR(MP_QSTR_uarray), MP_ROM_PTR(&mp_module_uarray) },
#else
    #define MODULE_DEF_MP_QSTR_UARRAY
#endif

#if (MODULE_ULAB_ENABLED)
    extern const struct _mp_obj_module_t ulab_user_cmodule;
    #define MODULE_DEF_MP_QSTR_ULAB { MP_ROM_QSTR(MP_QSTR_ulab), MP_ROM_PTR(&ulab_user_cmodule) },
#else
    #define MODULE_DEF_MP_QSTR_ULAB
#endif

#if MICROPY_PY_ONEWIRE
extern const struct _mp_obj_module_t mp_module_onewire;
    #define MODULE_DEF_MP_ONELWIRE { MP_ROM_QSTR(MP_QSTR_onewire), MP_ROM_PTR(&mp_module_onewire) },
#else
    #define MODULE_DEF_MP_ONELWIRE
#endif

extern const struct _mp_obj_module_t nxp_module;
#define MODULE_DEF_MP_QSTR_NXP_MODULE {MP_ROM_QSTR(MP_QSTR_nxp_module), MP_ROM_PTR(&nxp_module)},

#define MICROPY_REGISTERED_MODULES \
    MODULE_DEF_MP_QSTR_NXP_MODULE \
    MODULE_DEF_MP_QSTR_AUDIO \
    MODULE_DEF_MP_QSTR_BUZZER \
    MODULE_DEF_MP_QSTR_CEXAMPLE \
    MODULE_DEF_MP_QSTR_CPUFREQ \
    MODULE_DEF_MP_QSTR_FIR \
    MODULE_DEF_MP_QSTR_GIF \
    MODULE_DEF_MP_QSTR_IMAGE \
    MODULE_DEF_MP_QSTR_IMU \
    MODULE_DEF_MP_QSTR_LCD \
    MODULE_DEF_MP_QSTR_MICRO_SPEECH \
    MODULE_DEF_MP_QSTR_MJPEG \
    MODULE_DEF_MP_QSTR_OMV \
    MODULE_DEF_MP_QSTR_SENSOR \
    MODULE_DEF_MP_QSTR_TF \
	MODULE_DEF_MP_QSTR_CMM \
    MODULE_DEF_MP_QSTR_TOF \
    MODULE_DEF_MP_QSTR_TV \
    MODULE_DEF_MP_QSTR_UARRAY \
    MODULE_DEF_MP_QSTR_ULAB \
    MODULE_DEF_MP_ONELWIRE \
    
// MICROPY_REGISTERED_MODULES
