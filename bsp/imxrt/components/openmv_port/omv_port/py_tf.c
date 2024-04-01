/* This file is part of the OpenMV project.
 * Copyright (c) 2013-2019 Ibrahim Abdelkader <iabdalkader@openmv.io> & Kwabena W. Agyeman <kwagyeman@openmv.io>
 * This work is licensed under the MIT license, see the file LICENSE for details.
 */

#include "py/obj.h"
#include "py/runtime.h"
#include "py_helper.h"
#include "py_image.h"
#include "ff_wrapper.h"
#include "py/objarray.h"
#include "py_assert.h"
#include "libtf.h"

#define GRAYSCALE_RANGE ((COLOR_GRAYSCALE_MAX) - (COLOR_GRAYSCALE_MIN))
#define GRAYSCALE_MID   (((GRAYSCALE_RANGE) + 1) / 2)

typedef struct HookOp_struct {
	uint32_t opCode;
	const char *pszName;
}HookOp_t;
#define CFG_TS_CNT	512
#define CFG_OPC_CNT	48
typedef struct tagOpProfile {
	uint32_t usTime;
	uint32_t x10000Rate;
	uint16_t ndx;
	uint16_t opCode;
	char szOpName[32];
}OpProfile_t;

OpProfile_t g_opPfs[CFG_TS_CNT], g_opcPfs[CFG_OPC_CNT];
volatile uint32_t g_tsNdx=0;
volatile uint32_t g_usTotal=0;
volatile uint32_t g_ts0=0;

static volatile int g_tf_profiling_en = 0;

static uint32_t rt_tick_get_us(){
	uint32_t tick = rt_tick_get();
	uint32_t us = (SysTick->LOAD - SysTick->VAL) / (SystemCoreClock/1000000);
	us += tick * 1000;
	return us;
}

uint32_t TFLm_HookBeforeInvoke(HookOp_t *pCtx) {
    if (!g_tf_profiling_en) return 0;

	g_ts0 = rt_tick_get_us();
	if (g_tsNdx < CFG_TS_CNT) {
		OpProfile_t *p = g_opPfs + g_tsNdx;
		p->ndx = (uint16_t) g_tsNdx;
		if (strlen(pCtx->pszName) < 32) {
			strcpy(p->szOpName, pCtx->pszName);
		} else {
			memcpy(p->szOpName, pCtx->pszName, 31);
			p->szOpName[31] = 0;
		}
		p->opCode = (uint16_t) pCtx->opCode;
	}		
	return 0;
}

uint32_t TFLm_HookAfterInvoke(HookOp_t *pCtx) {
    if (!g_tf_profiling_en) return 0;
	uint32_t dt = rt_tick_get_us() - g_ts0;
	if (g_tsNdx < CFG_TS_CNT) {
		OpProfile_t *p = g_opPfs + g_tsNdx;
		p->usTime = dt;
		g_tsNdx++;
	}
	g_usTotal += dt;	
	return 0;
}

static void SortDescending(OpProfile_t *p, uint32_t cnt) {
	int i, j;
	OpProfile_t tmp;
	for (i=0; i<cnt; i++) {
		for (j=i+1; j<cnt; j++) {
			if (p[i].usTime < p[j].usTime) {
				tmp = p[i];
				p[i] = p[j];
				p[j] = tmp;
			}
		}
	}
}

void ShowProfiling(void)
{
	OpProfile_t *p;
	mp_printf(&mp_plat_print, "\r\n--------------------------------------------------");
	mp_printf(&mp_plat_print, "\r\n-------------------Profiling----------------------");
	mp_printf(&mp_plat_print, "\r\n--------------------------------------------------\r\n");
	mp_printf(&mp_plat_print, "Total inference time: %d.%03dms\r\n", g_usTotal/1000, g_usTotal % 1000);
	
	for (int i=0; i<g_tsNdx; i++) {
		g_opPfs[i].x10000Rate = (uint32_t)((uint64_t)g_opPfs[i].usTime * 10000 / g_usTotal);
	}
	
	// SortDescending(g_opPfs, g_tsNdx);
	uint32_t usAcc = 0, pcntAcc = 0;
	mp_printf(&mp_plat_print, "odr, ndx, time ms,   unit%%, total%%\r\n");
	for (int i=0; i<g_tsNdx; i++) {
		// g_opPfs[i].x10000Rate = (uint32_t)((uint64_t)g_opPfs[i].usTime * 10000 / g_usTotal);
		usAcc += g_opPfs[i].usTime;
		pcntAcc = (uint64_t)usAcc * 10000 / g_usTotal;
		mp_printf(&mp_plat_print, "%03d, %03d, %03d.%03dms, %02d.%02d%%, %02d.%02d%%, %s\r\n", 
			i + 1, g_opPfs[i].ndx, 
			g_opPfs[i].usTime / 1000, g_opPfs[i].usTime % 1000, 
			g_opPfs[i].x10000Rate / 100, g_opPfs[i].x10000Rate % 100,
			pcntAcc / 100, pcntAcc % 100, g_opPfs[i].szOpName);
	}
	// calculate by operator type 
	mp_printf(&mp_plat_print, "\r\n--------------------------------------------------");
	mp_printf(&mp_plat_print, "\r\n                  per operator                    ");
	mp_printf(&mp_plat_print, "\r\n--------------------------------------------------\r\n");
	mp_printf(&mp_plat_print, "Total inference time: %d.%03dms\r\n", g_usTotal/1000, g_usTotal % 1000);
	{
		int opCodeNdx, i, opCodeTypeCnt = 0;
		const char *pszName;
		uint32_t pcnt;
		usAcc = 0;
		pcntAcc = 0;
		for (opCodeNdx=0; opCodeNdx < 256; opCodeNdx++) {
			uint32_t opcUs = 0;
			uint32_t opInstanceCnt = 0;
			for (i=0; i<g_tsNdx; i++) {
				if (g_opPfs[i].opCode != opCodeNdx) {
					continue;
				}
				pszName = g_opPfs[i].szOpName;
				opcUs += g_opPfs[i].usTime;
				opInstanceCnt++;
			}
			if (0 == opcUs) 
				continue;
			if (opCodeTypeCnt >= CFG_OPC_CNT) {
				continue;
			}
			g_opcPfs[opCodeTypeCnt].ndx = opInstanceCnt;
			pcnt = (uint64_t)opcUs * 10000 / g_usTotal;
			g_opcPfs[opCodeTypeCnt].x10000Rate = pcnt;
			g_opcPfs[opCodeTypeCnt].usTime = opcUs;
			g_opcPfs[opCodeTypeCnt].opCode = opCodeNdx;
			if (strlen(pszName) < 32) {
				strcpy(g_opcPfs[opCodeTypeCnt].szOpName, pszName);
			} else {
				memcpy(g_opcPfs[opCodeTypeCnt].szOpName, pszName, 31);
				g_opcPfs[opCodeTypeCnt].szOpName[31] = 0;
			}
			opCodeTypeCnt++;
		}
		SortDescending(g_opcPfs, opCodeTypeCnt);
		mp_printf(&mp_plat_print, "odr, opc, time ms,   unit%%, total%%,count,name\r\n");
		for (int i=0; i < opCodeTypeCnt; i++) {
			usAcc += g_opcPfs[i].usTime;
			pcntAcc = (uint64_t)usAcc * 10000 / g_usTotal;
			mp_printf(&mp_plat_print, "%02d, %03d, %03d.%03dms, %02d.%02d%%, %02d.%02d%%, %03d, %s\r\n",
				i + 1, g_opcPfs[i].opCode, 
				g_opcPfs[i].usTime/1000, g_opcPfs[i].usTime%1000, 
				g_opcPfs[i].x10000Rate/100, g_opcPfs[i].x10000Rate%100,
				pcntAcc / 100, pcntAcc % 100, g_opcPfs[i].ndx, g_opcPfs[i].szOpName);					
		}
	}
	
	g_tsNdx=0;
	g_usTotal=0;
	g_ts0=0;
}

#ifdef IMLIB_ENABLE_TF

#define PY_TF_PUTCHAR_BUFFER_LEN 4096

char* py_tf_putchar_buffer = NULL;
size_t py_tf_putchar_buffer_len = 0;
int py_tf_putchar_buffer_index = 0;
STATIC void alloc_putchar_buffer()
{
    py_tf_putchar_buffer = (char *) fb_alloc0(PY_TF_PUTCHAR_BUFFER_LEN + 1, FB_ALLOC_NO_HINT);
    py_tf_putchar_buffer_len = PY_TF_PUTCHAR_BUFFER_LEN;
	py_tf_putchar_buffer_index = 0;
}

// TF Model Object
typedef struct py_tf_model_obj {
    mp_obj_base_t base;
    unsigned char *model_data;
    unsigned int model_data_len;
    libtf_parameters_t params;
} py_tf_model_obj_t;

STATIC void py_tf_model_print(const mp_print_t *print, mp_obj_t self_in, mp_print_kind_t kind)
{
    py_tf_model_obj_t *self = self_in;
    mp_printf(print,
              "{\"len\":%d, \"height\":%d, \"width\":%d, \"channels\":%d, \"dataypte\":%d}",
              self->model_data_len,
              self->params.input_height,
              self->params.input_width,
              self->params.input_channels,
              self->params.input_datatype);
}

// TF Classification Object
#define py_tf_classification_obj_size 5
typedef struct py_tf_classification_obj {
    mp_obj_base_t base;
    mp_obj_t x, y, w, h, output;
} py_tf_classification_obj_t;

STATIC void py_tf_classification_print(const mp_print_t *print, mp_obj_t self_in, mp_print_kind_t kind)
{
    py_tf_classification_obj_t *self = self_in;
    mp_printf(print,
              "{\"x\":%d, \"y\":%d, \"w\":%d, \"h\":%d, \"output\":",
              mp_obj_get_int(self->x),
              mp_obj_get_int(self->y),
              mp_obj_get_int(self->w),
              mp_obj_get_int(self->h));
    mp_obj_print_helper(print, self->output, kind);
    mp_printf(print, "}");
}

STATIC mp_obj_t py_tf_classification_subscr(mp_obj_t self_in, mp_obj_t index, mp_obj_t value)
{
    if (value == MP_OBJ_SENTINEL) { // load
        py_tf_classification_obj_t *self = self_in;
        if (MP_OBJ_IS_TYPE(index, &mp_type_slice)) {
            mp_bound_slice_t slice;
            if (!mp_seq_get_fast_slice_indexes(py_tf_classification_obj_size, index, &slice)) {
                nlr_raise(mp_obj_new_exception_msg(&mp_type_OSError, "only slices with step=1 (aka None) are supported"));
            }
            mp_obj_tuple_t *result = mp_obj_new_tuple(slice.stop - slice.start, NULL);
            mp_seq_copy(result->items, &(self->x) + slice.start, result->len, mp_obj_t);
            return result;
        }
        switch (mp_get_index(self->base.type, py_tf_classification_obj_size, index, false)) {
            case 0: return self->x;
            case 1: return self->y;
            case 2: return self->w;
            case 3: return self->h;
            case 4: return self->output;
        }
    }
    return MP_OBJ_NULL; // op not supported
}

mp_obj_t py_tf_classification_rect(mp_obj_t self_in)
{
    return mp_obj_new_tuple(4, (mp_obj_t []) {((py_tf_classification_obj_t *) self_in)->x,
                                              ((py_tf_classification_obj_t *) self_in)->y,
                                              ((py_tf_classification_obj_t *) self_in)->w,
                                              ((py_tf_classification_obj_t *) self_in)->h});
}

mp_obj_t py_tf_classification_x(mp_obj_t self_in) { return ((py_tf_classification_obj_t *) self_in)->x; }
mp_obj_t py_tf_classification_y(mp_obj_t self_in) { return ((py_tf_classification_obj_t *) self_in)->y; }
mp_obj_t py_tf_classification_w(mp_obj_t self_in) { return ((py_tf_classification_obj_t *) self_in)->w; }
mp_obj_t py_tf_classification_h(mp_obj_t self_in) { return ((py_tf_classification_obj_t *) self_in)->h; }
mp_obj_t py_tf_classification_output(mp_obj_t self_in) { return ((py_tf_classification_obj_t *) self_in)->output; }

STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_classification_rect_obj, py_tf_classification_rect);
STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_classification_x_obj, py_tf_classification_x);
STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_classification_y_obj, py_tf_classification_y);
STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_classification_w_obj, py_tf_classification_w);
STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_classification_h_obj, py_tf_classification_h);
STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_classification_output_obj, py_tf_classification_output);

STATIC const mp_rom_map_elem_t py_tf_classification_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_rect), MP_ROM_PTR(&py_tf_classification_rect_obj) },
    { MP_ROM_QSTR(MP_QSTR_x), MP_ROM_PTR(&py_tf_classification_x_obj) },
    { MP_ROM_QSTR(MP_QSTR_y), MP_ROM_PTR(&py_tf_classification_y_obj) },
    { MP_ROM_QSTR(MP_QSTR_w), MP_ROM_PTR(&py_tf_classification_w_obj) },
    { MP_ROM_QSTR(MP_QSTR_h), MP_ROM_PTR(&py_tf_classification_h_obj) },
    { MP_ROM_QSTR(MP_QSTR_output), MP_ROM_PTR(&py_tf_classification_output_obj) }
};

STATIC MP_DEFINE_CONST_DICT(py_tf_classification_locals_dict, py_tf_classification_locals_dict_table);

static const mp_obj_type_t py_tf_classification_type = {
    { &mp_type_type },
    .name  = MP_QSTR_tf_classification,
    .print = py_tf_classification_print,
    .subscr = py_tf_classification_subscr,
    .locals_dict = (mp_obj_t) &py_tf_classification_locals_dict
};

static const mp_obj_type_t py_tf_model_type;

#if defined(__CC_ARM) || defined (__CLANG_ARM)
extern char Image$$WEIT_CACHE_AREA$$Base[];
extern char Image$$OCRAM_AREA$$Base[];
extern char Image$$OCRAM_AREA_END$$Base[];
#define weit_cache_area Image$$WEIT_CACHE_AREA$$Base

#define ocram_area Image$$OCRAM_AREA$$Base
#define ocram_area_end Image$$OCRAM_AREA_END$$Base
#elif defined __GNUC__
extern char WEIT_CACHE_AREA;
extern char OCRAM_AREA;
extern char OCRAM_AREA_END;
#define weit_cache_area &WEIT_CACHE_AREA
#define ocram_area &OCRAM_AREA
#define ocram_area_end &OCRAM_AREA_END
#endif

void* conv_helper_enter(uint32_t ndx, uint32_t maxSizeInBytes, uint32_t *pAllocedSizeInBytes) 
{
    pAllocedSizeInBytes[0] = WEIT_CACHE_SIZE;
    return (void*) (weit_cache_area);
}

int conv_helper_disable(const q7_t* input_data)
{
    return 1;
    /*
    if(((uint32_t)input_data >= 0x20000000) && ((uint32_t)input_data <= (0x20000000 + sizeof(dtcm_arena)))) 
		return 1;
	else
		return 0;
    */
}
STATIC mp_obj_t int_py_tf_load(mp_obj_t path_obj, bool alloc_mode, bool weit_cache, bool helper_mode)
{
    if (!helper_mode) {
        fb_alloc_mark();
    }

//	if(weit_cache){
//		init_weigth_cache(weit_cache_area, WEIT_CACHE_SIZE);
//	}else{
//		// Using the weight cache by assert ptr_head==NULL, so if not use weit_cache, 
//		// need to deinit, in case open then close, will mislead the code
//		deinit_weight_cache();
//	}
    const char *path = mp_obj_str_get_str(path_obj);
    py_tf_model_obj_t *tf_model = m_new_obj(py_tf_model_obj_t);
    tf_model->base.type = &py_tf_model_type;


        FIL fp;
        file_read_open(&fp, path);
        tf_model->model_data_len = f_size(&fp);
        tf_model->model_data = alloc_mode
            ? fb_alloc(tf_model->model_data_len, FB_ALLOC_PREFER_SIZE)
            : xalloc(tf_model->model_data_len);
        read_data(&fp, tf_model->model_data, tf_model->model_data_len);
        file_close(&fp);


    if (!helper_mode) {
        alloc_putchar_buffer();
    }

    uint32_t tensor_arena_size; 
    
	
	tf_model->params.tcm_arena = fb_alloc_all(&tf_model->params.tcm_arena_size, FB_ALLOC_PREFER_SPEED);
    tf_model->params.ocram_arena = (uint8_t *)ocram_area;
    tf_model->params.ocram_arena_size =  ocram_area_end - ocram_area;
	uint8_t *tensor_arena = fb_alloc_all(&tensor_arena_size, FB_ALLOC_PREFER_SIZE);
	
    PY_ASSERT_FALSE_MSG(libtf_get_parameters(tf_model->model_data,
                                                 tensor_arena,
                                                 tensor_arena_size,
                                                 &tf_model->params),
                        py_tf_putchar_buffer - (PY_TF_PUTCHAR_BUFFER_LEN - py_tf_putchar_buffer_len));

    fb_free(); // free fb_alloc_all()

    if (!helper_mode) {
        fb_free(); // free alloc_putchar_buffer()
    }

    // In this mode we leave the model allocated on the frame buffer.
    // py_tf_free_from_fb() must be called to free the model allocated on the frame buffer.
    // On error everything is cleaned because of fb_alloc_mark().

    if ((!helper_mode) && (!alloc_mode)) {
        fb_alloc_free_till_mark();
    } else if ((!helper_mode) && alloc_mode) {
        fb_alloc_mark_permanent(); // tf_model->model_data will not be popped on exception.
    }

    return tf_model;
}

STATIC mp_obj_t py_tf_load(uint n_args, const mp_obj_t *args, mp_map_t *kw_args)
{
    return int_py_tf_load(args[0], py_helper_keyword_int(n_args, args, 1, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_load_to_fb), false), py_helper_keyword_int(n_args, args, 2, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_weit_cache), true), false);
}
STATIC MP_DEFINE_CONST_FUN_OBJ_KW(py_tf_load_obj, 1, py_tf_load);

STATIC mp_obj_t py_tf_free_from_fb()
{
    fb_alloc_free_till_mark_past_mark_permanent();
    return mp_const_none;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_0(py_tf_free_from_fb_obj, py_tf_free_from_fb);

STATIC py_tf_model_obj_t *py_tf_load_alloc(mp_obj_t path_obj)
{
    if (MP_OBJ_IS_TYPE(path_obj, &py_tf_model_type)) {
        return (py_tf_model_obj_t *) path_obj;
    } else {
        return (py_tf_model_obj_t *) int_py_tf_load(path_obj, true, true, true);
    }
} 

typedef struct py_tf_input_data_callback_data {
    image_t *img;
    rectangle_t *roi;
	int offset, scale;
	uint8_t bgr_mode;//1:bgr,0:rgb
} py_tf_input_data_callback_data_t;



STATIC void py_tf_input_data_callback(void *callback_data,
                                      void *model_input,
                                      libtf_parameters_t *params)
{

    py_tf_input_data_callback_data_t *arg = (py_tf_input_data_callback_data_t *) callback_data;
    int shift = (params->input_datatype == LIBTF_DATATYPE_INT8) ? GRAYSCALE_MID : 0;
    float fscale = 1.0f / (arg->scale);
	float offset = arg->offset * fscale;

    float xscale = params->input_width  / ((float) arg->roi->w);
    float yscale = params->input_height / ((float) arg->roi->h);
    // MAX == KeepAspectRationByExpanding - MIN == KeepAspectRatio
    float scale = IM_MAX(xscale, yscale), scale_inv = 1 / scale;
    float x_offset = ((arg->roi->w * scale) - params->input_width) / 2;
    float y_offset = ((arg->roi->h * scale) - params->input_height) / 2;
	
	image_t dst_img;
	dst_img.w = params->input_width;
    dst_img.h = params->input_height;
    dst_img.data = (uint8_t *) model_input;
	
	if (params->input_channels == 1) {
        dst_img.pixfmt = PIXFORMAT_GRAYSCALE;
    } else if (params->input_channels == 3) {
        dst_img.pixfmt = PIXFORMAT_RGB565;
    } else {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Expected model input channels to be 1 or 3!"));
    }
	
	imlib_draw_image(&dst_img, arg->img, 0, 0, scale, scale, arg->roi,
                     -1, 256, NULL, NULL, IMAGE_HINT_BILINEAR | IMAGE_HINT_BLACK_BACKGROUND,
                     NULL, NULL);
	int size = (params->input_width * params->input_height) - 1; // must be int per countdown loop
	
	 if (params->input_channels == 1) { // GRAYSCALE
        if (params->input_datatype == LIBTF_DATATYPE_FLOAT) { // convert u8 -> f32
            uint8_t *model_input_u8 = (uint8_t *) model_input;
            float *model_input_f32 = (float *) model_input;

            for (; size >= 0; size -= 1) {
                model_input_f32[size] = model_input_u8[size] * fscale;
            }
        } else {
            if (shift) { // convert u8 -> s8
                uint8_t *model_input_8 = (uint8_t *) model_input;

                #if (__ARM_ARCH > 6)
                for (; size >= 3; size -= 4) {
                    *((uint32_t *) (model_input_8 + size - 3)) ^= 0x80808080;
                }
                #endif

                for (; size >= 0; size -= 1) {
                    model_input_8[size] ^= GRAYSCALE_MID;
                }
            }
        }
    } else if (params->input_channels == 3) { // RGB888
        int rgb_size = size * 3; // must be int per countdown loop

        if (params->input_datatype == LIBTF_DATATYPE_FLOAT) {
            uint16_t *model_input_u16 = (uint16_t *) model_input;
            float *model_input_f32 = (float *) model_input;

            for (; size >= 0; size -= 1, rgb_size -= 3) {
                int pixel = model_input_u16[size];
				if(arg->bgr_mode)
				{
					model_input_f32[rgb_size] = COLOR_RGB565_TO_B8(pixel) * fscale;
					model_input_f32[rgb_size + 1] = COLOR_RGB565_TO_G8(pixel) * fscale;
					model_input_f32[rgb_size + 2] = COLOR_RGB565_TO_R8(pixel) * fscale;
				}
				else
				{
					model_input_f32[rgb_size] = COLOR_RGB565_TO_R8(pixel) * fscale;
					model_input_f32[rgb_size + 1] = COLOR_RGB565_TO_G8(pixel) * fscale;
					model_input_f32[rgb_size + 2] = COLOR_RGB565_TO_B8(pixel) * fscale;
				}
            }
        } else {
            uint16_t *model_input_u16 = (uint16_t *) model_input;
            uint8_t *model_input_8 = (uint8_t *) model_input;

            for (; size >= 0; size -= 1, rgb_size -= 3) {
                int pixel = model_input_u16[size];
				if(arg->bgr_mode)
				{
					model_input_8[rgb_size] = COLOR_RGB565_TO_B8(pixel) ^ shift;
					model_input_8[rgb_size + 1] = COLOR_RGB565_TO_G8(pixel) ^ shift;
					model_input_8[rgb_size + 2] = COLOR_RGB565_TO_R8(pixel) ^ shift;
				}
				else
				{
					model_input_8[rgb_size] = COLOR_RGB565_TO_R8(pixel) ^ shift;
					model_input_8[rgb_size + 1] = COLOR_RGB565_TO_G8(pixel) ^ shift;
					model_input_8[rgb_size + 2] = COLOR_RGB565_TO_B8(pixel) ^ shift;
				}
            }
        }
    }
	

}

typedef struct py_tf_classify_output_data_callback_data {
    mp_obj_t out;
} py_tf_classify_output_data_callback_data_t;

STATIC void py_tf_classify_output_data_callback(void *callback_data,
                                                void *model_output,
                                                libtf_parameters_t *params)
{
    py_tf_classify_output_data_callback_data_t *arg = (py_tf_classify_output_data_callback_data_t *) callback_data;

    PY_ASSERT_TRUE_MSG(params->output_height == 1, "Expected model output height to be 1!");
    PY_ASSERT_TRUE_MSG(params->output_width == 1, "Expected model output width to be 1!");

    arg->out = mp_obj_new_list(params->output_channels, NULL);
    
    if (params->output_datatype == LIBTF_DATATYPE_FLOAT) {
        for (int i = 0, ii = params->output_channels; i < ii; i++) {
            ((mp_obj_list_t *) arg->out)->items[i] =
                mp_obj_new_float(((float *) model_output)[i]);
        }
    } else if (params->output_datatype == LIBTF_DATATYPE_INT8) {
        for (int i = 0, ii = params->output_channels; i < ii; i++) {
            ((mp_obj_list_t *) arg->out)->items[i] =
                mp_obj_new_float( ((float) (((int8_t *) model_output)[i] - params->output_zero_point)) * params->output_scale);
        }
    } else {
        for (int i = 0, ii = params->output_channels; i < ii; i++) {
            ((mp_obj_list_t *) arg->out)->items[i] =
                mp_obj_new_float( ((float) (((uint8_t *) model_output)[i] - params->output_zero_point)) * params->output_scale);
        }
    }
}


STATIC mp_obj_t py_tf_classify(uint n_args, const mp_obj_t *args, mp_map_t *kw_args)
{
	int tick;
    fb_alloc_mark();
    alloc_putchar_buffer();

    py_tf_model_obj_t *arg_model = py_tf_load_alloc(args[0]);
    image_t *arg_img = py_helper_arg_to_image_mutable(args[1]);

    rectangle_t roi;
    py_helper_keyword_rectangle_roi(arg_img, n_args, args, 2, kw_args, &roi);

    float arg_min_scale = py_helper_keyword_float(n_args, args, 3, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_min_scale), 1.0f);
    PY_ASSERT_TRUE_MSG((0.0f < arg_min_scale) && (arg_min_scale <= 1.0f), "0 < min_scale <= 1");

    float arg_scale_mul = py_helper_keyword_float(n_args, args, 4, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_scale_mul), 0.5f);
    PY_ASSERT_TRUE_MSG((0.0f <= arg_scale_mul) && (arg_scale_mul < 1.0f), "0 <= scale_mul < 1");

    float arg_x_overlap = py_helper_keyword_float(n_args, args, 5, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_x_overlap), 0.0f);
    PY_ASSERT_TRUE_MSG(((0.0f <= arg_x_overlap) && (arg_x_overlap < 1.0f)) || (arg_x_overlap == -1.0f), "0 <= x_overlap < 1");

    float arg_y_overlap = py_helper_keyword_float(n_args, args, 6, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_y_overlap), 0.0f);
    PY_ASSERT_TRUE_MSG(((0.0f <= arg_y_overlap) && (arg_y_overlap < 1.0f)) || (arg_y_overlap == -1.0f), "0 <= y_overlap < 1");
	
	int offset = py_helper_keyword_int(n_args, args, 7, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_offset), 128);	
	int fscale = py_helper_keyword_int(n_args, args, 8, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_scale), 128);
	int bgr_mode = py_helper_keyword_int(n_args, args, 9, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_bgr), 0);
	
    
    uint8_t *tensor_arena = fb_alloc(arg_model->params.tensor_arena_size, FB_ALLOC_PREFER_SIZE);
    mp_obj_t objects_list = mp_obj_new_list(0, NULL);

    arg_model->params.tcm_arena = fb_alloc_all(&arg_model->params.tcm_arena_size, FB_ALLOC_PREFER_SPEED);
    arg_model->params.ocram_arena = (uint8_t *)ocram_area;
    arg_model->params.ocram_arena_size =  ocram_area_end - ocram_area;

    for (float scale = 1.0f; scale >= arg_min_scale; scale *= arg_scale_mul) {
        // Either provide a subtle offset to center multiple detection windows or center the only detection window.
        for (int y = roi.y + ((arg_y_overlap != -1.0f) ? (fmodf(roi.h, (roi.h * scale)) / 2.0f) : ((roi.h - (roi.h * scale)) / 2.0f));
            // Finish when the detection window is outside of the ROI.
            (y + (roi.h * scale)) <= (roi.y + roi.h);
            // Step by an overlap amount accounting for scale or just terminate after one iteration.
            y += ((arg_y_overlap != -1.0f) ? (roi.h * scale * (1.0f - arg_y_overlap)) : roi.h)) {
            // Either provide a subtle offset to center multiple detection windows or center the only detection window.
            for (int x = roi.x + ((arg_x_overlap != -1.0f) ? (fmodf(roi.w, (roi.w * scale)) / 2.0f) : ((roi.w - (roi.w * scale)) / 2.0f));
                // Finish when the detection window is outside of the ROI.
                (x + (roi.w * scale)) <= (roi.x + roi.w);
                // Step by an overlap amount accounting for scale or just terminate after one iteration.
                x += ((arg_x_overlap != -1.0f) ? (roi.w * scale * (1.0f - arg_x_overlap)) : roi.w)) {

                rectangle_t new_roi;
                rectangle_init(&new_roi, x, y, roi.w * scale, roi.h * scale);

                if (rectangle_overlap(&roi, &new_roi)) { // Check if new_roi is null...

                    py_tf_input_data_callback_data_t py_tf_input_data_callback_data;
                    py_tf_input_data_callback_data.img = arg_img;
                    py_tf_input_data_callback_data.roi = &new_roi;
					py_tf_input_data_callback_data.offset = offset;
					py_tf_input_data_callback_data.scale = fscale;
					py_tf_input_data_callback_data.bgr_mode = bgr_mode;
					tick = rt_tick_get();
                    py_tf_classify_output_data_callback_data_t py_tf_classify_output_data_callback_data;
                    PY_ASSERT_FALSE_MSG(libtf_invoke(arg_model->model_data,
                                                     tensor_arena,
                                                     &arg_model->params,
                                                     py_tf_input_data_callback,
                                                     &py_tf_input_data_callback_data,
                                                     py_tf_classify_output_data_callback,
                                                     &py_tf_classify_output_data_callback_data),
                                        py_tf_putchar_buffer - (PY_TF_PUTCHAR_BUFFER_LEN - py_tf_putchar_buffer_len));

					//mp_printf(&mp_plat_print, "TFLite Inference during %dms\r\n",rt_tick_get()-tick);
                    py_tf_classification_obj_t *o = m_new_obj(py_tf_classification_obj_t);
                    o->base.type = &py_tf_classification_type;
                    o->x = mp_obj_new_int(new_roi.x);
                    o->y = mp_obj_new_int(new_roi.y);
                    o->w = mp_obj_new_int(new_roi.w);
                    o->h = mp_obj_new_int(new_roi.h);
                    o->output = py_tf_classify_output_data_callback_data.out;
                    mp_obj_list_append(objects_list, o);
                }
            }
        }
    }

    fb_alloc_free_till_mark();

    return objects_list;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_KW(py_tf_classify_obj, 2, py_tf_classify);

typedef struct py_tf_logistic_input_data_callback_data
{
    mp_obj_t *input;
    int scale;
    int offset;
    int data_size;
}py_tf_logistic_input_data_callback_data_t;

typedef struct py_tf_logistic_output_data_callback_data
{
    mp_obj_t out;
}py_tf_logistic_output_data_callback_data_t;
STATIC void py_tf_logistic_input_data_callback(void *callback_data,
                                      void *model_input,
                                      libtf_parameters_t *params)
{
    py_tf_logistic_input_data_callback_data_t *arg = (py_tf_logistic_input_data_callback_data_t*)callback_data;
    int shift = (params->input_datatype == LIBTF_DATATYPE_INT8) ? GRAYSCALE_MID : 0;
    float fscale = 1.0f / (arg->scale);
	float offset = arg->offset * fscale;
    int len = params->input_height * params->input_width * params->input_channels;

    PY_ASSERT_TRUE_MSG((len == arg->data_size), "input param size not match with model input");
    for(int i=0;i<len;i++)
    {
        if(params->input_datatype == LIBTF_DATATYPE_FLOAT)
            ((float *)model_input)[i] = mp_obj_get_int(arg->input[i])*fscale - offset;
        else
            ((uint8_t *)model_input)[i] =  mp_obj_get_int(arg->input[i])^shift;
        
        mp_printf(&mp_plat_print, "%d:%f ",mp_obj_get_int(arg->input[i]),((float *)model_input)[i]);
    }
}

STATIC void py_tf_logistic_output_data_callback(void *callback_data,
                                                void *model_output,
                                                libtf_parameters_t *params)
{
    py_tf_logistic_output_data_callback_data_t *arg = (py_tf_logistic_output_data_callback_data_t*)callback_data;
	int shift = (params->input_datatype == LIBTF_DATATYPE_INT8) ? GRAYSCALE_MID : 0;
	
    arg->out = mp_obj_new_list(params->output_channels, NULL);
    for (unsigned int i = 0; i < params->output_channels; i++) {
        if(params->input_datatype == LIBTF_DATATYPE_FLOAT) {
            ((mp_obj_list_t *) arg->out)->items[i] = mp_obj_new_float((((uint8_t *) model_output)[i] ^ shift) / 255.0f);
        } else {
            ((mp_obj_list_t *) arg->out)->items[i] = mp_obj_new_float(((float *) model_output)[i]);
        }
    }
}

STATIC mp_obj_t py_tf_logistic(uint n_args, const mp_obj_t *args, mp_map_t *kw_args)
{
    py_tf_model_obj_t *arg_model = py_tf_load_alloc(args[0]);
    py_tf_logistic_input_data_callback_data_t py_tf_logistic_input_data_callback_data;
    py_tf_logistic_output_data_callback_data_t py_tf_logistic_output_data_callback_data;

    int data_size = py_helper_keyword_int(n_args, args, 2, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_size), 1);	
    int fscale = py_helper_keyword_int(n_args, args, 3, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_scale), 128);
    int offset = py_helper_keyword_int(n_args, args, 3, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_offset), 0);
    py_tf_logistic_input_data_callback_data.input = mp_obj_new_list(data_size, NULL);
    mp_obj_get_array_fixed_n(args[1], data_size, &py_tf_logistic_input_data_callback_data.input);
    py_tf_logistic_input_data_callback_data.data_size = data_size;

    py_tf_logistic_input_data_callback_data.scale = fscale;
    py_tf_logistic_input_data_callback_data.offset = offset;
    fb_alloc_mark();

    arg_model->params.tcm_arena = fb_alloc_all(&arg_model->params.tcm_arena_size, FB_ALLOC_PREFER_SPEED);
    arg_model->params.ocram_arena = (uint8_t *)ocram_area;
    arg_model->params.ocram_arena_size =  ocram_area_end - ocram_area;

    uint8_t *tensor_arena = fb_alloc(arg_model->params.tensor_arena_size, FB_ALLOC_PREFER_SIZE);
    mp_obj_t objects_list = mp_obj_new_list(0, NULL);

    PY_ASSERT_FALSE_MSG(libtf_invoke(arg_model->model_data,
                                                     tensor_arena,
                                                     &arg_model->params,
                                                     py_tf_logistic_input_data_callback,
                                                     &py_tf_logistic_input_data_callback_data,
                                                     py_tf_logistic_output_data_callback,
                                                     &py_tf_logistic_output_data_callback_data),
                                        py_tf_putchar_buffer - (PY_TF_PUTCHAR_BUFFER_LEN - py_tf_putchar_buffer_len));

    mp_obj_list_append(objects_list, py_tf_logistic_output_data_callback_data.out);                                    
    fb_alloc_free_till_mark();

    return objects_list;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_KW(py_tf_logistic_obj,2,py_tf_logistic);


STATIC void py_tf_profile_output_data_callback(void *callback_data,
                                                void *model_output,
                                                const unsigned int output_height,
                                                const unsigned int output_width,
                                                const unsigned int output_channels,
                                                const bool signed_or_unsigned,
                                                const bool is_float)
{
 
}

void py_tf_profile_print(const char* format, ...)
{ 
  va_list args;
  va_start(args, format);
  mp_vprintf(&mp_plat_print, format,args);
  va_end(args);
}

STATIC mp_obj_t py_tf_classify_profile(uint n_args, const mp_obj_t *args, mp_map_t *kw_args)
{
	int tick;
    fb_alloc_mark();
    alloc_putchar_buffer();

    py_tf_model_obj_t *arg_model = py_tf_load_alloc(args[0]);
    image_t *arg_img = py_helper_arg_to_image_mutable(args[1]);
	
	int offset = py_helper_keyword_int(n_args, args, 7, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_offset), 128);	
	int fscale = py_helper_keyword_int(n_args, args, 8, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_scale), 128);

    arg_model->params.tcm_arena = fb_alloc_all(&arg_model->params.tcm_arena_size, FB_ALLOC_PREFER_SPEED);
    arg_model->params.ocram_arena = (uint8_t *)ocram_area;
    arg_model->params.ocram_arena_size =  ocram_area_end - ocram_area;

    uint8_t *tensor_arena = fb_alloc(arg_model->params.tensor_arena_size, FB_ALLOC_PREFER_SIZE);

    mp_obj_t objects_list = mp_obj_new_list(0, NULL);

    arg_model->params.GetCurrentTicks = rt_tick_get_us;
	arg_model->params.profile_enable = 1;
	arg_model->params.print_func = py_tf_profile_print;
	
    py_tf_input_data_callback_data_t py_tf_input_data_callback_data;
    rectangle_t new_roi;
    rectangle_init(&new_roi, 0, 0, arg_img->w, arg_img->h);
    py_tf_input_data_callback_data.img = arg_img;
    py_tf_input_data_callback_data.roi = &new_roi;
    py_tf_input_data_callback_data.offset = offset;
    py_tf_input_data_callback_data.scale = fscale;
    py_tf_classify_output_data_callback_data_t py_tf_classify_output_data_callback_data;
    PY_ASSERT_FALSE_MSG(libtf_invoke(arg_model->model_data,
                                        tensor_arena,
                                        &arg_model->params,
                                        py_tf_input_data_callback,
                                        &py_tf_input_data_callback_data,
                                        py_tf_profile_output_data_callback,
                                        &py_tf_classify_output_data_callback_data),
                        py_tf_putchar_buffer - (PY_TF_PUTCHAR_BUFFER_LEN - py_tf_putchar_buffer_len));


	mp_printf(&mp_plat_print, "%s",py_tf_putchar_buffer);
    arg_model->params.profile_enable = 0;

    fb_alloc_free_till_mark();

    return objects_list;
}


STATIC MP_DEFINE_CONST_FUN_OBJ_KW(py_tf_classify_profile_obj, 2, py_tf_classify_profile);

typedef struct py_tf_detect_output_data_callback_data {
    mp_obj_t out;
} py_tf_detect_output_data_callback_data_t;

STATIC void py_tf_detect_output_data_callback(void *callback_data,
                                                void *model_output,
                                                libtf_parameters_t *params)
{
    py_tf_classify_output_data_callback_data_t *arg = (py_tf_classify_output_data_callback_data_t *) callback_data;
    int shift = (params->input_datatype == LIBTF_DATATYPE_INT8) ? GRAYSCALE_MID : 0;
	
	if (params->output_channels != 4)
		PY_ASSERT_TRUE_MSG(false,"not a detect model or not have post processing node");
	// here is a OD outputs, with 4 output tensor
	float** outputs = (float**)model_output;
	
	float* boxes = *outputs++;
	float* labels = *outputs++;
	float* scores = *outputs++;
	float* nums = *outputs;
	
    arg->out = mp_obj_new_list((uint32_t)(*nums), NULL);
	
	// all in float
    for (unsigned int i = 0; i < (*nums); i++) {
		float* box = boxes + 4 * i;
		mp_obj_t tmp_list = mp_obj_new_list(6, NULL);
		((mp_obj_list_t *)tmp_list)->items[0] = mp_obj_new_float(box[0]);
		((mp_obj_list_t *)tmp_list)->items[1] = mp_obj_new_float(box[1]);
		((mp_obj_list_t *)tmp_list)->items[2] = mp_obj_new_float(box[2]);
		((mp_obj_list_t *)tmp_list)->items[3] = mp_obj_new_float(box[3]);
		((mp_obj_list_t *)tmp_list)->items[4] = mp_obj_new_float(labels[i]);
		((mp_obj_list_t *)tmp_list)->items[5] = mp_obj_new_float(scores[i]);
        ((mp_obj_list_t *) arg->out)->items[i] = tmp_list;
    }
}

void py_tf_debug(const char* format, ...)
{
    va_list args;
    va_start(args, format);
	mp_vprintf(&mp_plat_print, format,args);
    va_end(args);
}

STATIC mp_obj_t py_tf_detect(uint n_args, const mp_obj_t *args, mp_map_t *kw_args)
{
    fb_alloc_mark();
    alloc_putchar_buffer();

    py_tf_model_obj_t *arg_model = py_tf_load_alloc(args[0]);
    image_t *arg_img = py_image_cobj(args[1]);
	
	rectangle_t roi;
    py_helper_keyword_rectangle_roi(arg_img, n_args, args, 2, kw_args, &roi);
	
	int offset = py_helper_keyword_int(n_args, args, 3, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_offset), 128);	
	int fscale = py_helper_keyword_int(n_args, args, 4, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_scale), 128);
	int bgr_mode = py_helper_keyword_int(n_args, args, 5, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_bgr), 0);
	
    arg_model->params.tcm_arena = fb_alloc_all(&arg_model->params.tcm_arena_size, FB_ALLOC_PREFER_SPEED);
    arg_model->params.ocram_arena = (uint8_t *)ocram_area;
    arg_model->params.ocram_arena_size =  ocram_area_end - ocram_area;

    uint8_t *tensor_arena = fb_alloc(arg_model->params.tensor_arena_size, FB_ALLOC_PREFER_SIZE);

	rectangle_t new_roi;
	rectangle_init(&new_roi, 0, 0, roi.w, roi.h);

	py_tf_input_data_callback_data_t py_tf_input_data_callback_data;
	py_tf_input_data_callback_data.img = arg_img;
	py_tf_input_data_callback_data.roi = &new_roi;
	py_tf_input_data_callback_data.offset = offset;
	py_tf_input_data_callback_data.scale = fscale;
	py_tf_input_data_callback_data.bgr_mode = bgr_mode;
	
	py_tf_detect_output_data_callback_data_t py_tf_detect_output_data_callback_data;
	PY_ASSERT_FALSE_MSG(libtf_invoke(arg_model->model_data,
									 tensor_arena,
									 &arg_model->params,
									 py_tf_input_data_callback,
									 &py_tf_input_data_callback_data,
									 py_tf_detect_output_data_callback,
									 &py_tf_detect_output_data_callback_data),
						py_tf_putchar_buffer - (PY_TF_PUTCHAR_BUFFER_LEN - py_tf_putchar_buffer_len));

	mp_obj_t objects_list = py_tf_detect_output_data_callback_data.out;
		

    fb_alloc_free_till_mark();

    return objects_list;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_KW(py_tf_detect_obj, 2, py_tf_detect);


STATIC void py_tf_fastest_det_output_data_callback(void *callback_data,
                                                void *model_output,
                                                libtf_parameters_t *params)
{
    py_tf_classify_output_data_callback_data_t *arg = (py_tf_classify_output_data_callback_data_t *) callback_data;
    libtf_fastest_det_output_data_t *results = (libtf_fastest_det_output_data_t*)model_output;
	
	
    arg->out = mp_obj_new_list((uint32_t)(results->num), NULL);
	
	// all in float
    for (unsigned int i = 0; i < (results->num); i++) {
		
		mp_obj_t tmp_list = mp_obj_new_list(6, NULL);
		((mp_obj_list_t *)tmp_list)->items[0] = mp_obj_new_float(results->results[i].x1);
		((mp_obj_list_t *)tmp_list)->items[1] = mp_obj_new_float(results->results[i].y1);
		((mp_obj_list_t *)tmp_list)->items[2] = mp_obj_new_float(results->results[i].x2);
		((mp_obj_list_t *)tmp_list)->items[3] = mp_obj_new_float(results->results[i].y2);
		((mp_obj_list_t *)tmp_list)->items[4] = mp_obj_new_float(results->results[i].label);
		((mp_obj_list_t *)tmp_list)->items[5] = mp_obj_new_float(results->results[i].score);
        ((mp_obj_list_t *) arg->out)->items[i] = tmp_list;
    }
}
STATIC mp_obj_t py_tf_fastest_det(uint n_args, const mp_obj_t *args, mp_map_t *kw_args)
{
    fb_alloc_mark();
    alloc_putchar_buffer();

    py_tf_model_obj_t *arg_model = py_tf_load_alloc(args[0]);
    image_t *arg_img = py_image_cobj(args[1]);
	
	rectangle_t roi;
    py_helper_keyword_rectangle_roi(arg_img, n_args, args, 2, kw_args, &roi);
	
	int offset = py_helper_keyword_int(n_args, args, 3, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_offset), 128);	
	int fscale = py_helper_keyword_int(n_args, args, 4, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_scale), 128);
	int bgr_mode = py_helper_keyword_int(n_args, args, 5, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_bgr), 1);
    float threshold = py_helper_keyword_float(n_args, args, 6, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_threshold), 0.45);
    float score_threshold = py_helper_keyword_float(n_args, args, 7, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_score_threshold), 0.70);
    int topn = py_helper_keyword_int(n_args, args, 8, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_topn), 0);
	
    arg_model->params.tcm_arena = fb_alloc_all(&arg_model->params.tcm_arena_size, FB_ALLOC_PREFER_SPEED);
    arg_model->params.ocram_arena = (uint8_t *)ocram_area;
    arg_model->params.ocram_arena_size =  ocram_area_end - ocram_area;

    uint8_t *tensor_arena = fb_alloc(arg_model->params.tensor_arena_size, FB_ALLOC_PREFER_SIZE);

	rectangle_t new_roi;
	rectangle_init(&new_roi, 0, 0, roi.w, roi.h);

	py_tf_input_data_callback_data_t py_tf_input_data_callback_data;
	py_tf_input_data_callback_data.img = arg_img;
	py_tf_input_data_callback_data.roi = &new_roi;
	py_tf_input_data_callback_data.offset = offset;
	py_tf_input_data_callback_data.scale = fscale;
	py_tf_input_data_callback_data.bgr_mode = bgr_mode;
	
    libtf_fastest_det_parameters_t fastdet_param;
    fastdet_param.originalImageWidth = 1;
	fastdet_param.originalImageHeight = 1;
	fastdet_param.threshold = threshold;
	fastdet_param.score_thres = score_threshold;
	fastdet_param.nms = 0.45;
	fastdet_param.topN = topn;

	py_tf_detect_output_data_callback_data_t py_tf_fastest_output_data_callback_data;
	PY_ASSERT_FALSE_MSG(libtf_fastdet(arg_model->model_data,
									 tensor_arena,
									 &arg_model->params,
                                     &fastdet_param,
									 &py_tf_input_data_callback,
									 &py_tf_input_data_callback_data,
									 py_tf_fastest_det_output_data_callback,
									 &py_tf_fastest_output_data_callback_data),
						py_tf_putchar_buffer - (PY_TF_PUTCHAR_BUFFER_LEN - py_tf_putchar_buffer_len));

	mp_obj_t objects_list = py_tf_fastest_output_data_callback_data.out;
		

    fb_alloc_free_till_mark();

    return objects_list;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_KW(py_tf_fastest_det_obj, 2, py_tf_fastest_det);
typedef struct py_tf_segment_output_data_callback_data {
    mp_obj_t out;
} py_tf_segment_output_data_callback_data_t;




STATIC void py_tf_segment_output_data_callback(void *callback_data,
                                               void *model_output,
                                               libtf_parameters_t *params)
{
    py_tf_segment_output_data_callback_data_t *arg = (py_tf_segment_output_data_callback_data_t *) callback_data;
    int shift = (params->input_datatype == LIBTF_DATATYPE_INT8) ? GRAYSCALE_MID : 0;

    arg->out = mp_obj_new_list(params->output_channels, NULL);
    for (unsigned int i = 0; i < params->output_channels; i++) {
        image_t img = {
            .w = params->output_width,
            .h = params->output_height,
            .pixfmt = PIXFORMAT_GRAYSCALE,
            .pixels = xalloc(params->output_width * params->output_height * sizeof(uint8_t))
        };
        ((mp_obj_list_t *) arg->out)->items[i] = py_image_from_struct(&img);
        for (unsigned int y = 0; y < params->output_height; y++) {
            unsigned int row = y * params->output_width * params->output_channels;
            uint8_t *row_ptr = IMAGE_COMPUTE_GRAYSCALE_PIXEL_ROW_PTR(&img, y);
            for (unsigned int x = 0; x < params->output_width; x++) {
                unsigned int col = x * params->output_channels;
                if (params->input_datatype != LIBTF_DATATYPE_FLOAT) {
                    IMAGE_PUT_GRAYSCALE_PIXEL_FAST(row_ptr, x, ((uint8_t *) model_output)[row + col + i] ^ shift);
                } else {
                    IMAGE_PUT_GRAYSCALE_PIXEL_FAST(row_ptr, x, ((float *) model_output)[i] * 255);
                }
            }
        }
    }
}

STATIC mp_obj_t py_tf_segment(uint n_args, const mp_obj_t *args, mp_map_t *kw_args)
{
    fb_alloc_mark();
    alloc_putchar_buffer();

    py_tf_model_obj_t *arg_model = py_tf_load_alloc(args[0]);
    image_t *arg_img = py_helper_arg_to_image_mutable(args[1]);

    rectangle_t roi;
    py_helper_keyword_rectangle_roi(arg_img, n_args, args, 2, kw_args, &roi);

    arg_model->params.tcm_arena = fb_alloc_all(&arg_model->params.tcm_arena_size, FB_ALLOC_PREFER_SPEED);
    arg_model->params.ocram_arena = (uint8_t *)ocram_area;
    arg_model->params.ocram_arena_size =  ocram_area_end - ocram_area;

    uint8_t *tensor_arena = fb_alloc(arg_model->params.tensor_arena_size, FB_ALLOC_PREFER_SIZE);

    py_tf_input_data_callback_data_t py_tf_input_data_callback_data;
    py_tf_input_data_callback_data.img = arg_img;
    py_tf_input_data_callback_data.roi = &roi;

    py_tf_segment_output_data_callback_data_t py_tf_segment_output_data_callback_data;

    PY_ASSERT_FALSE_MSG(libtf_invoke(arg_model->model_data,
                                     tensor_arena,
                                     &arg_model->params,
                                     py_tf_input_data_callback,
                                     &py_tf_input_data_callback_data,
                                     py_tf_segment_output_data_callback,
                                     &py_tf_segment_output_data_callback_data),
                        py_tf_putchar_buffer - (PY_TF_PUTCHAR_BUFFER_LEN - py_tf_putchar_buffer_len));

    fb_alloc_free_till_mark();

    return py_tf_segment_output_data_callback_data.out;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_KW(py_tf_segment_obj, 2, py_tf_segment);


typedef struct py_tf_invoke_output_data_callback_data {
    mp_obj_t out;
} py_tf_invoke_output_data_callback_data_t;

#define py_tf_invoke_obj_size 3
typedef struct py_tf_invoke_obj {
    mp_obj_base_t base;
    mp_obj_t during, output;
} py_tf_invoke_obj_t;


STATIC void py_tf_invoke_print(const mp_print_t *print, mp_obj_t self_in, mp_print_kind_t kind)
{
    py_tf_invoke_obj_t *self = self_in;
    mp_printf(print,
              "{\"during\":%dms",
              mp_obj_get_int(self->during));
    mp_obj_print_helper(print, self->output, kind);
    mp_printf(print, "}");
}

STATIC mp_obj_t py_tf_invoke_subscr(mp_obj_t self_in, mp_obj_t index, mp_obj_t value)
{
    if (value == MP_OBJ_SENTINEL) { // load
        py_tf_invoke_obj_t *self = self_in;
        if (MP_OBJ_IS_TYPE(index, &mp_type_slice)) {
            mp_bound_slice_t slice;
            if (!mp_seq_get_fast_slice_indexes(py_tf_invoke_obj_size, index, &slice)) {
                nlr_raise(mp_obj_new_exception_msg(&mp_type_OSError, "only slices with step=1 (aka None) are supported"));
            }
            mp_obj_tuple_t *result = mp_obj_new_tuple(slice.stop - slice.start, NULL);
            mp_seq_copy(result->items, &(self->during) + slice.start, result->len, mp_obj_t);
            return result;
        }
        switch (mp_get_index(self->base.type, py_tf_invoke_obj_size, index, false)) {
            case 0: return self->during;
            case 2: return self->output;
        }
    }
    return MP_OBJ_NULL; // op not supported
}

mp_obj_t py_tf_invoke_during(mp_obj_t self_in) { return ((py_tf_invoke_obj_t *) self_in)->during; }
mp_obj_t py_tf_invoke_output(mp_obj_t self_in) { return ((py_tf_invoke_obj_t *) self_in)->output; }

STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_invoke_during_obj, py_tf_invoke_during);
STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_invoke_output_obj, py_tf_invoke_output);

STATIC const mp_rom_map_elem_t py_tf_invoke_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_during), MP_ROM_PTR(&py_tf_invoke_during_obj) },
    { MP_ROM_QSTR(MP_QSTR_output), MP_ROM_PTR(&py_tf_invoke_output_obj) }
};

STATIC MP_DEFINE_CONST_DICT(py_tf_invoke_locals_dict, py_tf_invoke_locals_dict_table);

static const mp_obj_type_t py_tf_invoke_type = {
    { &mp_type_type },
    .name  = MP_QSTR_tf_invoke,
    .print = py_tf_invoke_print,
    .subscr = py_tf_invoke_subscr,
    .locals_dict = (mp_obj_t) &py_tf_invoke_locals_dict
};
STATIC void py_tf_invoke_output_data_callback(void *callback_data,
                                                void *model_output,
                                                libtf_parameters_t *params)
{

    int shift = (params->output_datatype == LIBTF_DATATYPE_INT8) ? GRAYSCALE_MID : 0;
    py_tf_invoke_output_data_callback_data_t *arg = (py_tf_invoke_output_data_callback_data_t *) callback_data;
    uint32_t size = params->output_height * params->output_width * params->output_channels;
	float* buffer = (float*)xalloc(sizeof(float) * size);
    
    for (unsigned int i = 0; i < params->output_channels; i++) 
	{
        if (params->output_datatype != LIBTF_DATATYPE_FLOAT) {
            buffer[i] = (((uint8_t *) model_output)[i] ^ shift) / 255.0f;
        } else {
            buffer[i] = ((float *) model_output)[i];
        }
    }
	
	mp_obj_list_t* output_list = mp_obj_new_list(2, NULL);
	
	mp_obj_list_t* list_shape = mp_obj_new_list(4, NULL);
	list_shape->items[0] = mp_obj_new_int(1);
	list_shape->items[1] = mp_obj_new_int(params->output_height);
	list_shape->items[2] = mp_obj_new_int(params->output_width);
	list_shape->items[3] = mp_obj_new_int(params->output_channels);
	
	output_list->items[0] = mp_obj_new_bytearray_by_ref(size*sizeof(float), buffer);
	output_list->items[1] = list_shape;
		
	mp_obj_list_append(arg->out, output_list);
	
	return;
}
STATIC mp_obj_t py_tf_invoke(uint n_args, const mp_obj_t *args, mp_map_t *kw_args)
{
    fb_alloc_mark();
    alloc_putchar_buffer();

    py_tf_model_obj_t *arg_model = py_tf_load_alloc(args[0]);
    image_t *arg_img = py_image_cobj(args[1]);
	
	rectangle_t roi;
    py_helper_keyword_rectangle_roi(arg_img, n_args, args, 2, kw_args, &roi);
	
	int offset = py_helper_keyword_int(n_args, args, 3, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_offset), 128);	
	int fscale = py_helper_keyword_int(n_args, args, 4, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_scale), 128);
	int bgr_mode = py_helper_keyword_int(n_args, args, 5, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_bgr), 0);
	
    arg_model->params.tcm_arena = fb_alloc_all(&arg_model->params.tcm_arena_size, FB_ALLOC_PREFER_SPEED);
    arg_model->params.ocram_arena = (uint8_t *)ocram_area;
    arg_model->params.ocram_arena_size =  ocram_area_end - ocram_area;
    
    uint8_t *tensor_arena = fb_alloc(arg_model->params.tensor_arena_size , FB_ALLOC_PREFER_SIZE);

	rectangle_t new_roi;
	rectangle_init(&new_roi, 0, 0, roi.w, roi.h);

	py_tf_input_data_callback_data_t py_tf_input_data_callback_data;
	py_tf_input_data_callback_data.img = arg_img;
	py_tf_input_data_callback_data.roi = &new_roi;
	py_tf_input_data_callback_data.offset = offset;
	py_tf_input_data_callback_data.scale = fscale;
	py_tf_input_data_callback_data.bgr_mode = bgr_mode;
	
	py_tf_invoke_output_data_callback_data_t py_tf_invoke_output_data_callback_data;
    py_tf_invoke_output_data_callback_data.out = mp_obj_new_list(0, NULL);
    g_tf_profiling_en = 1;
	PY_ASSERT_FALSE_MSG(libtf_invoke(arg_model->model_data,
									 tensor_arena,
									 &arg_model->params,
									 py_tf_input_data_callback,
									 &py_tf_input_data_callback_data,
									 py_tf_invoke_output_data_callback,
									 &py_tf_invoke_output_data_callback_data),
						py_tf_putchar_buffer - (PY_TF_PUTCHAR_BUFFER_LEN - py_tf_putchar_buffer_len));		
    g_tf_profiling_en = 0;

    py_tf_invoke_obj_t *o = m_new_obj(py_tf_invoke_obj_t);
    o->base.type = &py_tf_invoke_type;
    o->during = mp_obj_new_int(g_usTotal/1000);
    o->output = py_tf_invoke_output_data_callback_data.out;

    fb_alloc_free_till_mark();
    g_tsNdx=0;
	g_usTotal=0;
	g_ts0=0;
    
    return o;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_KW(py_tf_invoke_obj, 2, py_tf_invoke);



uint8_t *ResizeRgb565(uint8_t *pIn, uint8_t *pOut, int w0, int h0, int w1, int h1) {
	uint8_t *p = (uint8_t*)pOut;
	uint8_t *pRet = p;
	const uint8_t *p0;
	int y, x, ySrc, xSrc;
	float fySrc, fxSrc;
	float dx = (float)(w0) / (float)(w1);
	float dy = (float)(h0) / (float)(h1);
	for (y=0, fySrc = 0; y<h1; y++, fySrc += dy) {
		
		for (x=0, fxSrc = 0;x<w1; x++, fxSrc += dx) {
			xSrc = (int)(fxSrc);
			ySrc = (int)(fySrc);
			p0 = pIn + (w0 * ySrc + xSrc) * 2;
			*p++ = *p0++;
			*p++ = *p0++;
		}
	}
	return pRet;
}

static mp_obj_t py_image_resize(uint n_args, const mp_obj_t *args)
{
    image_t *arg_img = py_helper_arg_to_image_mutable(args[0]);

    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    image_t image;

    image.w = w;
    image.h = h;
    image.pixfmt = arg_img->pixfmt;
    image.data = xalloc(image_size(&image));

    if (image.pixfmt  != PIXFORMAT_RGB565)
    {
        PY_ASSERT_TRUE_MSG((image_size(&image) <= fb_avail()), "The image is not RGB565!");
    }

    ResizeRgb565(arg_img->data, image.data, arg_img->w,arg_img->h,w,h);
    return py_image_from_struct(&image);
}
STATIC MP_DEFINE_CONST_FUN_OBJ_KW(py_image_resize_obj, 1, py_image_resize);


mp_obj_t py_tf_len(mp_obj_t self_in) { return mp_obj_new_int(((py_tf_model_obj_t *) self_in)->model_data_len); }
mp_obj_t py_tf_input_height(mp_obj_t self_in) { return mp_obj_new_int(((py_tf_model_obj_t *) self_in)->params.input_height); }
mp_obj_t py_tf_input_width(mp_obj_t self_in) { return mp_obj_new_int(((py_tf_model_obj_t *) self_in)->params.input_width); }
mp_obj_t py_tf_input_channels(mp_obj_t self_in) { return mp_obj_new_int(((py_tf_model_obj_t *) self_in)->params.input_channels); }
mp_obj_t py_tf_input_datatype(mp_obj_t self_in) { return mp_obj_new_int(((py_tf_model_obj_t *) self_in)->params.input_datatype); }

mp_obj_t py_tf_output_height(mp_obj_t self_in) { return mp_obj_new_int(((py_tf_model_obj_t *) self_in)->params.output_height); }
mp_obj_t py_tf_output_width(mp_obj_t self_in) { return mp_obj_new_int(((py_tf_model_obj_t *) self_in)->params.output_width); }
mp_obj_t py_tf_output_channels(mp_obj_t self_in) { return mp_obj_new_int(((py_tf_model_obj_t *) self_in)->params.output_channels); }
mp_obj_t py_tf_output_datatype(mp_obj_t self_in) { return mp_obj_new_int(((py_tf_model_obj_t *) self_in)->params.output_datatype); }

STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_len_obj, py_tf_len);
STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_input_height_obj, py_tf_input_height);
STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_input_width_obj, py_tf_input_width);
STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_input_channels_obj, py_tf_input_channels);
STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_input_datatype_obj, py_tf_input_datatype);
STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_output_height_obj, py_tf_output_height);
STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_output_width_obj, py_tf_output_width);
STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_output_channels_obj, py_tf_output_channels);
STATIC MP_DEFINE_CONST_FUN_OBJ_1(py_tf_output_datatype_obj, py_tf_output_datatype);

STATIC const mp_rom_map_elem_t locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_len), MP_ROM_PTR(&py_tf_len_obj) },
    { MP_ROM_QSTR(MP_QSTR_input_height), MP_ROM_PTR(&py_tf_input_height_obj) },
    { MP_ROM_QSTR(MP_QSTR_input_width), MP_ROM_PTR(&py_tf_input_width_obj) },
    { MP_ROM_QSTR(MP_QSTR_input_channels), MP_ROM_PTR(&py_tf_input_channels_obj) },
    { MP_ROM_QSTR(MP_QSTR_input_datatype), MP_ROM_PTR(&py_tf_input_datatype_obj) },

    { MP_ROM_QSTR(MP_QSTR_output_height), MP_ROM_PTR(&py_tf_output_height_obj) },
    { MP_ROM_QSTR(MP_QSTR_output_width), MP_ROM_PTR(&py_tf_output_width_obj) },
    { MP_ROM_QSTR(MP_QSTR_output_channels), MP_ROM_PTR(&py_tf_output_channels_obj) },
    { MP_ROM_QSTR(MP_QSTR_output_datatype), MP_ROM_PTR(&py_tf_output_datatype_obj) },
};

STATIC MP_DEFINE_CONST_DICT(locals_dict, locals_dict_table);

STATIC const mp_obj_type_t py_tf_model_type = {
    { &mp_type_type },
    .name  = MP_QSTR_tf_model,
    .print = py_tf_model_print,
    .locals_dict = (mp_obj_t) &locals_dict
};

#endif // IMLIB_ENABLE_TF

STATIC const mp_rom_map_elem_t globals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_OBJ_NEW_QSTR(MP_QSTR_tf) },
#ifdef IMLIB_ENABLE_TF
    { MP_ROM_QSTR(MP_QSTR_load),            MP_ROM_PTR(&py_tf_load_obj) },
    { MP_ROM_QSTR(MP_QSTR_free_from_fb),    MP_ROM_PTR(&py_tf_free_from_fb_obj) },
    { MP_ROM_QSTR(MP_QSTR_classify),        MP_ROM_PTR(&py_tf_classify_obj) },
    { MP_ROM_QSTR(MP_QSTR_logistic),        MP_ROM_PTR(&py_tf_logistic_obj) },
    { MP_ROM_QSTR(MP_QSTR_regress),        MP_ROM_PTR(&py_tf_logistic_obj) },
    { MP_ROM_QSTR(MP_QSTR_profile),         MP_ROM_PTR(&py_tf_classify_profile_obj)},
    { MP_ROM_QSTR(MP_QSTR_detect),          MP_ROM_PTR(&py_tf_detect_obj) },
    { MP_ROM_QSTR(MP_QSTR_fastdetect),          MP_ROM_PTR(&py_tf_fastest_det_obj) },
    { MP_ROM_QSTR(MP_QSTR_segment),         MP_ROM_PTR(&py_tf_segment_obj) },
    { MP_ROM_QSTR(MP_QSTR_invoke),         MP_ROM_PTR(&py_tf_invoke_obj) },
	{ MP_ROM_QSTR(MP_QSTR_image_resize),MP_ROM_PTR(&py_image_resize_obj) },
	
#else
    { MP_ROM_QSTR(MP_QSTR_load),            MP_ROM_PTR(&py_func_unavailable_obj) },
    { MP_ROM_QSTR(MP_QSTR_free_from_fb),    MP_ROM_PTR(&py_func_unavailable_obj) },
    { MP_ROM_QSTR(MP_QSTR_classify),        MP_ROM_PTR(&py_func_unavailable_obj) },
    { MP_ROM_QSTR(MP_QSTR_segment),         MP_ROM_PTR(&py_func_unavailable_obj) }
#endif // IMLIB_ENABLE_TF
};

STATIC MP_DEFINE_CONST_DICT(globals_dict, globals_dict_table);

const mp_obj_module_t tf_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_t) &globals_dict
};