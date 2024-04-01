/*
 * Copyright (c) 2006-2018, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */
#include "py/obj.h"
#include "py/runtime.h"
#include "py_helper.h"
#include "py_image.h"
#include "ff_wrapper.h"
#include "py/objarray.h"
#include "py_assert.h"
#include "ndarray.h"

// nxp Model Object
typedef struct py_nxp_module_obj {
    mp_obj_base_t base;
    unsigned int tensor_buffer_len;
    unsigned int tensor_len;
} py_nxp_module_obj_t;

static struct py_nxp_module_obj nxp_obj;

STATIC mp_obj_t py_tf_alloc_float_tensor(uint n_args, const mp_obj_t *args, mp_map_t *kw_args)
{    
    int len = py_helper_keyword_int(n_args, args, 1, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_len), 128);
	int size = len*sizeof(float);
	
    void* buf = (float *)gc_alloc(size,0);
    memset(buf,0x0,size);
    nxp_obj.tensor_buffer_len = len;

    return mp_obj_new_bytearray_by_ref(len, buf);
}
STATIC MP_DEFINE_CONST_FUN_OBJ_KW(py_tf_alloc_obj, 0, py_tf_alloc_float_tensor);

STATIC mp_obj_t py_tf_add_float_tensor(uint n_args, const mp_obj_t *args, mp_map_t *kw_args)
{
    mp_obj_array_t *o;
	mp_map_elem_t * kw_arg = mp_map_lookup(kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_base), MP_MAP_LOOKUP);
	if(kw_arg)
	{
		o = (mp_obj_array_t*)kw_arg->value;
	}
	
    if(!mp_obj_is_type(o,&mp_type_bytearray))
	{
		nlr_raise(mp_obj_new_exception_msg(&mp_type_ValueError, "base is not byte array"));
	}
	void *tensor_buffer = (void*)o->items;
    if(tensor_buffer == NULL)
    {
        nlr_raise(mp_obj_new_exception_msg(&mp_type_ValueError, "TF tensor buffer is empty"));
    }
	
    if(tensor_buffer == NULL)
    {
        nlr_raise(mp_obj_new_exception_msg(&mp_type_ValueError, "TF tensor buffer is empty"));
    }

    int offset = py_helper_keyword_int(n_args, args, 3, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_offset), 0);
    int len = py_helper_keyword_int(n_args, args, 4, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_len), 128);
    nxp_obj.tensor_len = len;
	
	if(offset + len > o->len)
		nlr_raise(mp_obj_new_exception_msg(&mp_type_ValueError, "TF tensor buffer overflow"));
	
    float *buffer_addr = &(tensor_buffer[offset]);
	
	kw_arg = mp_map_lookup(kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_tensor), MP_MAP_LOOKUP);
	if(kw_arg)
	{
		o = (mp_obj_array_t*)kw_arg->value;
	}
	if(mp_obj_is_type(o,&ulab_ndarray_type))
	{
		ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(o);
		for (int i=0;i<len;i++)
		{
			float *array = (float*)ndarray->array;
			buffer_addr[i] = array[i];
		}
		
	}else
	{
		py_helper_keyword_float_array(n_args, args, 2, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_tensor), buffer_addr, len);
	}
    return mp_obj_new_int(len);
}
STATIC MP_DEFINE_CONST_FUN_OBJ_KW(py_tf_add_float_tensor_obj, 0, py_tf_add_float_tensor);

float py_tf_calc_tensor_angle(float *tensor_x, float *tensor_y, int len)
{
    float iM1, iM2, iDot;
	float fM1, fM2, fDot;

    iM1 = 0;
	iM2 = 0;
	iDot = 0;
	
    for (int i=0;i<len;i++)
    {
		int8_t x = (int8_t)(tensor_x[i]*10);
		int8_t y = (int8_t)(tensor_y[i]*10);
		
        iDot += x*y;
		iM1 += x*x;
		iM2 +=  y*y;
    }
    fDot = (float) iDot;
	fM1 = sqrtf((float)iM1);
	fM2 = sqrtf((float)iM2);
	if(fM1 ==0.0) fM1 = 1;
	if(fM2 ==0.0) fM2 = 1;
	float cosVal = fDot / (fM1 * fM2);
	float angle = acosf(cosVal) * 180 / 3.141592654f;
    return angle;
}
STATIC mp_obj_t py_tf_compare_float_tensor(uint n_args, const mp_obj_t *args, mp_map_t *kw_args)
{
    int len = py_helper_keyword_int(n_args, args, 3, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_len), 128);
   
    mp_map_elem_t *kw_arg = mp_map_lookup(kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_tensor), MP_MAP_LOOKUP);
	float *tensor;
	mp_obj_array_t *o;
	if(kw_arg)
	{
		o = (mp_obj_array_t*)kw_arg->value;
	}
	
    if(!mp_obj_is_type(o,&mp_type_bytearray))
	{
		nlr_raise(mp_obj_new_exception_msg(&mp_type_ValueError, "tensor is not byte array"));
	}
	tensor= (float*)o->items;
	
	kw_arg = mp_map_lookup(kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_base), MP_MAP_LOOKUP);
	if(kw_arg)
	{
		o = (mp_obj_array_t*)kw_arg->value;
	}
	
    if(!mp_obj_is_type(o,&mp_type_bytearray))
	{
		nlr_raise(mp_obj_new_exception_msg(&mp_type_ValueError, "base is not byte array"));
	}
	void *tensor_buffer = (void*)o->items;
    if(tensor_buffer == NULL)
    {
        nlr_raise(mp_obj_new_exception_msg(&mp_type_ValueError, "TF tensor buffer is empty"));
    }
	
    int count = o->len / len/sizeof(float);
    mp_obj_t tmp_list = mp_obj_new_list(count, NULL);
    for (int i=0;i<count;i++)
    {
        float *tensor_buf = &(tensor_buffer[i*len]);
        float angle = py_tf_calc_tensor_angle(tensor_buf,tensor,len);
        ((mp_obj_list_t *)tmp_list)->items[i] = mp_obj_new_float(angle); 
    }

    return tmp_list;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_KW(py_tf_compare_float_tensor_obj, 0, py_tf_compare_float_tensor);


STATIC mp_obj_t py_tf_compare_tensor_angle(uint n_args, const mp_obj_t *args, mp_map_t *kw_args)
{
    int len = py_helper_keyword_int(n_args, args, 3, kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_len), 128);
   
    mp_map_elem_t *kw_arg = mp_map_lookup(kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_tensor), MP_MAP_LOOKUP);
	float *tensor;
	mp_obj_array_t *o;
	if(kw_arg)
	{
		o = (mp_obj_array_t*)kw_arg->value;
	}
	
    if(!mp_obj_is_type(o,&mp_type_bytearray))
	{
		uint32_t t = ((mp_obj_base_t *)MP_OBJ_TO_PTR(o))->type;
		
		nlr_raise(mp_obj_new_exception_msg_varg(&mp_type_ValueError, MP_ERROR_TEXT("tesnor is not byte array :%x:%x"), t,o));
	}
	tensor= (float*)o->items;
	
	kw_arg = mp_map_lookup(kw_args, MP_OBJ_NEW_QSTR(MP_QSTR_base), MP_MAP_LOOKUP);
	if(kw_arg)
	{
		o = (mp_obj_array_t*)kw_arg->value;
	}
	
    if(!mp_obj_is_type(o,&mp_type_bytearray))
	{
		uint32_t t = ((mp_obj_base_t *)MP_OBJ_TO_PTR(o))->type;
		
		nlr_raise(mp_obj_new_exception_msg_varg(&mp_type_ValueError, MP_ERROR_TEXT("base is not byte array :%x:%x"), t,o));
	}
	float *base = (float*)o->items;
    if(base == NULL)
    {
        nlr_raise(mp_obj_new_exception_msg(&mp_type_ValueError, "TF tensor buffer is empty"));
    }
	

    float angle = py_tf_calc_tensor_angle(base,tensor,len);

    

    return mp_obj_new_float(angle);
}
STATIC MP_DEFINE_CONST_FUN_OBJ_KW(py_tf_compare_tensor_angle_obj, 0, py_tf_compare_tensor_angle);

static uint8_t *ResizeRgb565(uint8_t *pIn, uint8_t *pOut, int w0, int h0, int w1, int h1) {
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
    //image_t *arg_img = py_helper_arg_to_image_mutable(args[0]);
	image_t *arg_img = py_image_cobj(args[0]);
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

STATIC const mp_rom_map_elem_t globals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_alloc_float_tensor),MP_ROM_PTR(&py_tf_alloc_obj) },
    { MP_ROM_QSTR(MP_QSTR_add_float_tensor),MP_ROM_PTR(&py_tf_add_float_tensor_obj) },
    { MP_ROM_QSTR(MP_QSTR_compare_float_tensor),MP_ROM_PTR(&py_tf_compare_float_tensor_obj) },
	{ MP_ROM_QSTR(MP_QSTR_compare_tensor_angle),MP_ROM_PTR(&py_tf_compare_tensor_angle_obj) },
	{ MP_ROM_QSTR(MP_QSTR_image_resize),MP_ROM_PTR(&py_image_resize_obj) },

};

STATIC MP_DEFINE_CONST_DICT(globals_dict, globals_dict_table);

const mp_obj_module_t nxp_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_t) &globals_dict
};

MP_REGISTER_MODULE(MP_QSTR_nxp_module, nxp_module, 1);