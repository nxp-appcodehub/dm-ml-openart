/* This file is part of the OpenMV project.
 * Copyright (c) 2013-2019 Ibrahim Abdelkader <iabdalkader@openmv.io> & Kwabena W. Agyeman <kwagyeman@openmv.io>
 * This work is licensed under the MIT license, see the file LICENSE for details.
 */

#ifndef __LIBTF_H
#define __LIBTF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum libtf_datatype {
    LIBTF_DATATYPE_UINT8,
    LIBTF_DATATYPE_INT8,
    LIBTF_DATATYPE_FLOAT
} libtf_datatype_t;

typedef struct libtf_parameters {
    size_t tensor_arena_size;
    size_t input_height, input_width, input_channels;
    libtf_datatype_t input_datatype;
    float input_scale;
    int input_zero_point;
    size_t output_height, output_width, output_channels;
    libtf_datatype_t output_datatype;
    float output_scale;
    int output_zero_point;
	uint8_t *tcm_arena;
	uint32_t tcm_arena_size;
	uint8_t *ocram_arena;
	uint32_t ocram_arena_size;
    int profile_enable;
    uint32_t (*GetCurrentTicks)(void);
    void (*print_func)(const char* format, ...);

} libtf_parameters_t;


typedef struct libtf_fastest_det_parameters{
	int originalImageWidth;
	int originalImageHeight;

	float threshold;
	float score_thres;
	float nms;
	int topN;

}libtf_fastest_det_parameters_t;

typedef struct libtf_box{
	float x1,y1,x2,y2;
	float score;
	unsigned int label;
}libtf_box_t;

typedef struct libtf_fastest_det_output_data{
	int num;
	libtf_box_t results[20];
}libtf_fastest_det_output_data_t;

// Call this first to get the shape of the model input.
// Returns 0 on success and 1 on failure.
// Errors are printed to stdout.
int libtf_get_parameters(const unsigned char *model_data, // TensorFlow Lite binary model (8-bit quant).
                             unsigned char *tensor_arena, // As big as you can make it scratch buffer.
                             unsigned int tensor_arena_size, // Size of the above scratch buffer.
                             libtf_parameters_t *params); // Struct to hold model parameters.


// Callback to populate the model input data byte array (laid out in [height][width][channel] order).
typedef void (*libtf_input_data_callback_t)(void *callback_data,
                                            void *model_input,
                                            libtf_parameters_t *params); // Actual is float32 (not optimal - network should be fixed). Input should be ([0:255]->[0.0f:+1.0f]).

// Callback to use the model output data byte array (laid out in [height][width][channel] order).
typedef void (*libtf_output_data_callback_t)(void *callback_data,
                                             void *model_output,
                                             libtf_parameters_t *params); // Actual is float32 (not optimal - network should be fixed). Output is [0.0f:+1.0f].

// Returns 0 on success and 1 on failure.
// Errors are printed to stdout.
int libtf_invoke(const unsigned char *model_data, // TensorFlow Lite binary model (8-bit quant).
                 unsigned char *tensor_arena, // As big as you can make it scratch buffer.
                 libtf_parameters_t *params, // Size of the above scratch buffer.
                 libtf_input_data_callback_t input_callback, // Callback to populate the model input data byte array.
                 void *input_callback_data, // User data structure passed to input callback.
                 libtf_output_data_callback_t output_callback, // Callback to use the model output data byte array.
                 void *output_callback_data); // User data structure passed to output callback.

// Returns 0 on success and 1 on failure.
// Errors are printed to stdout.
int libtf_initialize_micro_features();

// Returns 0 on success and 1 on failure.
// Errors are printed to stdout.
// Converts audio sample data into a more compact form that's
// appropriate for feeding into a neural network.
int libtf_generate_micro_features(const int16_t* input, // Audio samples
                                  int input_size, // Audio samples size
                                  int output_size, // Slice size
                                  int8_t* output, // Slice data
                                  size_t* num_samples_read); // Number of samples used.

int libtf_fastdet(const unsigned char *model_data,
                     unsigned char *tensor_arena, libtf_parameters_t *params,
				     libtf_fastest_det_parameters_t *fastdet_params,
                     libtf_input_data_callback_t input_callback, void *input_callback_data,
                     libtf_output_data_callback_t output_callback, void *output_callback_data);
#ifdef __cplusplus
}
#endif

#endif // __LIBTF_H