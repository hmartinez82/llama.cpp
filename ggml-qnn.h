#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif


#define GGML_QNN_NAME           "QNN"
#define GGML_QNN_MAX_DEVICES    1

//Only support for QNN HTP backends for now
enum QNNBackend {
    QNN_HTP
    //QNN_CPU
    //QNN_GPU
};

// Provides GGML description of the QNN backend for the given device number
GGML_API ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(size_t device_index);

/**
 *
 * @param device            0: QNN_HTP, 1: QNN_CPU 
 * @param qnn_lib_path      QNN library path where the dynamic library files can be found
 *                          which can be obtained through JNI from Java layer on Android
 * @return
 */
GGML_API ggml_backend_t ggml_backend_qnn_init(size_t dev_num, const char * qnn_lib_path);

GGML_API bool           ggml_backend_is_qnn(ggml_backend_t backend);

GGML_API void           ggml_backend_qnn_set_n_threads(ggml_backend_t backend, int n_threads);

GGML_API int            ggml_backend_qnn_get_device_count(void);
GGML_API void           ggml_backend_qnn_get_device_description(int device, char * description, size_t description_size);

GGML_API int            ggml_backend_qnn_reg_devices(void);

#ifdef __cplusplus
}
#endif
