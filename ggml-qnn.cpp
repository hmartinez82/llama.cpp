#include <memory>
#include <unordered_map>

#include "ggml-qnn.h"
#include "qnn-logging.h"
#include "qnn-wrapper.h"
#include "ggml-backend-impl.h"

static std::unordered_map<size_t, std::shared_ptr<qnn_wrapper>> qnn_backends;

// Global declare the GGML backend interfaces for the QNN backends
static ggml_backend_buffer_type ggml_backend_buffer_types_qnn[GGML_QNN_MAX_DEVICES];

// Forward GGML backend to QNN backend instances
static const char * ggml_backend_qnn_buffer_get_name(ggml_backend_buffer_type_t buft) {
    return reinterpret_cast<qnn_wrapper*>(buft->context)->ggml_backend_qnn_buffer_get_name();
}

static void ggml_backend_qnn_buffer_free_buffer(ggml_backend_buffer_type_t buft) {
    reinterpret_cast<qnn_wrapper*>(buft->context)->ggml_backend_qnn_buffer_free_buffer();
}
static ggml_backend_buffer_t ggml_backend_qnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return reinterpret_cast<qnn_wrapper*>(buft->context)->ggml_backend_qnn_buffer_type_alloc_buffer(size);
}

static size_t ggml_backend_qnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return reinterpret_cast<qnn_wrapper*>(buft->context)->ggml_backend_qnn_buffer_type_get_alignment();
}

static size_t ggml_backend_qnn_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return reinterpret_cast<qnn_wrapper*>(buft->context)->ggml_backend_qnn_buffer_type_get_max_size();
}

static bool ggml_backend_qnn_buffer_type_supports_backend(ggml_backend_buffer_type_t buft,
                                                          ggml_backend_t backend) {
    return reinterpret_cast<qnn_wrapper*>(buft->context)->ggml_backend_qnn_buffer_type_supports_backend(backend);
}

static bool ggml_backend_qnn_buffer_is_host(ggml_backend_buffer_type_t buft) {
    return reinterpret_cast<qnn_wrapper*>(buft->context)->ggml_backend_qnn_buffer_is_host();
}

//static void ggml_backend_qnn_buffer_free_buffer(ggml_backend_buffer_t buft);
//static void ggml_backend_qnn_buffer_init_tensor(ggml_backend_buffer_t buft, ggml_tensor * tensor);
//static void ggml_backend_qnn_buffer_set_tensor(ggml_backend_buffer_t buft, ggml_tensor * tensor, const void * data, size_t offset, size_t size);
//static void ggml_backend_qnn_buffer_get_tensor(ggml_backend_buffer_t buft, const ggml_tensor * tensor, void * data, size_t offset, size_t size);
//static bool ggml_backend_qnn_buffer_cpy_tensor(ggml_backend_buffer_t buft, const struct ggml_tensor * src, struct ggml_tensor * dst);
//static void ggml_backend_qnn_buffer_synchronize(ggml_backend_buffer_t buft);


ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(size_t device_index) {
    if (device_index >= GGML_QNN_MAX_DEVICES) {
        QNN_LOG_DEBUG("ggml_backend_qnn_buffer_type error: device_index:%d is out of range [0, %d]",
                       device_index, GGML_QNN_MAX_DEVICES - 1);
        return nullptr;
    }

    if (qnn_backends.contains(device_index)) {
        return &ggml_backend_buffer_types_qnn[device_index];
    }
    
    qnn_backends.emplace(device_index, std::make_shared<qnn_wrapper>("QnnHtp"));
    ggml_backend_buffer_types_qnn[device_index] = {
        {
        /* .get_name         = */ ggml_backend_qnn_buffer_get_name,
        /* .alloc_buffer     = */ ggml_backend_qnn_buffer_type_alloc_buffer,
        /* .get_alignment    = */ ggml_backend_qnn_buffer_type_get_alignment,
        /* .get_max_size     = */ ggml_backend_qnn_buffer_type_get_max_size,
        /* .get_alloc_size   = */ nullptr,
        /* .supports_backend = */ ggml_backend_qnn_buffer_type_supports_backend,
        /* .is_host          = */ ggml_backend_qnn_buffer_is_host
        },
        qnn_backends[device_index].get() // To be used in C-style callbacks
    };

    return nullptr;
}
