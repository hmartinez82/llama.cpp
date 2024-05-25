#ifndef GGML_QNN_BACKEND_H
#define GGML_QNN_BACKEND_H

#include <filesystem>
#include <mutex>
#include <string>
#include <memory>

#include <QnnInterface.h>
#include <Saver/QnnSaver.h>
#include <System/QnnSystemInterface.h>
#include <dylib.hpp>

#include "qnn-logging.h"
#include "qnn-wrapper.h"
#include "qnn-system-wrapper.h"

class ggml_qnn_backend
{
public:
    using _pfn_QnnSaver_initialize                          = decltype(QnnSaver_initialize);
    using _pfn_QnnInterface_getProviders                    = decltype(QnnInterface_getProviders);
    using _pfn_QnnSystemInterface_getProviders              = decltype(QnnSystemInterface_getProviders);
    using BackendIdType                                     = decltype(QnnInterface_t{}.backendId);

    explicit ggml_qnn_backend(const std::filesystem::path & lib_path, const std::string & backend_name);

    ~ggml_qnn_backend();

    int qnn_init(const QnnSaver_Config_t ** saver_config);

    int qnn_finalize();

    const QNN_INTERFACE_VER_TYPE &get_qnn_raw_interface() {
    if (!_qnn_wrapper.is_loaded()) {
        QNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
    }
    return _qnn_raw_interface;
    }

    const QNN_SYSTEM_INTERFACE_VER_TYPE &get_qnn_raw_system_interface() {
        if (!_qnn_system_wrapper.is_loaded()) {
            QNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_raw_system_interface;
    }

private:
    static constexpr const int s_required_num_providers = 1;

    static std::mutex _init_mutex;
    
    static std::unordered_map<BackendIdType, std::shared_ptr<dylib>> _loaded_lib_handle;
    
    static std::unordered_map<std::filesystem::path, BackendIdType> _lib_path_to_backend_id;
    
    static std::unordered_map<BackendIdType, const QnnInterface_t *> _loaded_backend;

    std::filesystem::path _lib_path;

    std::string _backend_name;

    BackendIdType _backend_id;

    qnn_wrapper _qnn_wrapper;

    qnn_system_wrapper _qnn_system_wrapper;

    QNN_INTERFACE_VER_TYPE _qnn_raw_interface;

    QNN_SYSTEM_INTERFACE_VER_TYPE _qnn_raw_system_interface;

    Qnn_LogHandle_t _qnn_log_handle = nullptr;

    Qnn_DeviceHandle_t _qnn_device_handle = nullptr;

    Qnn_BackendHandle_t _qnn_backend_handle = nullptr;

    Qnn_ContextHandle_t _qnn_context_handle = nullptr;

    Qnn_ProfileHandle_t _qnn_profile_handle = nullptr;

    int load_interface(std::filesystem::path & lib_path);

    int load_saver(std::filesystem::path & saverlib_path, const QnnSaver_Config_t ** saver_config);

    int unload_backend();

    int load_system();

    int unload_system();
};

#endif // GGML_QNN_BACKEND_H