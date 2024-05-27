#include "qnn-wrapper.h"
#include "qnn-logging.h"

qnn_wrapper::qnn_wrapper(const std::filesystem::path & lib_folder, const std::string & backend_name) :
    _lib_folder(lib_folder), _backend_name(backend_name) {

}

BackendIdType qnn_wrapper::get_backend_id() const {
    return _interface_instance->backendId;
}

bool qnn_wrapper::is_loaded() const {
    return (_interface_instance != nullptr);
}

int qnn_wrapper::load(const QnnDevice_Config_t* device_config) {
    int ret = load_interface(_lib_folder, _backend_name);

    if (ret != 0) {
        QNN_LOG_ERROR("Failed to load QNN backend");
        return ret;
    }

    // Setup QNN logging for the backend
    QnnLog_Level_t qnn_log_level = QNN_LOG_LEVEL_WARN;
#if defined(GGML_QNN_SDK_DEBUG)
    qnn_log_level = QNN_LOG_LEVEL_DEBUG;
#elif defined(GGML_QNN_DEBUG)
    qnn_log_level = QNN_LOG_LEVEL_INFO;
#endif

    log_create(qnn_logcallback, qnn_log_level, &_qnn_log_handle);

    if (nullptr == _qnn_log_handle) {
        QNN_LOG_WARN("Failed to initialize QNN logging");
    } else {
        QNN_LOG_DEBUG("QNN log initialized successfully");
    }

    // This is the minimum required set of properties for the backend to be considered
    auto qnnStatus = property_has_capability(QNN_PROPERTY_BACKEND_SUPPORT_COMPOSITION);
    if (QNN_PROPERTY_NOT_SUPPORTED == qnnStatus) {
        QNN_LOG_ERROR("Backend does not support QNN_PROPERTY_BACKEND_SUPPORT_COMPOSITION");
        return 7;
    }
    if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnStatus) {
        QNN_LOG_WARN("Property QNN_PROPERTY_BACKEND_SUPPORT_COMPOSITION is not known to backend");
    }

    std::vector<const QnnBackend_Config_t *> temp_backend_config;
    backend_create(_qnn_log_handle, temp_backend_config.empty() ? nullptr : temp_backend_config.data(),
                   &_qnn_backend_handle);

    if (nullptr == _qnn_backend_handle) {
        QNN_LOG_ERROR("Failed to initilize QNN backend");
        return 8;
    } else {
        QNN_LOG_DEBUG("QNN backend Initialization successful");
    }

    //QnnHtpDevice_CustomConfig_t customConfig;
    //customConfig.option    = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
    //customConfig.arch.arch = QNN_HTP_DEVICE_ARCH_V68;
    //customConfig.arch.deviceId = 0;  // Id of device to be used. If single device is used by default 0.
    //QnnDevice_Config_t devConfig;
    //devConfig.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    //devConfig.customConfig = &customConfig;

    // Only try to create device if device_config is not nullptr
    if (device_config != nullptr) {
        const QnnDevice_Config_t* pDeviceConfig[] = {device_config, NULL};

        qnnStatus = device_create(_qnn_log_handle, pDeviceConfig, &_qnn_device_handle);
        if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
            QNN_LOG_WARN("Failed to create QNN device");
        } else {
            QNN_LOG_DEBUG("Device created successfully");
        }
    }

    std::vector<const QnnContext_Config_t *> temp_context_config;
    context_create(_qnn_backend_handle, _qnn_device_handle,
                   temp_context_config.empty() ? nullptr
                                               : temp_context_config.data(),
                   &_qnn_context_handle);
    if (nullptr == _qnn_context_handle) {
        QNN_LOG_WARN("Failed to create QNN context");
        return 9;
    } else {
        QNN_LOG_DEBUG("QNN context created successfully");
    }

    return 0;
}


void qnn_wrapper::unload() {

    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    if (nullptr != _qnn_context_handle) {
        error = context_free(_qnn_context_handle, _qnn_profile_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("Failed to free QNN context handle: ID %u, error %d",
                  get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_context_handle = nullptr;
    }

    if (nullptr != _qnn_profile_handle) {
        error = profile_free(_qnn_profile_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("Failed to free QNN profiling: ID %u, error %d\n",
                  get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_profile_handle = nullptr;
    }

    if (nullptr != _qnn_device_handle) {
        error = device_free(_qnn_device_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("Failed to free QNN device: ID %u, error %d\n",
                  get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_device_handle = nullptr;
    }

    if (nullptr != _qnn_backend_handle) {
        error = backend_free(_qnn_backend_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("Failed to free QNN backend: ID %u, error %d\n",
                  get_backend_id(), QNN_GET_ERROR_CODE(error));
        }
        _qnn_backend_handle = nullptr;

    }

    if (nullptr != _qnn_log_handle) {
        error = log_free(_qnn_log_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("Failed to free QNN logging: ID %u, error %d\n",
                  get_backend_id(), QNN_GET_ERROR_CODE(error));
        }
        _qnn_log_handle = nullptr;
    }
}


int qnn_wrapper::load_interface(const std::filesystem::path & lib_folder, const std::string & lib_name) {
    using namespace std;

    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    auto full_lib_path = lib_folder / lib_name;
    QNN_LOG_DEBUG("Backend library path:%s", full_lib_path.generic_string().c_str());

    // Load QNN backend library and QnnInterface_getProviders function
    pfn_QnnInterface_getProviders* get_providers = nullptr;
    try {
        _lib = make_shared<dylib>(lib_folder, lib_name, true);
        get_providers = _lib->get_function<remove_pointer_t<pfn_QnnInterface_getProviders>>("QnnInterface_getProviders");
    } catch (const dylib::load_error & ex) {

        QNN_LOG_ERROR("Failed to load QNN backend library %s. %s",  full_lib_path.generic_string().c_str(), ex.what());
        return 1;
    } catch (const dylib::symbol_error & ex) {
        QNN_LOG_ERROR("Failed to load symbol QnnInterface_getProviders from %s. %s",  full_lib_path.generic_string().c_str(), ex.what());
        return 2;
    }

    // get QnnInterface Providers
    std::uint32_t num_providers = 0;
    const QnnInterface_t **provider_list = nullptr;
    error = get_providers(&provider_list, &num_providers);
    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("Failed to get providers, error %d", QNN_GET_ERROR_CODE(error));
        return 3;
    }
    QNN_LOG_DEBUG("Number of providers: %d", num_providers);
    if (num_providers != s_required_num_providers) {
        QNN_LOG_ERROR("Number of interface providers is %d instead of required %d", num_providers, s_required_num_providers);
        return 4;
    }

    if (nullptr == provider_list) {
        QNN_LOG_ERROR("Failed to get QNN interface providers");
        return 5;
    }
    bool found_valid_interface = false;
    for (size_t idx = 0; idx < num_providers; idx++) {
        if (QNN_API_VERSION_MAJOR == provider_list[idx]->apiVersion.coreApiVersion.major &&
            QNN_API_VERSION_MINOR <= provider_list[idx]->apiVersion.coreApiVersion.minor) {
            found_valid_interface = true;
            _interface_instance = provider_list[idx]; //Bind to wrapping pointer
            break;
        }
    }

    if (!found_valid_interface) {
        QNN_LOG_ERROR("Unable to find a valid QNN interface");
        return 6;
    } else {
        QNN_LOG_INFO("Found a valid QNN interface");
    }

    return 0;
}
