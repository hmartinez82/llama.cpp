#include "ggml-qnn-backend.h"

#include <HTP/QnnHtpDevice.h>

std::mutex ggml_qnn_backend::_init_mutex;

std::unordered_map<ggml_qnn_backend::BackendIdType, std::shared_ptr<dylib>> ggml_qnn_backend::_loaded_lib_handle;

std::unordered_map<std::filesystem::path, ggml_qnn_backend::BackendIdType> ggml_qnn_backend::_lib_path_to_backend_id;

std::unordered_map<ggml_qnn_backend::BackendIdType, const QnnInterface_t *> ggml_qnn_backend::_loaded_backend;

ggml_qnn_backend::ggml_qnn_backend(const std::filesystem::path & lib_path, const std::string & backend_name) :
                                   _lib_path(lib_path),
                                   _backend_name(backend_name)
{

};

ggml_qnn_backend::~ggml_qnn_backend() {
}

int ggml_qnn_backend::qnn_init(const QnnSaver_Config_t ** saver_config) {
    BackendIdType backend_id = QNN_BACKEND_ID_NULL;
    QNN_LOG_DEBUG("Enter qnn_init()");

    const std::lock_guard<std::mutex> lock(_init_mutex);

    //if (0 != load_system()) {
    //    QNN_LOG_WARN("can not load QNN system lib, pls check why?\n");
    //    return 1;
    //} else {
    //    QNN_LOG_DEBUG("load QNN system lib successfully\n");
    //}

    std::filesystem::path backend_lib_path = _lib_path / _backend_name;

    if (0 == _lib_path_to_backend_id.count(backend_lib_path)) {
        int is_load_ok = load_interface(backend_lib_path);
        if (0 != is_load_ok) {
            QNN_LOG_WARN("Failed to load QNN backend");
            return 2;
        }
    }

    std::filesystem::path saver_lib_path = _lib_path / "QnnSaver";
    int is_load_ok = load_saver(saver_lib_path, saver_config);
    if (0 != is_load_ok) {
        QNN_LOG_WARN("Failed to load QNN backend");
        return 3;
    } else {
        QNN_LOG_DEBUG("QNN Saver loaded successfully");
    }
        

    backend_id = _lib_path_to_backend_id[backend_lib_path];
    if (0 == _loaded_backend.count(backend_id) ||
        0 == _loaded_lib_handle.count(backend_id)) {
        QNN_LOG_WARN("Library %s is loaded but loaded backend count=%zu, loaded lib_handle count=%zu",
              backend_lib_path.c_str(),
              _loaded_backend.count(backend_id),
              _loaded_lib_handle.count(backend_id));
        return 4;
    }

    _qnn_wrapper.set_qnn_interface(_loaded_backend[backend_id]);

    std::vector<const QnnBackend_Config_t *> temp_backend_config;
    _qnn_wrapper.qnn_backend_create(_qnn_log_handle, temp_backend_config.empty() ? nullptr
                                                                                   : temp_backend_config.data(),
                                      &_qnn_backend_handle);
    if (nullptr == _qnn_backend_handle) {
        QNN_LOG_WARN("Failed to initilize QNN backend");
        return 5;
    } else {
        QNN_LOG_DEBUG("QNN backend Initialization successful");
    }

    if (nullptr != _qnn_raw_interface.propertyHasCapability) {
        auto qnnStatus = _qnn_raw_interface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_NOT_SUPPORTED == qnnStatus) {
            QNN_LOG_WARN("Device QNN_PROPERTY_GROUP_DEVICE is not supported");
        }
        if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnStatus) {
            QNN_LOG_WARN("Device QNN_PROPERTY_GROUP_DEVICE is not known to backend");
        }
    }

    QnnHtpDevice_CustomConfig_t customConfig;
    customConfig.option    = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
    customConfig.arch.arch = QNN_HTP_DEVICE_ARCH_V68;
    customConfig.arch.deviceId = 0;  // Id of device to be used. If single device is used by default 0.
    QnnDevice_Config_t devConfig;
    devConfig.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    devConfig.customConfig = &customConfig;
    const QnnDevice_Config_t* pDeviceConfig[] = {&devConfig, NULL};

    auto qnnStatus = _qnn_raw_interface.deviceCreate(_qnn_log_handle, nullptr, &_qnn_device_handle);
    if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
        QNN_LOG_WARN("Failed to create QNN device");
    } else {
        QNN_LOG_DEBUG("Device created successfully");
    }

    //if (ggml_qnn_profile_level::profile_off != _profile_level) {
    //    QNN_LOG_INFO("profiling turned on; level = %d", _profile_level);
    //    if (ggml_qnn_profile_level::profile_basic == _profile_level) {
    //        QNN_LOG_INFO("basic profiling requested. creating Qnn Profile object\n");
    //        if (QNN_PROFILE_NO_ERROR != _qnn_raw_interface.profileCreate(
    //                _qnn_backend_handle, QNN_PROFILE_LEVEL_BASIC, &_qnn_profile_handle)) {
    //            QNN_LOG_WARN("unable to create profile handle in the backend\n");
    //            return 7;
    //        } else {
    //            QNN_LOG_DEBUG("initialize qnn profile successfully\n");
    //        }
    //    } else if (ggml_qnn_profile_level::profile_detail == _profile_level) {
    //        QNN_LOG_INFO("detailed profiling requested. Creating Qnn Profile object\n");
    //        if (QNN_PROFILE_NO_ERROR != _qnn_raw_interface.profileCreate(
    //                _qnn_backend_handle, QNN_PROFILE_LEVEL_DETAILED, &_qnn_profile_handle)) {
    //            QNN_LOG_WARN("unable to create profile handle in the backend\n");
    //            return 7;
    //        } else {
    //            QNN_LOG_DEBUG("initialize qnn profile successfully\n");
    //        }
    //    }
    //}


    //_rpc_lib_handle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    //if (nullptr == _rpc_lib_handle) {
    //    QNN_LOG_WARN("failed to load qualcomm's rpc lib, error:%s\n", dlerror());
    //    return 9;
    //} else {
    //    QNN_LOG_DEBUG("load rpcmem lib successfully\n");
    //    set_rpcmem_initialized(true);
    //}
    //_pfn_rpc_mem_init   = reinterpret_cast<pfn_rpc_mem_init>(dlsym(_rpc_lib_handle, "rpcmem_init"));
    //_pfn_rpc_mem_deinit = reinterpret_cast<pfn_rpc_mem_deinit>(dlsym(_rpc_lib_handle, "rpcmem_deinit"));
    //_pfn_rpc_mem_alloc  = reinterpret_cast<pfn_rpc_mem_alloc>(dlsym(_rpc_lib_handle,"rpcmem_alloc"));
    //_pfn_rpc_mem_free   = reinterpret_cast<pfn_rpc_mem_free>(dlsym(_rpc_lib_handle, "rpcmem_free"));
    //_pfn_rpc_mem_to_fd  = reinterpret_cast<pfn_rpc_mem_to_fd>(dlsym(_rpc_lib_handle,"rpcmem_to_fd"));
    //if (nullptr == _pfn_rpc_mem_alloc || nullptr == _pfn_rpc_mem_free
    //    || nullptr == _pfn_rpc_mem_to_fd) {
    //    QNN_LOG_WARN("unable to access symbols in QNN RPC lib. dlerror(): %s", dlerror());
    //    dlclose(_rpc_lib_handle);
    //    return 10;
    //}

    //if (nullptr != _pfn_rpc_mem_init) // make Qualcomm's SoC based low-end phone happy
    //    _pfn_rpc_mem_init();

    std::vector<const QnnContext_Config_t *> temp_context_config;
    _qnn_wrapper.qnn_context_create(_qnn_backend_handle, _qnn_device_handle,
                                      temp_context_config.empty() ? nullptr
                                                                  : temp_context_config.data(),
                                      &_qnn_context_handle);
    if (nullptr == _qnn_context_handle) {
        QNN_LOG_WARN("Failed to create QNN context");
        return 8;
    } else {
        QNN_LOG_DEBUG("QNN context created successfully");
    }

    QNN_LOG_DEBUG("Leave qnn_init()");

    return 0;
}

int ggml_qnn_backend::qnn_finalize() {
    QNN_LOG_DEBUG("Enter qnn_finalize()");

    int ret_status = 0;
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    //if (nullptr != _pfn_rpc_mem_deinit) // make Qualcomm's SoC based low-end phone happy
    //    _pfn_rpc_mem_deinit();

    //if (dlclose(_rpc_lib_handle) != 0) {
    //    QNN_LOG_WARN("failed to unload qualcomm's rpc lib, error:%s\n", dlerror());
    //} else {
    //    QNN_LOG_DEBUG("succeed to close rpcmem lib\n");
    //}

    if (nullptr != _qnn_context_handle) {
        error = _qnn_wrapper.qnn_context_free(_qnn_context_handle, _qnn_profile_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("Failed to free QNN context handle: ID %u, error %d",
                  _qnn_wrapper.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_context_handle = nullptr;
    }

    if (nullptr != _qnn_profile_handle) {
        error = _qnn_wrapper.qnn_profile_free(_qnn_profile_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("Failed to free QNN profiling: ID %u, error %d\n",
                  _qnn_wrapper.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_profile_handle = nullptr;
    }

    if (nullptr != _qnn_device_handle) {
        error = _qnn_wrapper.qnn_device_free(_qnn_device_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("Failed to free QNN device: ID %u, error %d\n",
                  _qnn_wrapper.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_device_handle = nullptr;
    }

    if (nullptr != _qnn_backend_handle) {
        error = _qnn_wrapper.qnn_backend_free(_qnn_backend_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("Failed to free QNN backend: ID %u, error %d\n",
                  _qnn_wrapper.get_backend_id(), QNN_GET_ERROR_CODE(error));
        }
        _qnn_backend_handle = nullptr;

    }

    if (nullptr != _qnn_log_handle) {
        error = _qnn_wrapper.qnn_log_free(_qnn_log_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("Failed to free QNN logging: ID %u, error %d\n",
                  _qnn_wrapper.get_backend_id(), QNN_GET_ERROR_CODE(error));
        }
        _qnn_log_handle = nullptr;
    }

    //unload_backend();

    //unload_system();

    QNN_LOG_DEBUG("Leave qnn_finalize()");

    return ret_status;
}

int ggml_qnn_backend::load_interface(std::filesystem::path & lib_path) {
    using namespace std;

    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    QNN_LOG_DEBUG("Backend library path:%s", lib_path.generic_string().c_str());

    // load QNN backend library and get_provider function
    shared_ptr<dylib> lib;
    _pfn_QnnInterface_getProviders* get_providers = nullptr;
    try {
        lib = make_shared<dylib>(lib_path);
        get_providers = lib->get_function<remove_pointer_t<_pfn_QnnInterface_getProviders>>("QnnInterface_getProviders");
    } catch (const dylib::load_error & ex) {
        QNN_LOG_ERROR("Failed to load QNN backend library %s. %s",  lib_path.generic_string().c_str(), ex.what());
        return 1;
    } catch (const dylib::symbol_error & ex) {
        QNN_LOG_ERROR("Failed to load symbol QnnInterface_getProviders from %s. %s",  lib_path.generic_string().c_str(), ex.what());
        return 2;
    }

    // get QnnInterface Providers
    std::uint32_t num_providers = 0;
    const QnnInterface_t **provider_list = nullptr;
    error = get_providers(&provider_list, &num_providers);
    if (error != QNN_SUCCESS) {
        QNN_LOG_WARN("Failed to get providers, error %d", QNN_GET_ERROR_CODE(error));
        return 3;
    }
    QNN_LOG_DEBUG("Number of providers: %d", num_providers);
    if (num_providers != s_required_num_providers) {
        QNN_LOG_WARN("Number of interface providers is %d instead of required %d", num_providers, s_required_num_providers);
        return 4;
    }

    if (nullptr == provider_list) {
        QNN_LOG_WARN("Failed to get QNN interface providers");
        return 5;
    }
    bool found_valid_interface = false;
    QNN_INTERFACE_VER_TYPE qnn_interface;
    for (size_t idx = 0; idx < num_providers; idx++) {
        if (QNN_API_VERSION_MAJOR == provider_list[idx]->apiVersion.coreApiVersion.major &&
            QNN_API_VERSION_MINOR <= provider_list[idx]->apiVersion.coreApiVersion.minor) {
            found_valid_interface = true;
            qnn_interface = provider_list[idx]->QNN_INTERFACE_VER_NAME;
            break;
        }
    }

    if (!found_valid_interface) {
        QNN_LOG_WARN("Unable to find a valid QNN interface");
        return 6;
    } else {
        QNN_LOG_INFO("Found a valid QNN interface");
    }

    _qnn_raw_interface = qnn_interface;

    BackendIdType backend_id = provider_list[0]->backendId;
    _lib_path_to_backend_id[lib_path] = backend_id;
    if (_loaded_backend.count(backend_id) > 0) {
        QNN_LOG_WARN("Backend library path %s is loaded, but backend %d already exists",
              lib_path.generic_string().c_str(), backend_id);
    }
    _loaded_backend[backend_id] = provider_list[0];
    if (_loaded_lib_handle.count(backend_id) > 0) {
        QNN_LOG_WARN("Closing backend id %d loaded library", backend_id);
        _loaded_lib_handle.erase(backend_id);
    }
    _loaded_lib_handle[backend_id] = lib;
    _backend_id = backend_id;

    // Setup QNN logging
    QnnLog_Level_t qnn_log_level = QNN_LOG_LEVEL_WARN;
#if defined(GGML_QNN_SDK_DEBUG)
    qnn_log_level = QNN_LOG_LEVEL_DEBUG;
#elif defined(GGML_QNN_DEBUG)
    qnn_log_level = QNN_LOG_LEVEL_INFO;
#endif

    _qnn_raw_interface.logCreate(qnn_logcallback, qnn_log_level, &_qnn_log_handle);

    if (nullptr == _qnn_log_handle) {
        QNN_LOG_WARN("Failed to initialize QNN logging");
        return 7;
    } else {
        QNN_LOG_DEBUG("QNN log initialized successfully");
    }

    return 0;
}

int ggml_qnn_backend::load_saver(std::filesystem::path & saverlib_path, const QnnSaver_Config_t ** saver_config) {
    
    // load QNN Saver library
    try {
        dylib saverLib(saverlib_path);
        auto saver_initialize = saverLib.get_function<_pfn_QnnSaver_initialize>("QnnSaver_initialize");
        auto error = saver_initialize(saver_config);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("Failed to QnnSaver_initialize. Error %d", QNN_GET_ERROR_CODE(error));
            return 8;
        }
    } catch (const dylib::load_error & ex) {
        QNN_LOG_ERROR("Failed to load QNN Saver library %s. %s",  saverlib_path.generic_string().c_str(), ex.what());
        return 9;
    } catch (const dylib::symbol_error & ex) {
        QNN_LOG_ERROR("Failed to load symbol QnnSaver_initialize from %s. %s",  saverlib_path.generic_string().c_str(), ex.what());
        return 10;
    }

    return 0;
}
