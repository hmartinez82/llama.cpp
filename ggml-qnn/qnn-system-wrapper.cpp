#include <dylib.hpp>

#include "qnn-system-wrapper.h"
#include "qnn-logging.h"

static const char* system_lib_name = "QnnSystem";

qnn_system_wrapper::qnn_system_wrapper(const std::filesystem::path & lib_folder) : _lib_folder(lib_folder) {

}

bool qnn_system_wrapper::is_loaded() const {
    return (_system_interface_instance != nullptr);
}

int qnn_system_wrapper::load() {
    int ret = load_interface(_lib_folder, system_lib_name);

    if (ret != 0) {
        QNN_LOG_ERROR("Failed to load QNN System");
        return ret;
    }

    system_context_create(&_qnn_system_handle);
    if (nullptr == _qnn_system_handle) {
        QNN_LOG_WARN("Cannot create QNN System context");
    } else {
        QNN_LOG_INFO("QNN System context created");
    }

    return 0;
}

void qnn_system_wrapper::unload() {
    if (nullptr != _qnn_system_handle) {
        auto result = system_context_free(_qnn_system_handle);
        if (result != QNN_SUCCESS) {
            QNN_LOG_WARN("Failed to free QNN system context\n");
        }
        _qnn_system_handle = nullptr;
    }
}

int qnn_system_wrapper::load_interface(const std::filesystem::path & lib_folder, const std::string & lib_name) 
{
    using namespace std;

    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    auto full_lib_path = lib_folder / lib_name;
    QNN_LOG_DEBUG("Backend library path:%s", full_lib_path.generic_string().c_str());

    // Load QNN backend library and QnnInterface_getProviders function
    pfn_QnnSystemInterface_getProviders* get_providers = nullptr;
    try {
        _lib = make_shared<dylib>(lib_folder, lib_name, true);
        get_providers = _lib->get_function<pfn_QnnSystemInterface_getProviders>("QnnSystemInterface_getProviders");
    } catch (const dylib::load_error & ex) {
        QNN_LOG_ERROR("Failed to load QNN System library %s. %s",  full_lib_path.generic_string().c_str(), ex.what());
        return 1;
    } catch (const dylib::symbol_error & ex) {
        QNN_LOG_ERROR("Failed to load symbol QnnSystemInterface_getProviders from %s. %s",  full_lib_path.generic_string().c_str(), ex.what());
        return 2;
    }

    uint32_t num_providers = 0;
    const QnnSystemInterface_t ** provider_list = nullptr;
    error = get_providers(&provider_list, &num_providers);
    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("Failed to get providers, error %d\n", QNN_GET_ERROR_CODE(error));
        return 3;
    }

    if (num_providers != s_required_num_providers) {
        QNN_LOG_ERROR("Number of interface providers is %d instead of required %d", num_providers, s_required_num_providers);
        return 4;
    }

    if (nullptr == provider_list) {
        QNN_LOG_ERROR("Cannot get system providers");
        return 5;
    }

    bool found_valid_system_interface = false;
    for (size_t idx = 0; idx < num_providers; idx++) {
        if (QNN_SYSTEM_API_VERSION_MAJOR ==
            provider_list[idx]->systemApiVersion.major &&
            QNN_SYSTEM_API_VERSION_MINOR <=
            provider_list[idx]->systemApiVersion.minor) {
            found_valid_system_interface = true;
            _system_interface_instance = provider_list[idx];
            break;
        }
    }
    if (!found_valid_system_interface) {
        QNN_LOG_ERROR("Unabled to find a valid QNN System interface");
        return 6;
    } else {
        QNN_LOG_INFO("Valid QNN System interface found");
    }

    return 0;
}