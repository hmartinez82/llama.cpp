#include "ggml-qnn-backend.h"

#include <CPU/QnnCpuCommon.h>
#include <HTP/QnnHtpCommon.h>

std::mutex ggml_qnn_backend::_init_mutex;

ggml_qnn_backend::ggml_qnn_backend(const std::filesystem::path & lib_path) : _lib_path(lib_path)
{

};

ggml_qnn_backend::~ggml_qnn_backend() {
    finalize();
}

int ggml_qnn_backend::init(std::initializer_list<BackendIdType> backends_to_initialize, const QnnSaver_Config_t ** saver_config) {
    using namespace std;

    const lock_guard<mutex> lock(_init_mutex);

    QNN_LOG_DEBUG("Enter init()");

    // Initialize QNN backends
    for (auto backend_id : backends_to_initialize) {

        const char* backend_name = nullptr;
        if (backend_id == QNN_BACKEND_ID_CPU) {
            backend_name = "QnnCpu";
        } else if (backend_id == QNN_BACKEND_ID_HTP) {
            backend_name = "QnnHtp";
        } else {
            QNN_LOG_ERROR("Unsupported backend id %d", backend_id);
        }

        std::shared_ptr<qnn_wrapper>wrapper = make_shared<qnn_wrapper>(backend_name);
        int load_error = wrapper->load(_lib_path, nullptr);
        if(0 != load_error)
        {
            QNN_LOG_WARN("Failed to load %s backend", backend_name);
            return load_error;
        }

        _qnn_wrappers[backend_id] = wrapper;
    }

    // Initialize system wrapper
    _qnn_system_wrapper = make_shared<qnn_system_wrapper>(_lib_path);
    int ret = _qnn_system_wrapper->load();
    if (0 != ret) {
        QNN_LOG_WARN("Failed to load QNN System");
    } else {
        QNN_LOG_DEBUG("QNN System loaded successfully");
    }


    ret = load_saver(_lib_path, "QnnSaver", saver_config);
    if (0 != ret) {
        QNN_LOG_WARN("Failed to load QNN Saver");
    } else {
        QNN_LOG_DEBUG("QNN Saver loaded successfully");
    }

    QNN_LOG_DEBUG("Leave init()");

    return 0;
}

int ggml_qnn_backend::finalize() {
    QNN_LOG_DEBUG("Enter finalize()");

    _qnn_wrappers.clear();

    QNN_LOG_DEBUG("Leave finalize()");

    return 0;
}

int ggml_qnn_backend::load_saver(const std::filesystem::path & lib_folder, const std::string & lib_name, const QnnSaver_Config_t ** saver_config) {
    // Load QNN Saver library
    try {
        dylib saverLib(lib_folder, lib_name, true);
        auto saver_initialize = saverLib.get_function<_pfn_QnnSaver_initialize>("QnnSaver_initialize");
        auto error = saver_initialize(saver_config);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("Failed to QnnSaver_initialize. Error %d", QNN_GET_ERROR_CODE(error));
            return 8;
        }
    } catch (const dylib::load_error & ex) {
        auto full_lib_path = lib_folder / lib_name;
        QNN_LOG_ERROR("Failed to load QNN Saver library %s. %s",  full_lib_path.generic_string().c_str(), ex.what());
        return 9;
    } catch (const dylib::symbol_error & ex) {
        auto full_lib_path = lib_folder / lib_name;
        QNN_LOG_ERROR("Failed to load symbol QnnSaver_initialize from %s. %s",  full_lib_path.generic_string().c_str(), ex.what());
        return 10;
    }

    return 0;
}
