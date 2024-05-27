#ifndef GGML_QNN_BACKEND_H
#define GGML_QNN_BACKEND_H

#include <initializer_list>
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
    using BackendIdType                                     = decltype(QnnInterface_t{}.backendId);

    explicit ggml_qnn_backend(const std::filesystem::path & lib_path);

    ~ggml_qnn_backend();

    int init(std::initializer_list<BackendIdType> backends_to_initialize,
             const QnnSaver_Config_t ** saver_config = nullptr);

    int finalize();


private:
    static std::mutex _init_mutex;
    
    std::filesystem::path _lib_path;

    std::shared_ptr<qnn_system_wrapper> _qnn_system_wrapper;

    std::unordered_map<BackendIdType, std::shared_ptr<qnn_wrapper>> _qnn_wrappers;

    int load_saver(const std::filesystem::path & lib_folder, const std::string & lib_name, const QnnSaver_Config_t ** saver_config);
};

#endif // GGML_QNN_BACKEND_H
