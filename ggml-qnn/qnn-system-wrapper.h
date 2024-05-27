#ifndef QNN_SYSTEM_WRAPPER_H
#define QNN_SYSTEM_WRAPPER_H

#include <dylib.hpp>
#include <memory>
#include <filesystem>
#include <functional>
#include <string>

#include <System/QnnSystemInterface.h>

#define DEFINE_SHIM_FUNCTION_SYS_INTERFACE(F, pointer_name)                  \
  template <typename... Args>                                                \
  inline auto F(Args... args) const {                                  \
    return (_system_interface_instance->QNN_SYSTEM_INTERFACE_VER_NAME.pointer_name)( \
        std::forward<Args>(args)...);                                        \
  }

/**
*  Wrapper class for Qualcomm QNN(AI Engine Direct) SDK System interface
*/
class qnn_system_wrapper
{
public:

    qnn_system_wrapper(const std::filesystem::path & lib_folder);

    ~qnn_system_wrapper() {
        unload();
    }

    // QnnSystem
    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_create, systemContextCreate);

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_get_binary_info, systemContextGetBinaryInfo);

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_free, systemContextFree);

    bool is_loaded() const;

    int load();

    void unload();

private:
    using pfn_QnnSystemInterface_getProviders              = decltype(QnnSystemInterface_getProviders);

    static constexpr const int s_required_num_providers = 1;

    const QnnSystemInterface_t *_system_interface_instance = nullptr;

    QnnSystemContext_Handle_t _qnn_system_handle = nullptr;

    std::filesystem::path _lib_folder;

    std::shared_ptr<dylib> _lib;

    int load_interface(const std::filesystem::path & lib_folder, const std::string & lib_name);
};

#endif // QNN_SYSTEM_WRAPPER_H
