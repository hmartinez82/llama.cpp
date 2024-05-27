#ifndef QNN_WRAPPER_H
#define QNN_WRAPPER_H

#include <dylib.hpp>
#include <memory>
#include <filesystem>
#include <functional>
#include <string>

#include <QnnInterface.h>
#include <Saver/QnnSaver.h>

#define DEFINE_SHIM_FUNCTION_INTERFACE(F, pointer_name)           \
  template <typename... Args>                                     \
  inline auto F(Args... args) const {                       \
    return (_interface_instance->QNN_INTERFACE_VER_NAME.pointer_name)( \
        std::forward<Args>(args)...);                             \
  }

using BackendIdType                                     = decltype(QnnInterface_t{}.backendId);


/**
*   Wrapper class for Qualcomm QNN(AI Engine Direct) SDK interface
*/
class qnn_wrapper {

public:
    qnn_wrapper(const std::filesystem::path & lib_folder, const std::string & backend_name);

    ~qnn_wrapper() {
        unload();
    }

    // QnnBackend
    DEFINE_SHIM_FUNCTION_INTERFACE(backend_create, backendCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_free, backendFree);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_register_op_package, backendRegisterOpPackage);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_validate_op_config, backendValidateOpConfig);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_get_api_version, backendGetApiVersion);

    // QnnDevice
    DEFINE_SHIM_FUNCTION_INTERFACE(device_create, deviceCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_free, deviceFree);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_infrastructure, deviceGetInfrastructure);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_platform_info, deviceGetPlatformInfo);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_info, deviceGetInfo);

    // QnnContext
    DEFINE_SHIM_FUNCTION_INTERFACE(context_create, contextCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary_size, contextGetBinarySize);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary, contextGetBinary);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_create_from_binary, contextCreateFromBinary);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_free, contextFree);

    // QnnGraph
    DEFINE_SHIM_FUNCTION_INTERFACE(graph_create, graphCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_add_node, graphAddNode);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_finalize, graphFinalize);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_execute, graphExecute);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_retrieve, graphRetrieve);

    // QnnLog
    DEFINE_SHIM_FUNCTION_INTERFACE(log_create, logCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(log_free, logFree);

    DEFINE_SHIM_FUNCTION_INTERFACE(log_set_log_level, logSetLogLevel);

    // QnnProfile
    DEFINE_SHIM_FUNCTION_INTERFACE(profile_create, profileCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_events, profileGetEvents);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_sub_events, profileGetSubEvents);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_event_data, profileGetEventData);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_free, profileFree);

    // QnnMem
    DEFINE_SHIM_FUNCTION_INTERFACE(mem_register, memRegister);

    DEFINE_SHIM_FUNCTION_INTERFACE(mem_de_register, memDeRegister);

    // QnnProperty
    DEFINE_SHIM_FUNCTION_INTERFACE(property_has_capability, propertyHasCapability);

    // QnnTensor
    DEFINE_SHIM_FUNCTION_INTERFACE(tensor_create_context_tensor, tensorCreateContextTensor);

    DEFINE_SHIM_FUNCTION_INTERFACE(tensor_create_graph_tensor, tensorCreateGraphTensor);

    BackendIdType get_backend_id() const;

    bool is_loaded() const;

    int load(const QnnDevice_Config_t* device_config);

    void unload();

private:
    using pfn_QnnInterface_getProviders                    = decltype(QnnInterface_getProviders);

    using pfn_QnnSaver_initialize                          = decltype(QnnSaver_initialize);

    static constexpr const int s_required_num_providers = 1;

    std::shared_ptr<dylib> _lib;

    const QnnInterface_t* _interface_instance = nullptr;

    Qnn_LogHandle_t _qnn_log_handle = nullptr;

    Qnn_BackendHandle_t _qnn_backend_handle = nullptr;

    Qnn_DeviceHandle_t _qnn_device_handle = nullptr;

    Qnn_ContextHandle_t _qnn_context_handle = nullptr;

    Qnn_ProfileHandle_t _qnn_profile_handle = nullptr;

    const std::filesystem::path _lib_folder;

    const std::string _backend_name;

    int load_interface(const std::filesystem::path & lib_folder, const std::string & lib_name);
};

#endif // QNN_WRAPPER_H
