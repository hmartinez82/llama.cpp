#ifndef QNN_SYSTEM_WRAPPER_H
#define QNN_SYSTEM_WRAPPER_H

#include "qnn-interface-shim.h"
#include <System/QnnSystemInterface.h>

/**
*   *   Wrapper class of Qualcomm QNN(Qualcomm AI Engine Direct) SDK System interface
*/
class qnn_system_wrapper
{
public:

    // QnnSystem
    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_create, systemContextCreate);

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_get_binary_info, systemContextGetBinaryInfo);

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_free, systemContextFree);

    inline void set_qnn_system_interface(const QnnSystemInterface_t * qnn_sys_interface) {
        _qnn_sys_interface = qnn_sys_interface;
    }

    inline bool is_loaded() const {
        return (_qnn_sys_interface != nullptr);
    }

private:
    
    const QnnSystemInterface_t *_qnn_sys_interface = nullptr;
};

#endif // QNN_SYSTEM_WRAPPER_H