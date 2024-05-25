#ifndef QNN_INTERFACE_SHIM_H
#define QNN_INTERFACE_SHIM_H

#include <functional>

#define DEFINE_SHIM_FUNCTION_INTERFACE(F, pointer_name)           \
  template <typename... Args>                                     \
  inline auto qnn_##F(Args... args) const {                       \
    return (_qnn_interface->QNN_INTERFACE_VER_NAME.pointer_name)( \
        std::forward<Args>(args)...);                             \
  }


#define DEFINE_SHIM_FUNCTION_SYS_INTERFACE(F, pointer_name)                  \
  template <typename... Args>                                                \
  inline auto qnn_##F(Args... args) const {                                  \
    return (_qnn_sys_interface->QNN_SYSTEM_INTERFACE_VER_NAME.pointer_name)( \
        std::forward<Args>(args)...);                                        \
  }

#endif QNN_INTERFACE_SHIM_H