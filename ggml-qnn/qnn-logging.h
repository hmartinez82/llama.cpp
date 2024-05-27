#ifndef QNN_LOGGING_H
#define QNN_LOGGING_H

#include <QnnLog.h>

//#include <ggml.h>
enum ggml_log_level {
    GGML_LOG_LEVEL_ERROR = 2,
    GGML_LOG_LEVEL_WARN  = 3,
    GGML_LOG_LEVEL_INFO  = 4,
    GGML_LOG_LEVEL_DEBUG = 5
};

#define GGML_QNN_LOGBUF_LEN                             4096

#define QNN_LOG_ERROR(...) ggml_qnn_log_internal(GGML_LOG_LEVEL_ERROR, __FILE__, __FUNCTION__, __LINE__, true, __VA_ARGS__)
#define QNN_LOG_WARN(...)  ggml_qnn_log_internal(GGML_LOG_LEVEL_WARN , __FILE__, __FUNCTION__, __LINE__, true, __VA_ARGS__)
#define QNN_LOG_INFO(...)  ggml_qnn_log_internal(GGML_LOG_LEVEL_DEBUG , __FILE__, __FUNCTION__, __LINE__, true, __VA_ARGS__)

#ifdef GGML_QNN_DEBUG
#define QNN_LOG_DEBUG(...) ggml_qnn_log_internal(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, true, __VA_ARGS__)
#else
#define QNN_LOG_DEBUG(...)
#endif

// GGML QNN internal logging function
void ggml_qnn_log_internal(ggml_log_level level, const char * file, const char * func, int line,
                           bool append_newline, const char * format, ...);

// Callback for QNN logging
void qnn_logcallback(const char * fmt,QnnLog_Level_t level, uint64_t timestamp, va_list argp);

#endif // QNN_LOGGING_H
