#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <QnnLog.h>

#include "qnn-logging.h"

void qnn_logcallback(const char * fmt, QnnLog_Level_t level, uint64_t timestamp, va_list argp) {

    char logbuf[GGML_QNN_LOGBUF_LEN];

    ggml_log_level ggml_level = GGML_LOG_LEVEL_DEBUG;
    switch (level) {
        case QNN_LOG_LEVEL_ERROR:
            ggml_level = GGML_LOG_LEVEL_ERROR;
            break;
        case QNN_LOG_LEVEL_WARN:
            ggml_level = GGML_LOG_LEVEL_WARN;
            break;
        case QNN_LOG_LEVEL_INFO:
            ggml_level = GGML_LOG_LEVEL_INFO;
            break;
        case QNN_LOG_LEVEL_DEBUG:
            ggml_level = GGML_LOG_LEVEL_DEBUG;
            break;
        case QNN_LOG_LEVEL_VERBOSE:
            ggml_level = GGML_LOG_LEVEL_DEBUG;
            break;
        case QNN_LOG_LEVEL_MAX:
            ggml_level = GGML_LOG_LEVEL_DEBUG;
            break;
    }

    double ms = (double) timestamp / 1000000.0;

    memset(logbuf, 0, GGML_QNN_LOGBUF_LEN);
    vsnprintf(reinterpret_cast<char *const>(logbuf), GGML_QNN_LOGBUF_LEN, fmt, argp);
    ggml_qnn_log_internal(ggml_level, __FILE__, __FUNCTION__, __LINE__, true, "%8.2fms %s", ms, logbuf);

}

void ggml_qnn_log_internal(ggml_log_level level, const char * file, const char * func, int line,
                           bool append_newline, const char * format, ...) {
    static std::mutex ggml_qnn_log_internal_mutex;
    static char s_ggml_qnn_log_internal_buf[GGML_QNN_LOGBUF_LEN];

    {
        std::lock_guard<std::mutex> lock(ggml_qnn_log_internal_mutex);
        va_list args;
        va_start(args, format);
        int len_prefix = snprintf(s_ggml_qnn_log_internal_buf, GGML_QNN_LOGBUF_LEN, "[%s, %d]: ", func, line);
        int len = vsnprintf(s_ggml_qnn_log_internal_buf + len_prefix, GGML_QNN_LOGBUF_LEN - len_prefix, format, args);
        if (len < (GGML_QNN_LOGBUF_LEN - len_prefix)) {
#if (defined __ANDROID__) || (defined ANDROID)
            //for Android APP
            __android_log_print(level, "ggml-qnn", append_newline ? "%s\n" : "%s", s_ggml_qnn_log_internal_buf);
            //for Android terminal
            printf(append_newline ? "%s\n" : "%s", s_ggml_qnn_log_internal_buf);
#else
            printf(append_newline ? "%s\n" : "%s", s_ggml_qnn_log_internal_buf);
#endif
        }
        va_end(args);
    }
}