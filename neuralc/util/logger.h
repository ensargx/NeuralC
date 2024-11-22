#pragma once
#include <stdarg.h>

typedef enum
{
    DEBUG,
    WARN,
    ERROR
} LogLevel;

void log_message(LogLevel level, const char* fmt, va_list args);
void log_debug(const char* fmt, ...);
void log_error(const char* fmt, ...);
void log_warn(const char* fmt, ...);
