#pragma once

#include "util/compiler.h"
#include <stdarg.h>

typedef enum
{
    DEBUG,
    WARN,
    ERROR
} LogLevel;

void log_message(LogLevel level, const char* fmt, va_list args) FORMAT_PRINTF(2, 0);
void log_debug(const char* fmt, ...) FORMAT_PRINTF(1, 2);
void log_warn(const char* fmt, ...) FORMAT_PRINTF(1, 2);
void log_error(const char* fmt, ...) FORMAT_PRINTF(1, 2);

