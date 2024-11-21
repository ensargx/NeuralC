#pragma once
#include <stdarg.h>

void log_message(const char* level, const char* fmt, va_list args);
void log_debug(const char* fmt, ...);
void log_error(const char* fmt, ...);
void log_warn(const char* fmt, ...);
