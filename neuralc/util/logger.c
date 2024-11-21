#include "logger.h"

#include <stdio.h>
#include <stdarg.h>
#include <time.h>

void log_message(const char* level, const char* fmt, va_list args)
{
    time_t now;
    struct tm *timeinfo;
    char timeStr[32];
    
    time(&now);
    timeinfo = localtime(&now);
    strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", timeinfo);

    printf("[%s] [%s] ", timeStr, level);

    vprintf(fmt, args);

    printf("\n");
}

void log_debug(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    log_message("DEBUG", fmt, args);
    va_end(args);
}

void log_error(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    log_message("ERROR", fmt, args);
    va_end(args);
}

void log_warn(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    log_message("WARN", fmt, args);
    va_end(args);
}

