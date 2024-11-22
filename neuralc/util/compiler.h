#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define FORMAT_PRINTF(fmt_idx, args_idx) __attribute__((format(printf, fmt_idx, args_idx)))
#else
#define FORMAT_PRINTF(fmt_idx, args_idx)
#endif
