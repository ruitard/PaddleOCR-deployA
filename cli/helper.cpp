#include "helper.hpp"

#ifdef WIN32

#include <Windows.h>

static std::wstring u8str2wstring(const std::string &str) {
    int iTextLen = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), str.length(), nullptr, 0);
    std::wstring wstr(iTextLen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), str.length(), wstr.data(), iTextLen);
    return wstr;
}

static std::string wstring2acp(const std::wstring &wstr) {
    int iTextLen =
        WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), wstr.length(), nullptr, 0, nullptr, nullptr);
    std::string acpstr(iTextLen, '\0');
    WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, acpstr.data(), iTextLen, nullptr, nullptr);
    return acpstr;
}

static std::string u8str2acp(const std::string &str) { return wstring2acp(u8str2wstring(str)); }

std::string console_string(const std::string &str) { return u8str2acp(str); }

#endif

#ifdef linux

std::string console_string(const std::string &str) { return str; }

#endif