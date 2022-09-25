/**
 * @file iniparser.hpp
 * @brief
 * 不支持行末注释，解析时，每一行有四种情况
 * 1、空行
 * 2、注释
 * 3、section名字
 * 4、键值对
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <map>
#include <string_view>
#include <utility>

namespace inipp {

namespace string {

inline std::string_view trim(std::string_view str) {
    return {std::find_if_not(str.begin(), str.end(), ::isspace),
            std::find_if_not(str.rbegin(), str.rend(), ::isspace).base()};
}

inline auto chop_one_shot(std::string_view str, char delim)
    -> std::pair<std::string_view, std::string_view> {
    auto pivot = std::find(str.begin(), str.end(), delim);
    if (pivot == std::end(str)) {
        throw std::out_of_range("missing delimiter");
    }
    return {{str.begin(), pivot}, {std::next(pivot), str.end()}};
}

template <typename T> inline bool extract(const std::string &value, T &dst) {
    char c;
    std::istringstream is{value};
    T result;
    if ((is >> std::boolalpha >> result) && !(is >> c)) {
        dst = result;
        return true;
    } else {
        return false;
    }
}

inline bool extract(std::string_view value, std::string &dst) {
    dst = value;
    return true;
}

} // namespace string

class Ini {
private:
    using section_type = std::map<std::string, std::string>;
    using sections_type = std::map<std::string, section_type>;

    sections_type sections;

public:
    template <typename T> T must_get(const std::string &sec, const std::string &key) {
        T result;
        auto &value = sections.at(sec).at(key);
        if (!string::extract(value, result)) {
            throw std::runtime_error("format error while unpacking value from ini");
        }
        return result;
    }

    bool parse(std::istream &is) {
        std::string line;
        std::string current_section;
        while (std::getline(is, line)) {
            line = string::trim(line);
            if (line.empty()) {
                continue;
            }
            if (line.starts_with('#') || line.starts_with(';')) {
                continue;
            }
            if (line.starts_with('[') && line.ends_with(']')) {
                current_section = line.substr(1, line.length() - 2);
            } else {
                const auto [key, value] = string::chop_one_shot(line, '=');
                if (key.empty() || sections[current_section].contains(std::string(key))) {
                    // Empty key and multiple definitions of the same key are not allowed.
                    return false;
                }
                sections[current_section][std::string(key)] = value;
            }
        }
        return true;
    }

    void clear() { sections.clear(); }
};

} // namespace inipp