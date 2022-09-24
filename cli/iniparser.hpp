/**
 * @file iniparser.hpp
 * @brief based on https://github.com/mcmtroffaes/inipp
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
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <ranges>
#include <map>

namespace utils {

namespace string {

inline std::string ltrim(const std::string &str) {
    std::string s = str;
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](char ch) { return !std::isspace(ch); }));
    return s;
}

inline std::string rtrim(const std::string &str) {
    std::string s = str;
    s.erase(std::find_if(s.rbegin(), s.rend(), [](char ch) { return !std::isspace(ch); }).base(), s.end());
    return s;
}

inline std::string trim(const std::string &str) { return rtrim(ltrim(str)); }

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

inline bool extract(const std::string &value, std::string &dst) {
    dst = value;
    return true;
}

} // namespace string

namespace serde {

class Ini {
    using section_type = std::map<std::string, std::string>;
    using sections_type = std::map<std::string, section_type>;

    sections_type sections;

public:
    template <typename T> bool get(const std::string &sec, const std::string &key, T &dst) {
        try {
            dst = must_get<T>(sec, key);
        } catch (const std::out_of_range &) {
            return false;
        } catch (const std::runtime_error &) {
            return false;
        }
        return true;
    }

    template <typename T> T must_get(const std::string &sec, const std::string &key) {
        T result;
        auto &value = sections.at(sec).at(key);
        if (!string::extract(value, result)) {
            throw std::runtime_error("format error while unpacking value from ini");
        }
        return result;
    }

    void generate(std::ostream &os) const {
        for (auto const &sec : sections) {
            os << '[' << sec.first << ']' << std::endl;
            for (auto const &val : sec.second) {
                os << val.first << '=' << val.second << std::endl;
            }
            os << std::endl;
        }
    }

    bool parse(std::istream &is) {
        std::string line;
        bool first_section_seen = false;
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
                first_section_seen = true;
                current_section = line.substr(1, line.length() - 2);
            } else {
                if (!first_section_seen) {
                    // Key-value pairs appear before first section header.
                    return false;
                }
                std::vector<std::string> tokens = [&line]() {
                    std::vector<std::string> tokens;
                    for (auto token : line | std::views::split('=')) {
                        tokens.emplace_back(token.begin(), token.end());
                    }
                    return tokens;
                }();
                if (tokens.size() != 2) {
                    // There is no '=‘ character.
                    return false;
                }
                auto key = string::trim(tokens[0]);
                auto value = string::trim(tokens[1]);
                if (key.empty()) {
                    // Empty key is not allowed.
                    return false;
                }
                if (sections[current_section].count(key)) {
                    // Multiple definitions of the same key.
                    return false;
                }
                sections[current_section][key] = value;
            }
        }
        return true;
    }

    void clear() { sections.clear(); }
};

} // namespace serde

} // namespace utils