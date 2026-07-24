#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

enum class MACEStreamedEdgesMode {
    legacy,
    r1,
    all,
};

inline MACEStreamedEdgesMode parse_mace_streamed_edges_mode(std::string_view mode)
{
    if (mode == "legacy")
        return MACEStreamedEdgesMode::legacy;
    if (mode == "r1")
        return MACEStreamedEdgesMode::r1;
    if (mode == "all")
        return MACEStreamedEdgesMode::all;
    throw std::invalid_argument(
        "streamed_edges must be one of 'legacy', 'r1', or 'all'.");
}

inline std::string mace_streamed_edges_mode_name(MACEStreamedEdgesMode mode)
{
    switch (mode) {
    case MACEStreamedEdgesMode::legacy:
        return "legacy";
    case MACEStreamedEdgesMode::r1:
        return "r1";
    case MACEStreamedEdgesMode::all:
        return "all";
    }
    throw std::logic_error("Invalid streamed-edge mode.");
}
