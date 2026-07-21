#pragma once

#include "nlohmann/json.hpp"

void validate_mace_nonlinear_schema(const nlohmann::json& data);
bool is_published_mh1_architecture(const nlohmann::json& data);
