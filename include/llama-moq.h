#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include "llama.h"

#include <string>
#include <vector>

struct moq_tensor_slot {
    std::string name;
    ggml_tensor ** model_ref = nullptr;
    ggml_tensor * base_tensor = nullptr;
    ggml_tensor * active_tensor = nullptr;
    bool replaced = false;
};

struct moq_slot_registry {
    std::vector<moq_tensor_slot> slots;
};

struct moq_replacement {
    std::string name;
    ggml_tensor * tensor = nullptr;
};

LLAMA_API bool moq_register_tensor_slots(llama_model & model, moq_slot_registry & reg);

LLAMA_API bool moq_replace_tensor(
    moq_slot_registry & reg,
    const std::string & name,
    ggml_tensor * new_tensor);

LLAMA_API bool moq_restore_tensor(
    moq_slot_registry & reg,
    const std::string & name);

LLAMA_API bool moq_replace_tensor_batch(
    moq_slot_registry & reg,
    const std::vector<moq_replacement> & repls);

LLAMA_API bool moq_restore_tensor_batch(
    moq_slot_registry & reg,
    const std::vector<std::string> & names);

LLAMA_API void moq_restore_all(moq_slot_registry & reg);

LLAMA_API ggml_tensor * moq_get_base_tensor(
    moq_slot_registry & reg,
    const std::string & name);

LLAMA_API ggml_backend_buffer_type_t moq_get_base_tensor_buft(
    moq_slot_registry & reg,
    const std::string & name);

LLAMA_API void moq_invalidate_graph(llama_context * ctx);
