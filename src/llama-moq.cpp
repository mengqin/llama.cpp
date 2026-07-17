#include "llama-moq.h"

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <cstdio>

static moq_tensor_slot * moq_find_slot(moq_slot_registry & reg, const std::string & name) {
    for (auto & slot : reg.slots) {
        if (slot.name == name) {
            return &slot;
        }
    }
    return nullptr;
}

static void moq_register_slot(moq_slot_registry & reg, const char * name, ggml_tensor ** ref) {
    if (ref == nullptr || *ref == nullptr) {
        return;
    }

    moq_tensor_slot slot;
    slot.name          = name;
    slot.model_ref     = ref;
    slot.base_tensor   = *ref;
    slot.active_tensor = *ref;
    slot.replaced      = false;
    reg.slots.push_back(slot);
}

bool moq_register_tensor_slots(llama_model & model, moq_slot_registry & reg) {
    reg.slots.clear();

    moq_register_slot(reg, "token_embd.weight", &model.tok_embd);
    moq_register_slot(reg, "output.weight",     &model.output);

    char name[128];
    for (size_t il = 0; il < model.layers.size(); ++il) {
        llama_layer & layer = model.layers[il];

        snprintf(name, sizeof(name), "blk.%zu.attn_qkv.weight", il);
        moq_register_slot(reg, name, &layer.wqkv);

        snprintf(name, sizeof(name), "blk.%zu.attn_q.weight", il);
        moq_register_slot(reg, name, &layer.wq);

        snprintf(name, sizeof(name), "blk.%zu.attn_k.weight", il);
        moq_register_slot(reg, name, &layer.wk);

        snprintf(name, sizeof(name), "blk.%zu.attn_v.weight", il);
        moq_register_slot(reg, name, &layer.wv);

        snprintf(name, sizeof(name), "blk.%zu.attn_gate.weight", il);
        moq_register_slot(reg, name, &layer.wqkv_gate);

        snprintf(name, sizeof(name), "blk.%zu.attn_output.weight", il);
        moq_register_slot(reg, name, &layer.wo);

        snprintf(name, sizeof(name), "blk.%zu.ffn_gate.weight", il);
        moq_register_slot(reg, name, &layer.ffn_gate);

        snprintf(name, sizeof(name), "blk.%zu.ffn_up.weight", il);
        moq_register_slot(reg, name, &layer.ffn_up);

        snprintf(name, sizeof(name), "blk.%zu.ffn_down.weight", il);
        moq_register_slot(reg, name, &layer.ffn_down);

        snprintf(name, sizeof(name), "blk.%zu.ssm_alpha.weight", il);
        moq_register_slot(reg, name, &layer.ssm_alpha);

        snprintf(name, sizeof(name), "blk.%zu.ssm_beta.weight", il);
        moq_register_slot(reg, name, &layer.ssm_beta);

        snprintf(name, sizeof(name), "blk.%zu.ssm_out.weight", il);
        moq_register_slot(reg, name, &layer.ssm_out);
    }

    LLAMA_LOG_INFO("%s: registered %zu MoQ tensor slots\n", __func__, reg.slots.size());
    return !reg.slots.empty();
}

bool moq_replace_tensor(moq_slot_registry & reg, const std::string & name, ggml_tensor * new_tensor) {
    moq_tensor_slot * slot = moq_find_slot(reg, name);
    if (slot == nullptr) {
        LLAMA_LOG_WARN("%s: MoQ tensor slot not found: %s\n", __func__, name.c_str());
        return false;
    }
    if (new_tensor == nullptr) {
        LLAMA_LOG_WARN("%s: cannot replace %s with null tensor\n", __func__, name.c_str());
        return false;
    }

    *slot->model_ref    = new_tensor;
    slot->active_tensor = new_tensor;
    slot->replaced      = true;
    return true;
}

bool moq_restore_tensor(moq_slot_registry & reg, const std::string & name) {
    moq_tensor_slot * slot = moq_find_slot(reg, name);
    if (slot == nullptr) {
        LLAMA_LOG_WARN("%s: MoQ tensor slot not found: %s\n", __func__, name.c_str());
        return false;
    }

    *slot->model_ref    = slot->base_tensor;
    slot->active_tensor = slot->base_tensor;
    slot->replaced      = false;
    return true;
}

bool moq_replace_tensor_batch(moq_slot_registry & reg, const std::vector<moq_replacement> & repls) {
    std::vector<moq_tensor_slot *> slots;
    slots.reserve(repls.size());

    for (const auto & repl : repls) {
        moq_tensor_slot * slot = moq_find_slot(reg, repl.name);
        if (slot == nullptr) {
            LLAMA_LOG_WARN("%s: MoQ tensor slot not found: %s\n", __func__, repl.name.c_str());
            return false;
        }
        if (repl.tensor == nullptr) {
            LLAMA_LOG_WARN("%s: cannot replace %s with null tensor\n", __func__, repl.name.c_str());
            return false;
        }
        slots.push_back(slot);
    }

    for (size_t i = 0; i < repls.size(); ++i) {
        moq_tensor_slot * slot = slots[i];
        *slot->model_ref    = repls[i].tensor;
        slot->active_tensor = repls[i].tensor;
        slot->replaced      = true;
    }
    return true;
}

bool moq_restore_tensor_batch(moq_slot_registry & reg, const std::vector<std::string> & names) {
    std::vector<moq_tensor_slot *> slots;
    slots.reserve(names.size());

    for (const auto & name : names) {
        moq_tensor_slot * slot = moq_find_slot(reg, name);
        if (slot == nullptr) {
            LLAMA_LOG_WARN("%s: MoQ tensor slot not found: %s\n", __func__, name.c_str());
            return false;
        }
        slots.push_back(slot);
    }

    for (moq_tensor_slot * slot : slots) {
        *slot->model_ref    = slot->base_tensor;
        slot->active_tensor = slot->base_tensor;
        slot->replaced      = false;
    }
    return true;
}

void moq_restore_all(moq_slot_registry & reg) {
    for (auto & slot : reg.slots) {
        if (slot.model_ref != nullptr && slot.base_tensor != nullptr) {
            *slot.model_ref    = slot.base_tensor;
            slot.active_tensor = slot.base_tensor;
            slot.replaced      = false;
        }
    }
}

ggml_tensor * moq_get_base_tensor(moq_slot_registry & reg, const std::string & name) {
    moq_tensor_slot * slot = moq_find_slot(reg, name);
    return slot ? slot->base_tensor : nullptr;
}

ggml_backend_buffer_type_t moq_get_base_tensor_buft(moq_slot_registry & reg, const std::string & name) {
    ggml_tensor * tensor = moq_get_base_tensor(reg, name);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return ggml_backend_cpu_buffer_type();
    }
    return ggml_backend_buffer_get_type(tensor->buffer);
}

void moq_invalidate_graph(llama_context * ctx) {
    if (ctx != nullptr) {
        ctx->invalidate_graph();
    }
}
