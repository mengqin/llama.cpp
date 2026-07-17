#include "ggml.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

struct params {
    std::string source;
    std::string output = "moq_groups_qwen35_4b_decision.json";
    std::string report = "tensor_coverage_report.txt";
    std::string uncovered_csv = "uncovered_tensors.csv";
    std::string model_name = "Qwen3.5-4B";
};

struct tensor_info {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
    int n_dims = 0;
    std::vector<int64_t> shape;
    int64_t n_elements = 0;
    int64_t bytes = 0;
};

struct parsed_tensor {
    std::string name;
    int layer = -1;
    std::string role;
    bool recognized = false;
};

static void usage() {
    std::cout <<
        "llama-moq-make-groups --source model.gguf "
        "[--output moq_groups_qwen35_4b_decision.json] "
        "[--report tensor_coverage_report.txt] "
        "[--uncovered-csv uncovered_tensors.csv]\n";
}

static params parse_args(int argc, char ** argv) {
    params p;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const std::string & name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            usage();
            std::exit(0);
        } else if (a == "--source") {
            p.source = need(a);
        } else if (a == "--output" || a == "--out") {
            p.output = need(a);
        } else if (a == "--report") {
            p.report = need(a);
        } else if (a == "--uncovered-csv") {
            p.uncovered_csv = need(a);
        } else if (a == "--model-name") {
            p.model_name = need(a);
        } else {
            throw std::runtime_error("unknown argument: " + a);
        }
    }
    if (p.source.empty()) {
        throw std::runtime_error("--source is required");
    }
    return p;
}

static bool is_role(const std::string & s) {
    static const std::set<std::string> roles = {
        "attn_qkv",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_gate",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "ssm_alpha",
        "ssm_beta",
        "ssm_out",
        "token_embd",
        "output",
    };
    return roles.find(s) != roles.end();
}

static std::vector<std::string> role_order() {
    return {
        "attn_qkv",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_gate",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "ssm_alpha",
        "ssm_beta",
        "ssm_out",
        "token_embd",
        "output",
    };
}

static parsed_tensor parse_tensor_name(const std::string & name) {
    parsed_tensor out;
    out.name = name;

    if (name == "token_embd.weight") {
        out.role = "token_embd";
        out.recognized = true;
        return out;
    }
    if (name == "output.weight") {
        out.role = "output";
        out.recognized = true;
        return out;
    }

    const std::string prefix = "blk.";
    const std::string suffix = ".weight";
    if (name.rfind(prefix, 0) != 0 || name.size() <= prefix.size() + suffix.size()) {
        return out;
    }
    if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return out;
    }

    const size_t layer_begin = prefix.size();
    const size_t layer_end = name.find('.', layer_begin);
    if (layer_end == std::string::npos || layer_end == layer_begin) {
        return out;
    }
    for (size_t i = layer_begin; i < layer_end; ++i) {
        if (!std::isdigit((unsigned char) name[i])) {
            return out;
        }
    }
    const std::string role = name.substr(layer_end + 1, name.size() - suffix.size() - layer_end - 1);
    if (!is_role(role)) {
        return out;
    }

    out.layer = std::stoi(name.substr(layer_begin, layer_end - layer_begin));
    out.role = role;
    out.recognized = true;
    return out;
}

static std::string band_for_layer(int layer, int n_layers) {
    const int n_edge = std::max(1, (int) (n_layers * 0.20));
    if (layer < n_edge) {
        return "early";
    }
    if (layer >= n_layers - n_edge) {
        return "late";
    }
    return "middle";
}

static std::vector<tensor_info> read_tensors(const std::string & path) {
    ggml_context * ctx_meta = nullptr;
    gguf_init_params gguf_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &ctx_meta,
    };
    gguf_context * ctx = gguf_init_from_file(path.c_str(), gguf_params);
    if (ctx == nullptr || ctx_meta == nullptr) {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
        if (ctx_meta != nullptr) {
            ggml_free(ctx_meta);
        }
        throw std::runtime_error("failed to open GGUF: " + path);
    }

    std::vector<tensor_info> tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t != nullptr; t = ggml_get_next_tensor(ctx_meta, t)) {
        tensor_info ti;
        ti.name = ggml_get_name(t);
        ti.type = t->type;
        ti.n_dims = ggml_n_dims(t);
        ti.n_elements = ggml_nelements(t);
        for (int i = 0; i < ti.n_dims; ++i) {
            ti.shape.push_back(t->ne[i]);
        }
        const int64_t tid = gguf_find_tensor(ctx, ti.name.c_str());
        ti.bytes = tid >= 0 ? gguf_get_tensor_size(ctx, tid) : ggml_nbytes(t);
        tensors.push_back(std::move(ti));
    }
    gguf_free(ctx);
    ggml_free(ctx_meta);
    return tensors;
}

static std::string shape_string(const std::vector<int64_t> & shape) {
    std::ostringstream ss;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            ss << 'x';
        }
        ss << shape[i];
    }
    return ss.str();
}

static std::string role_guess(const std::string & name) {
    parsed_tensor pt = parse_tensor_name(name);
    if (pt.recognized) {
        return pt.role;
    }
    if (name.find("norm") != std::string::npos) {
        return "norm";
    }
    if (name.find("bias") != std::string::npos) {
        return "bias";
    }
    if (name.find("rope") != std::string::npos || name.find("freq") != std::string::npos) {
        return "rope_or_freq";
    }
    if (name.rfind("blk.", 0) == 0 && name.find(".weight") != std::string::npos) {
        return "other_block_weight";
    }
    if (name.find(".weight") != std::string::npos) {
        return "other_weight";
    }
    return "other";
}

static std::string uncovered_reason(const tensor_info & t) {
    parsed_tensor pt = parse_tensor_name(t.name);
    if (pt.recognized) {
        return "covered";
    }
    if (t.name.find("norm") != std::string::npos || t.name.find("bias") != std::string::npos) {
        return "small_fixed_norm_or_bias";
    }
    if (t.n_elements < 1024 * 1024) {
        return "small_fixed_tensor";
    }
    return "unsupported_tensor_slot";
}

int main(int argc, char ** argv) {
    try {
        params p = parse_args(argc, argv);
        std::vector<tensor_info> tensors = read_tensors(p.source);

        std::vector<parsed_tensor> parsed;
        parsed.reserve(tensors.size());
        int max_layer = -1;
        std::map<std::string, int> counts;
        std::map<std::string, tensor_info> tensor_by_name;
        int64_t total_params = 0;
        int64_t total_bytes = 0;
        for (const auto & t : tensors) {
            tensor_by_name[t.name] = t;
            total_params += t.n_elements;
            total_bytes += t.bytes;
            parsed_tensor pt = parse_tensor_name(t.name);
            if (pt.recognized) {
                max_layer = std::max(max_layer, pt.layer);
                counts[pt.name]++;
            }
            parsed.push_back(std::move(pt));
        }
        if (max_layer < 0) {
            throw std::runtime_error("no Qwen-style MoQ tensors were found in source GGUF");
        }
        const int n_layers = max_layer + 1;
        const int n_edge = std::max(1, (int) (n_layers * 0.20));

        std::map<std::string, std::vector<std::string>> grouped;
        std::set<std::string> covered;
        std::vector<tensor_info> uncovered;
        std::vector<std::string> duplicates;
        int64_t covered_params = 0;
        int64_t covered_bytes = 0;

        for (size_t i = 0; i < parsed.size(); ++i) {
            const auto & pt = parsed[i];
            const auto & ti = tensors[i];
            if (pt.recognized) {
                const std::string group_name =
                    pt.layer >= 0 ? pt.role + "_" + band_for_layer(pt.layer, n_layers) : pt.role;
                grouped[group_name].push_back(pt.name);
                if (covered.insert(pt.name).second) {
                    covered_params += ti.n_elements;
                    covered_bytes += ti.bytes;
                }
            } else {
                uncovered.push_back(ti);
            }
        }
        for (const auto & kv : counts) {
            if (kv.second > 1) {
                duplicates.push_back(kv.first);
            }
        }

        for (auto & kv : grouped) {
            std::sort(kv.second.begin(), kv.second.end(), [](const std::string & a, const std::string & b) {
                parsed_tensor pa = parse_tensor_name(a);
                parsed_tensor pb = parse_tensor_name(b);
                if (pa.layer != pb.layer) {
                    return pa.layer < pb.layer;
                }
                return a < b;
            });
        }
        std::sort(uncovered.begin(), uncovered.end(), [](const tensor_info & a, const tensor_info & b) {
            if (a.bytes != b.bytes) {
                return a.bytes > b.bytes;
            }
            return a.name < b.name;
        });

        json out;
        out["metadata"] = {
            {"model", p.model_name},
            {"source", p.source},
            {"n_layers", n_layers},
            {"band_rule", "role x layer band; early=first 20%, middle=middle 60%, late=last 20%"},
            {"early_layers", {0, n_edge - 1}},
            {"middle_layers", {n_edge, n_layers - n_edge - 1}},
            {"late_layers", {n_layers - n_edge, n_layers - 1}},
            {"note", "Expanded non-overlap MoQ decision groups generated from GGUF tensor names. Large supported model weights are grouped; small or unsupported tensors remain fixed."},
        };
        out["groups"] = json::array();

        const std::vector<std::string> bands = {"early", "middle", "late"};
        for (const auto & role : role_order()) {
            if (role == "token_embd" || role == "output") {
                auto it = grouped.find(role);
                if (it != grouped.end() && !it->second.empty()) {
                    json g;
                    g["name"] = role;
                    g["tensors"] = it->second;
                    out["groups"].push_back(std::move(g));
                }
                continue;
            }
            for (const auto & band : bands) {
                const std::string group_name = role + "_" + band;
                auto it = grouped.find(group_name);
                if (it == grouped.end() || it->second.empty()) {
                    continue;
                }
                json g;
                g["name"] = group_name;
                g["tensors"] = it->second;
                out["groups"].push_back(std::move(g));
            }
        }

        {
            std::ofstream f(p.output);
            f << std::setw(2) << out << "\n";
        }
        {
            std::ofstream u(p.uncovered_csv);
            u << "name,type,n_dims,shape,n_elements,bytes,role_guess,reason,can_group\n";
            for (const auto & t : uncovered) {
                u << '"' << t.name << '"' << ','
                  << ggml_type_name(t.type) << ','
                  << t.n_dims << ','
                  << '"' << shape_string(t.shape) << '"' << ','
                  << t.n_elements << ','
                  << t.bytes << ','
                  << role_guess(t.name) << ','
                  << uncovered_reason(t) << ','
                  << "false\n";
            }
        }
        {
            std::ofstream r(p.report);
            r << "MoQ tensor coverage report\n\n";
            r << "source: " << p.source << "\n";
            r << "n_layers: " << n_layers << "\n";
            r << "early_layers: 0.." << (n_edge - 1) << "\n";
            r << "middle_layers: " << n_edge << ".." << (n_layers - n_edge - 1) << "\n";
            r << "late_layers: " << (n_layers - n_edge) << ".." << (n_layers - 1) << "\n\n";
            r << "total_params: " << total_params << "\n";
            r << "total_tensor_bytes: " << total_bytes << "\n";
            r << "covered_params: " << covered_params << "\n";
            r << "covered_bytes: " << covered_bytes << "\n";
            r << "fixed_uncovered_bytes: " << (total_bytes - covered_bytes) << "\n";
            r << "minimum_reachable_bpw: "
              << (total_params > 0 ? (double) (total_bytes - covered_bytes) * 8.0 / (double) total_params : 0.0) << "\n\n";
            r << "covered tensors: " << covered.size() << "\n";
            r << "uncovered tensors: " << uncovered.size() << "\n";
            r << "duplicate tensors: " << duplicates.size() << "\n\n";

            r << "Groups:\n";
            for (const auto & g : out["groups"]) {
                r << "  " << g["name"].get<std::string>() << ": " << g["tensors"].size() << "\n";
            }

            r << "\nUncovered tensors:\n";
            if (uncovered.empty()) {
                r << "  none\n";
            } else {
                for (const auto & t : uncovered) {
                    r << "  " << t.name << " bytes=" << t.bytes
                      << " role_guess=" << role_guess(t.name)
                      << " reason=" << uncovered_reason(t) << "\n";
                }
            }

            std::map<std::string, int64_t> uncovered_by_role;
            for (const auto & t : uncovered) {
                uncovered_by_role[role_guess(t.name)] += t.bytes;
            }
            r << "\nUncovered bytes by role_guess:\n";
            for (const auto & kv : uncovered_by_role) {
                r << "  " << kv.first << ": " << kv.second << "\n";
            }

            r << "\nDuplicate tensors:\n";
            if (duplicates.empty()) {
                r << "  none\n";
            } else {
                for (const auto & name : duplicates) {
                    r << "  " << name << "\n";
                }
            }
        }

        std::cout << "wrote " << p.output << ", " << p.report << " and " << p.uncovered_csv << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
