#include "ggml.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

struct params {
    std::string elasticity;
    std::string groups;
    std::string model;
    std::string out_dir = "temp/moq_solver";
    std::vector<double> target_bpw;
    std::vector<std::string> solvers = {"lambda", "greedy"};

    double loss_mean_weight = 1.0;
    double loss_p999_weight = 0.30;
    double loss_p99_weight  = 0.0;
    double loss_ppl_weight  = 0.0;
    double loss_max_weight  = 0.0;

    bool has_fixed_bytes = false;
    bool has_total_params = false;
    bool allow_overlap_diagnostic = false;
    bool emit_topk_recipes = false;
    bool auto_tail_guard = false;
    bool export_tensor_types = false;
    int top_k = 1;
    double fixed_bytes = 0.0;
    double total_params = 0.0;
    std::string tail_guard_metric = "p999,max";
    double tail_guard_target_ratio = 4.0;
    double tail_guard_absolute_p999 = 0.0;
    double tail_guard_absolute_max = 0.0;
    double tail_guard_min_upgrade_gain = 0.0;
};

struct group_def {
    std::string name;
    std::vector<std::string> tensors;
};

struct group_overlap_report {
    std::vector<std::string> groups_with_no_tensors;
    std::vector<std::string> groups_missing_elasticity;
    std::map<std::string, std::vector<std::string>> tensor_to_groups;
    std::map<std::string, std::vector<std::string>> overlapped_tensors;
    bool has_overlap = false;
};

struct tensor_stat {
    double params = 0.0;
    double bytes = 0.0;
    ggml_type type = GGML_TYPE_COUNT;
    std::vector<int64_t> shape;
};

struct model_stats {
    bool available = false;
    std::string path;
    double total_params = 0.0;
    double total_tensor_bytes = 0.0;
    double covered_params = 0.0;
    double covered_model_bytes = 0.0;
    double covered_source_bytes = 0.0;
    double fixed_uncovered_bytes = 0.0;
    std::map<std::string, tensor_stat> tensors;
    std::vector<std::string> missing_group_tensors;
};

struct csv_table {
    std::vector<std::string> header;
    std::vector<std::unordered_map<std::string, std::string>> rows;
};

struct candidate {
    std::string group;
    std::string qtype;
    int n_tensors = 0;
    double source_bytes = 0.0;
    double quant_bytes = 0.0;
    double bpw_delta = 0.0;
    double ppl = 0.0;
    double mean_kld = 0.0;
    double max_kld = 0.0;
    double p99_kld = 0.0;
    double p999_kld = 0.0;
    double loss = 0.0;
    int quality_rank = 0;
    int size_rank = 0;
    bool input_dominated = false;
    bool dominated = false;
    std::string dominated_by;
};

struct tail_guard_entry {
    std::string group;
    std::string qtype;
    double quant_bytes = 0.0;
    double mean_kld = 0.0;
    double p999_kld = 0.0;
    double max_kld = 0.0;
    double loss = 0.0;
    double tail_score = 0.0;
    bool tail_risky = false;
    std::string reason;
    std::string min_safe_qtype;
};

struct tail_guard_group {
    std::string group;
    std::string min_safe_qtype;
    std::vector<std::string> forbid_qtypes;
};

struct tail_guard_result {
    std::vector<tail_guard_entry> entries;
    std::map<std::string, tail_guard_group> groups;
};

struct group_data {
    std::string name;
    std::vector<candidate> all;
    std::vector<candidate> pruned;
    double source_bytes = 0.0;
    double source_elements = 0.0;
};

struct recipe {
    std::string solver;
    double target_bpw = 0.0;
    double estimated_bpw = 0.0;
    double relative_group_bpw = 0.0;
    double absolute_model_bpw = 0.0;
    double budget_bytes = 0.0;
    double group_bytes = 0.0;
    double total_bytes = 0.0;
    double fixed_bytes = 0.0;
    double covered_bytes = 0.0;
    double total_loss = 0.0;
    double predicted_ppl = 0.0;
    double predicted_mean_kld = 0.0;
    double predicted_p99_kld = 0.0;
    double predicted_p999_kld = 0.0;
    double predicted_max_kld = 0.0;
    bool over_budget = false;
    std::map<std::string, candidate> choices;
};

static void usage() {
    std::cout <<
        "llama-moq-solve --elasticity elasticity_table.csv --groups groups.json "
        "--target-bpw 3.5,4.0 --solver lambda,greedy --out-dir temp/moq_solver "
        "[--allow-overlap-diagnostic]\n";
}

static std::vector<std::string> split_csv_line(const std::string & line) {
    std::vector<std::string> out;
    std::string cur;
    bool quote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (quote) {
            if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
                cur.push_back('"');
                ++i;
            } else if (c == '"') {
                quote = false;
            } else {
                cur.push_back(c);
            }
        } else if (c == '"') {
            quote = true;
        } else if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

static csv_table read_csv(const fs::path & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open CSV: " + path.string());
    }
    csv_table table;
    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("empty CSV: " + path.string());
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    table.header = split_csv_line(line);
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        auto fields = split_csv_line(line);
        std::unordered_map<std::string, std::string> row;
        for (size_t i = 0; i < table.header.size(); ++i) {
            row[table.header[i]] = i < fields.size() ? fields[i] : "";
        }
        table.rows.push_back(std::move(row));
    }
    return table;
}

static std::string get_s(const std::unordered_map<std::string, std::string> & row, const std::string & key) {
    auto it = row.find(key);
    return it == row.end() ? "" : it->second;
}

static double get_d(const std::unordered_map<std::string, std::string> & row, const std::string & key, double def = 0.0) {
    auto s = get_s(row, key);
    if (s.empty()) {
        return def;
    }
    try {
        return std::stod(s);
    } catch (...) {
        return def;
    }
}

static int get_i(const std::unordered_map<std::string, std::string> & row, const std::string & key, int def = 0) {
    auto s = get_s(row, key);
    if (s.empty()) {
        return def;
    }
    try {
        return std::stoi(s);
    } catch (...) {
        return def;
    }
}

static bool get_b(const std::unordered_map<std::string, std::string> & row, const std::string & key) {
    auto s = get_s(row, key);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s == "true" || s == "1" || s == "yes";
}

static std::vector<std::string> split_list(const std::string & s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char c) { return !std::isspace(c); }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char c) { return !std::isspace(c); }).base(), item.end());
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

static std::vector<double> split_double_list(const std::string & s) {
    std::vector<double> out;
    for (const auto & item : split_list(s)) {
        out.push_back(std::stod(item));
    }
    return out;
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
        } else if (a == "--elasticity") {
            p.elasticity = need(a);
        } else if (a == "--groups") {
            p.groups = need(a);
        } else if (a == "--model") {
            p.model = need(a);
        } else if (a == "--target-bpw") {
            p.target_bpw = split_double_list(need(a));
        } else if (a == "--solver") {
            p.solvers = split_list(need(a));
        } else if (a == "--out-dir") {
            p.out_dir = need(a);
        } else if (a == "--loss-mean-weight") {
            p.loss_mean_weight = std::stod(need(a));
        } else if (a == "--loss-p999-weight") {
            p.loss_p999_weight = std::stod(need(a));
        } else if (a == "--loss-p99-weight") {
            p.loss_p99_weight = std::stod(need(a));
        } else if (a == "--loss-ppl-weight") {
            p.loss_ppl_weight = std::stod(need(a));
        } else if (a == "--loss-max-weight") {
            p.loss_max_weight = std::stod(need(a));
        } else if (a == "--fixed-bytes") {
            p.fixed_bytes = std::stod(need(a));
            p.has_fixed_bytes = true;
        } else if (a == "--total-params") {
            p.total_params = std::stod(need(a));
            p.has_total_params = true;
        } else if (a == "--allow-overlap-diagnostic") {
            p.allow_overlap_diagnostic = true;
        } else if (a == "--top-k") {
            p.top_k = std::stoi(need(a));
            if (p.top_k <= 0) {
                throw std::runtime_error("--top-k must be positive");
            }
        } else if (a == "--emit-topk-recipes") {
            p.emit_topk_recipes = true;
        } else if (a == "--auto-tail-guard") {
            std::string v = need(a);
            std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char) std::tolower(c); });
            if (v != "on" && v != "off") {
                throw std::runtime_error("--auto-tail-guard must be on or off");
            }
            p.auto_tail_guard = v == "on";
        } else if (a == "--tail-guard-metric") {
            p.tail_guard_metric = need(a);
        } else if (a == "--tail-guard-target-ratio") {
            p.tail_guard_target_ratio = std::stod(need(a));
        } else if (a == "--tail-guard-absolute-p999") {
            p.tail_guard_absolute_p999 = std::stod(need(a));
        } else if (a == "--tail-guard-absolute-max") {
            p.tail_guard_absolute_max = std::stod(need(a));
        } else if (a == "--tail-guard-min-upgrade-gain") {
            p.tail_guard_min_upgrade_gain = std::stod(need(a));
        } else if (a == "--export-tensor-types") {
            p.export_tensor_types = true;
        } else {
            throw std::runtime_error("unknown argument: " + a);
        }
    }
    if (p.elasticity.empty()) {
        throw std::runtime_error("--elasticity is required");
    }
    if (p.groups.empty()) {
        throw std::runtime_error("--groups is required");
    }
    if (p.target_bpw.empty()) {
        throw std::runtime_error("--target-bpw is required");
    }
    for (const auto & s : p.solvers) {
        if (s != "lambda" && s != "greedy") {
            throw std::runtime_error("unsupported solver: " + s);
        }
    }
    std::sort(p.target_bpw.begin(), p.target_bpw.end());
    return p;
}

static std::vector<group_def> read_group_defs(const fs::path & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open groups JSON: " + path.string());
    }
    json j;
    in >> j;
    std::vector<group_def> out;
    if (!j.contains("groups") || !j["groups"].is_array()) {
        throw std::runtime_error("groups JSON missing array: groups");
    }
    int idx = 0;
    for (const auto & g : j["groups"]) {
        group_def gd;
        gd.name = g.value("name", "group_" + std::to_string(idx++));
        if (g.contains("tensors") && g["tensors"].is_array()) {
            for (const auto & t : g["tensors"]) {
                if (!t.is_string()) {
                    throw std::runtime_error("group contains non-string tensor: " + gd.name);
                }
                gd.tensors.push_back(t.get<std::string>());
            }
        }
        out.push_back(std::move(gd));
    }
    return out;
}

static std::vector<std::string> group_order(const std::vector<group_def> & defs) {
    std::vector<std::string> out;
    out.reserve(defs.size());
    for (const auto & g : defs) {
        out.push_back(g.name);
    }
    return out;
}

static group_overlap_report analyze_group_overlap(
        const std::vector<group_def> & defs,
        const std::map<std::string, group_data> & elasticity_groups) {
    group_overlap_report report;
    for (const auto & g : defs) {
        if (g.tensors.empty()) {
            report.groups_with_no_tensors.push_back(g.name);
        }
        if (elasticity_groups.find(g.name) == elasticity_groups.end()) {
            report.groups_missing_elasticity.push_back(g.name);
        }
        std::unordered_set<std::string> seen_in_group;
        for (const auto & t : g.tensors) {
            if (!seen_in_group.insert(t).second) {
                continue;
            }
            report.tensor_to_groups[t].push_back(g.name);
        }
    }
    for (const auto & kv : report.tensor_to_groups) {
        if (kv.second.size() > 1) {
            report.overlapped_tensors[kv.first] = kv.second;
        }
    }
    report.has_overlap = !report.overlapped_tensors.empty();
    return report;
}

static void write_group_overlap_report(const fs::path & path, const group_overlap_report & report) {
    std::ofstream out(path);
    out << "MoQ group overlap report\n\n";
    out << "overlap: " << (report.has_overlap ? "true" : "false") << "\n";
    out << "overlapped_tensors: " << report.overlapped_tensors.size() << "\n";
    out << "groups_with_no_tensors: " << report.groups_with_no_tensors.size() << "\n";
    out << "groups_missing_elasticity_rows: " << report.groups_missing_elasticity.size() << "\n\n";

    out << "Overlapped tensors:\n";
    if (report.overlapped_tensors.empty()) {
        out << "  none\n";
    } else {
        for (const auto & kv : report.overlapped_tensors) {
            out << "  " << kv.first << ":";
            for (const auto & g : kv.second) {
                out << " " << g;
            }
            out << "\n";
        }
    }

    out << "\nGroups with no tensors:\n";
    if (report.groups_with_no_tensors.empty()) {
        out << "  none\n";
    } else {
        for (const auto & g : report.groups_with_no_tensors) {
            out << "  " << g << "\n";
        }
    }

    out << "\nGroups missing elasticity rows:\n";
    if (report.groups_missing_elasticity.empty()) {
        out << "  none\n";
    } else {
        for (const auto & g : report.groups_missing_elasticity) {
            out << "  " << g << "\n";
        }
    }
}

static model_stats read_model_stats(
        const std::string & path,
        const std::vector<group_def> & defs,
        const std::map<std::string, group_data> & elasticity_groups) {
    model_stats stats;
    if (path.empty()) {
        return stats;
    }
    stats.available = true;
    stats.path = path;

    ggml_context * ctx_meta = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &ctx_meta,
    };
    gguf_context * ctx_gguf = gguf_init_from_file(path.c_str(), params);
    if (ctx_gguf == nullptr || ctx_meta == nullptr) {
        if (ctx_gguf != nullptr) {
            gguf_free(ctx_gguf);
        }
        if (ctx_meta != nullptr) {
            ggml_free(ctx_meta);
        }
        throw std::runtime_error("failed to open model GGUF for BPW stats: " + path);
    }

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t != nullptr; t = ggml_get_next_tensor(ctx_meta, t)) {
        const std::string name = ggml_get_name(t);
        tensor_stat ts;
        ts.params = (double) ggml_nelements(t);
        ts.type = t->type;
        const int n_dims = ggml_n_dims(t);
        for (int i = 0; i < n_dims; ++i) {
            ts.shape.push_back(t->ne[i]);
        }
        const int64_t tid = gguf_find_tensor(ctx_gguf, name.c_str());
        ts.bytes = tid >= 0 ? (double) gguf_get_tensor_size(ctx_gguf, tid) : (double) ggml_nbytes(t);
        stats.tensors[name] = ts;
        stats.total_params += ts.params;
        stats.total_tensor_bytes += ts.bytes;
    }

    std::set<std::string> covered;
    for (const auto & g : defs) {
        for (const auto & tensor_name : g.tensors) {
            if (!covered.insert(tensor_name).second) {
                continue;
            }
            auto it = stats.tensors.find(tensor_name);
            if (it == stats.tensors.end()) {
                stats.missing_group_tensors.push_back(tensor_name);
                continue;
            }
            stats.covered_params += it->second.params;
            stats.covered_model_bytes += it->second.bytes;
        }
        auto eit = elasticity_groups.find(g.name);
        if (eit != elasticity_groups.end()) {
            stats.covered_source_bytes += eit->second.source_bytes;
        }
    }
    stats.fixed_uncovered_bytes = std::max(0.0, stats.total_tensor_bytes - stats.covered_model_bytes);

    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);
    return stats;
}

static double compute_loss(const candidate & c, const params & p) {
    return p.loss_mean_weight * c.mean_kld +
           p.loss_p999_weight * c.p999_kld +
           p.loss_p99_weight  * c.p99_kld +
           p.loss_ppl_weight  * c.ppl +
           p.loss_max_weight  * c.max_kld;
}

static std::map<std::string, group_data> read_elasticity(const params & p) {
    auto table = read_csv(p.elasticity);
    std::map<std::string, group_data> groups;
    for (const auto & row : table.rows) {
        candidate c;
        c.group = get_s(row, "group");
        c.qtype = get_s(row, "qtype");
        if (c.group.empty() || c.qtype.empty()) {
            continue;
        }
        c.n_tensors = get_i(row, "n_tensors");
        c.source_bytes = get_d(row, "source_bytes");
        c.quant_bytes = get_d(row, "quant_bytes");
        c.bpw_delta = get_d(row, "bpw_delta");
        c.ppl = get_d(row, "ppl");
        c.mean_kld = get_d(row, "mean_kld");
        c.max_kld = get_d(row, "max_kld");
        c.p99_kld = get_d(row, "p99_kld");
        c.p999_kld = get_d(row, "p999_kld");
        c.quality_rank = get_i(row, "quality_rank");
        c.size_rank = get_i(row, "size_rank");
        c.input_dominated = get_b(row, "pareto_dominated");
        c.loss = compute_loss(c, p);
        auto & gd = groups[c.group];
        gd.name = c.group;
        gd.source_bytes = std::max(gd.source_bytes, c.source_bytes);
        gd.all.push_back(std::move(c));
    }
    for (auto & kv : groups) {
        kv.second.source_elements = kv.second.source_bytes / 2.0;
    }
    return groups;
}

static void pareto_prune(std::map<std::string, group_data> & groups) {
    constexpr double eps = 1e-15;
    for (auto & kv : groups) {
        auto & gd = kv.second;
        for (auto & c : gd.all) {
            for (const auto & o : gd.all) {
                if (o.qtype == c.qtype) {
                    continue;
                }
                const bool no_larger = o.quant_bytes <= c.quant_bytes + eps;
                const bool no_worse  = o.loss <= c.loss + eps;
                const bool strict    = o.quant_bytes < c.quant_bytes - eps || o.loss < c.loss - eps;
                if (no_larger && no_worse && strict) {
                    c.dominated = true;
                    c.dominated_by = o.qtype;
                    break;
                }
            }
        }
        gd.pruned.clear();
        for (const auto & c : gd.all) {
            if (!c.dominated) {
                gd.pruned.push_back(c);
            }
        }
        std::sort(gd.pruned.begin(), gd.pruned.end(), [](const candidate & a, const candidate & b) {
            if (a.quant_bytes != b.quant_bytes) {
                return a.quant_bytes < b.quant_bytes;
            }
            if (a.loss != b.loss) {
                return a.loss < b.loss;
            }
            return a.qtype < b.qtype;
        });
    }
}

static tail_guard_result apply_auto_tail_guard(
        std::map<std::string, group_data> & groups,
        const params & p) {
    tail_guard_result result;
    if (!p.auto_tail_guard) {
        return result;
    }

    const double ratio = std::max(1.0, p.tail_guard_target_ratio);
    for (auto & kv : groups) {
        auto & gd = kv.second;
        const auto candidates = gd.pruned;
        if (candidates.empty()) {
            continue;
        }

        double best_p999 = std::numeric_limits<double>::infinity();
        double best_max = std::numeric_limits<double>::infinity();
        for (const auto & c : candidates) {
            best_p999 = std::min(best_p999, std::max(0.0, c.p999_kld));
            best_max = std::min(best_max, std::max(0.0, c.max_kld));
        }
        best_p999 = std::max(best_p999, 1e-12);
        best_max = std::max(best_max, 1e-12);

        std::vector<tail_guard_entry> local;
        local.reserve(candidates.size());
        for (const auto & c : candidates) {
            tail_guard_entry e;
            e.group = gd.name;
            e.qtype = c.qtype;
            e.quant_bytes = c.quant_bytes;
            e.mean_kld = c.mean_kld;
            e.p999_kld = c.p999_kld;
            e.max_kld = c.max_kld;
            e.loss = c.loss;
            e.tail_score = c.p999_kld / best_p999;
            if (p.tail_guard_metric.find("max") != std::string::npos) {
                e.tail_score += 0.50 * c.max_kld / best_max;
            }

            std::vector<std::string> reasons;
            if (c.p999_kld > best_p999 * ratio) {
                reasons.push_back("p999_ratio");
            }
            if (p.tail_guard_metric.find("max") != std::string::npos && c.max_kld > best_max * ratio) {
                reasons.push_back("max_ratio");
            }
            if (p.tail_guard_absolute_p999 > 0.0 && c.p999_kld > p.tail_guard_absolute_p999) {
                reasons.push_back("p999_absolute");
            }
            if (p.tail_guard_absolute_max > 0.0 && c.max_kld > p.tail_guard_absolute_max) {
                reasons.push_back("max_absolute");
            }
            e.tail_risky = !reasons.empty();
            for (size_t i = 0; i < reasons.size(); ++i) {
                if (i > 0) {
                    e.reason += "|";
                }
                e.reason += reasons[i];
            }
            local.push_back(std::move(e));
        }

        std::sort(local.begin(), local.end(), [](const auto & a, const auto & b) {
            if (a.quant_bytes != b.quant_bytes) {
                return a.quant_bytes < b.quant_bytes;
            }
            return a.loss < b.loss;
        });

        const tail_guard_entry * min_safe = nullptr;
        for (const auto & e : local) {
            if (!e.tail_risky) {
                min_safe = &e;
                break;
            }
        }
        if (min_safe == nullptr) {
            min_safe = &*std::min_element(local.begin(), local.end(), [](const auto & a, const auto & b) {
                if (a.tail_score != b.tail_score) {
                    return a.tail_score < b.tail_score;
                }
                return a.quant_bytes < b.quant_bytes;
            });
        }

        tail_guard_group gg;
        gg.group = gd.name;
        gg.min_safe_qtype = min_safe->qtype;
        for (auto & e : local) {
            e.min_safe_qtype = min_safe->qtype;
            if (e.tail_risky && e.qtype != min_safe->qtype) {
                gg.forbid_qtypes.push_back(e.qtype);
            }
            result.entries.push_back(e);
        }
        result.groups[gd.name] = gg;

        std::vector<candidate> filtered;
        for (const auto & c : gd.pruned) {
            bool forbid = false;
            for (const auto & q : gg.forbid_qtypes) {
                if (c.qtype == q) {
                    forbid = true;
                    break;
                }
            }
            if (!forbid || c.qtype == gg.min_safe_qtype) {
                filtered.push_back(c);
            }
        }
        if (!filtered.empty()) {
            std::sort(filtered.begin(), filtered.end(), [](const candidate & a, const candidate & b) {
                if (a.quant_bytes != b.quant_bytes) {
                    return a.quant_bytes < b.quant_bytes;
                }
                if (a.loss != b.loss) {
                    return a.loss < b.loss;
                }
                return a.qtype < b.qtype;
            });
            gd.pruned = std::move(filtered);
        }
    }
    return result;
}

static void write_auto_tail_guard_report(const fs::path & path, const tail_guard_result & guard) {
    std::ofstream out(path);
    out << "group,qtype,quant_bytes,mean_kld,p999_kld,max_kld,loss,tail_score,tail_risky,reason,min_safe_qtype\n";
    for (const auto & e : guard.entries) {
        out << e.group << ','
            << e.qtype << ','
            << e.quant_bytes << ','
            << e.mean_kld << ','
            << e.p999_kld << ','
            << e.max_kld << ','
            << e.loss << ','
            << e.tail_score << ','
            << (e.tail_risky ? "true" : "false") << ','
            << e.reason << ','
            << e.min_safe_qtype << "\n";
    }
}

static void write_auto_tail_guard_json(const fs::path & path, const tail_guard_result & guard, const params & p) {
    json j;
    j["mode"] = "auto_tail_guard";
    j["enabled"] = p.auto_tail_guard;
    j["metric"] = p.tail_guard_metric;
    j["target_ratio"] = p.tail_guard_target_ratio;
    j["absolute_p999"] = p.tail_guard_absolute_p999;
    j["absolute_max"] = p.tail_guard_absolute_max;
    j["min_upgrade_gain"] = p.tail_guard_min_upgrade_gain;
    j["groups"] = json::object();
    for (const auto & kv : guard.groups) {
        j["groups"][kv.first] = {
            {"min_safe_qtype", kv.second.min_safe_qtype},
            {"forbid_qtypes", kv.second.forbid_qtypes},
        };
    }
    std::ofstream out(path);
    out << std::setw(2) << j << "\n";
}

static double source_elements_total(const std::vector<std::string> & order, const std::map<std::string, group_data> & groups) {
    double total = 0.0;
    for (const auto & name : order) {
        auto it = groups.find(name);
        if (it != groups.end()) {
            total += it->second.source_elements;
        }
    }
    return total;
}

static double budget_for_target(double target_bpw, const params & p, double rel_source_elements, const model_stats & stats) {
    if (stats.available) {
        return target_bpw * stats.total_params / 8.0 - stats.fixed_uncovered_bytes;
    }
    if (p.has_total_params) {
        return target_bpw * p.total_params / 8.0 - (p.has_fixed_bytes ? p.fixed_bytes : 0.0);
    }
    return target_bpw * rel_source_elements / 8.0;
}

static void finalize_recipe(recipe & r, const params & p, double rel_source_elements, const model_stats & stats) {
    r.group_bytes = 0.0;
    r.total_loss = 0.0;
    r.predicted_ppl = 0.0;
    r.predicted_mean_kld = 0.0;
    r.predicted_p99_kld = 0.0;
    r.predicted_p999_kld = 0.0;
    r.predicted_max_kld = 0.0;
    for (const auto & kv : r.choices) {
        r.group_bytes += kv.second.quant_bytes;
        r.total_loss  += kv.second.loss;
        r.predicted_ppl += kv.second.ppl;
        r.predicted_mean_kld += kv.second.mean_kld;
        r.predicted_p99_kld += kv.second.p99_kld;
        r.predicted_p999_kld += kv.second.p999_kld;
        r.predicted_max_kld = std::max(r.predicted_max_kld, kv.second.max_kld);
    }
    if (!r.choices.empty()) {
        r.predicted_ppl /= (double) r.choices.size();
    }
    r.covered_bytes = r.group_bytes;
    if (stats.available) {
        r.fixed_bytes = stats.fixed_uncovered_bytes;
        r.total_bytes = r.group_bytes + stats.fixed_uncovered_bytes;
        r.absolute_model_bpw = stats.total_params > 0.0 ? r.total_bytes * 8.0 / stats.total_params : 0.0;
        r.relative_group_bpw = stats.covered_params > 0.0 ? r.group_bytes * 8.0 / stats.covered_params : 0.0;
        r.estimated_bpw = r.absolute_model_bpw;
    } else {
        r.fixed_bytes = p.has_fixed_bytes ? p.fixed_bytes : 0.0;
        r.total_bytes = r.group_bytes + r.fixed_bytes;
        r.relative_group_bpw = rel_source_elements > 0.0 ? r.group_bytes * 8.0 / rel_source_elements : 0.0;
        r.absolute_model_bpw = p.has_total_params ? r.total_bytes * 8.0 / p.total_params : 0.0;
        r.estimated_bpw = p.has_total_params ? r.absolute_model_bpw : r.relative_group_bpw;
    }
    if (stats.available) {
        // already set above
    } else if (p.has_total_params) {
        r.estimated_bpw = r.total_bytes * 8.0 / p.total_params;
    } else {
        r.estimated_bpw = rel_source_elements > 0.0 ? r.group_bytes * 8.0 / rel_source_elements : 0.0;
    }
    r.over_budget = r.group_bytes > r.budget_bytes + 0.5;
}

static std::string recipe_key(const recipe & r) {
    std::ostringstream ss;
    for (const auto & kv : r.choices) {
        ss << kv.first << '=' << kv.second.qtype << ';';
    }
    return ss.str();
}

static recipe make_lambda_recipe(
        double lambda,
        double target_bpw,
        const params & p,
        const std::vector<std::string> & order,
        const std::map<std::string, group_data> & groups,
        double rel_source_elements,
        const model_stats & stats) {
    recipe r;
    r.solver = "lambda";
    r.target_bpw = target_bpw;
    r.budget_bytes = budget_for_target(target_bpw, p, rel_source_elements, stats);
    for (const auto & name : order) {
        const auto & cs = groups.at(name).pruned;
        const candidate * best = &cs.front();
        double best_score = best->loss + lambda * best->quant_bytes;
        for (const auto & c : cs) {
            double score = c.loss + lambda * c.quant_bytes;
            if (score < best_score - 1e-18 ||
                    (std::abs(score - best_score) <= 1e-18 && c.quant_bytes < best->quant_bytes)) {
                best = &c;
                best_score = score;
            }
        }
        r.choices[name] = *best;
    }
    finalize_recipe(r, p, rel_source_elements, stats);
    return r;
}

static std::vector<double> build_lambdas(const std::vector<std::string> & order, const std::map<std::string, group_data> & groups) {
    std::vector<double> lambdas = {0.0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10};
    for (const auto & name : order) {
        const auto & cs = groups.at(name).pruned;
        for (size_t i = 0; i < cs.size(); ++i) {
            for (size_t j = 0; j < cs.size(); ++j) {
                if (i == j || cs[i].quant_bytes == cs[j].quant_bytes) {
                    continue;
                }
                double lambda = (cs[i].loss - cs[j].loss) / (cs[j].quant_bytes - cs[i].quant_bytes);
                if (std::isfinite(lambda) && lambda >= 0.0) {
                    lambdas.push_back(lambda);
                    lambdas.push_back(lambda * 0.999);
                    lambdas.push_back(lambda * 1.001);
                }
            }
        }
    }
    lambdas.push_back(1e-9);
    lambdas.push_back(1e-8);
    std::sort(lambdas.begin(), lambdas.end());
    std::vector<double> uniq;
    for (double v : lambdas) {
        if (v < 0 || !std::isfinite(v)) {
            continue;
        }
        if (uniq.empty() || std::abs(v - uniq.back()) > std::max(1e-18, std::abs(v) * 1e-8)) {
            uniq.push_back(v);
        }
    }
    return uniq;
}

static recipe choose_for_budget(std::vector<recipe> recipes, double target_bpw, const params & p, double rel_source_elements, const model_stats & stats) {
    const double budget = budget_for_target(target_bpw, p, rel_source_elements, stats);
    recipe * best = nullptr;
    for (auto & r : recipes) {
        r.target_bpw = target_bpw;
        r.budget_bytes = budget;
        finalize_recipe(r, p, rel_source_elements, stats);
        if (r.group_bytes <= budget + 0.5) {
            if (best == nullptr ||
                    r.total_loss < best->total_loss - 1e-15 ||
                    (std::abs(r.total_loss - best->total_loss) <= 1e-15 && r.group_bytes > best->group_bytes)) {
                best = &r;
            }
        }
    }
    if (best != nullptr) {
        return *best;
    }
    return *std::min_element(recipes.begin(), recipes.end(), [](const recipe & a, const recipe & b) {
        if (a.group_bytes != b.group_bytes) {
            return a.group_bytes < b.group_bytes;
        }
        return a.total_loss < b.total_loss;
    });
}

static std::vector<recipe> build_lambda_pool(
        const params & p,
        const std::vector<std::string> & order,
        const std::map<std::string, group_data> & groups,
        double rel_source_elements,
        const model_stats & stats) {
    std::vector<recipe> pool;
    std::set<std::string> seen;
    for (double lambda : build_lambdas(order, groups)) {
        recipe r = make_lambda_recipe(lambda, p.target_bpw.back(), p, order, groups, rel_source_elements, stats);
        auto key = recipe_key(r);
        if (seen.insert(key).second) {
            pool.push_back(std::move(r));
        }
    }
    return pool;
}

static recipe solve_greedy(
        double target_bpw,
        const params & p,
        const std::vector<std::string> & order,
        const std::map<std::string, group_data> & groups,
        double rel_source_elements,
        const model_stats & stats) {
    recipe r;
    r.solver = "greedy";
    r.target_bpw = target_bpw;
    r.budget_bytes = budget_for_target(target_bpw, p, rel_source_elements, stats);
    for (const auto & name : order) {
        r.choices[name] = groups.at(name).pruned.front();
    }
    finalize_recipe(r, p, rel_source_elements, stats);

    while (true) {
        const std::string * best_group = nullptr;
        const candidate * best_cand = nullptr;
        double best_ratio = 0.0;
        double best_gain = 0.0;
        for (const auto & name : order) {
            const auto & cur = r.choices[name];
            for (const auto & c : groups.at(name).pruned) {
                double db = c.quant_bytes - cur.quant_bytes;
                double gain = cur.loss - c.loss;
                if (db <= 0.5 || gain <= 1e-18) {
                    continue;
                }
                if (r.group_bytes + db > r.budget_bytes + 0.5) {
                    continue;
                }
                double ratio = gain / db;
                if (best_cand == nullptr ||
                        ratio > best_ratio + 1e-24 ||
                        (std::abs(ratio - best_ratio) <= 1e-24 && gain > best_gain)) {
                    best_group = &name;
                    best_cand = &c;
                    best_ratio = ratio;
                    best_gain = gain;
                }
            }
        }
        if (best_cand == nullptr || best_group == nullptr) {
            break;
        }
        r.choices[*best_group] = *best_cand;
        finalize_recipe(r, p, rel_source_elements, stats);
    }
    return r;
}

static std::vector<recipe> solve_greedy_path(
        double target_bpw,
        const params & p,
        const std::vector<std::string> & order,
        const std::map<std::string, group_data> & groups,
        double rel_source_elements,
        const model_stats & stats) {
    std::vector<recipe> path;
    recipe r;
    r.solver = "greedy";
    r.target_bpw = target_bpw;
    r.budget_bytes = budget_for_target(target_bpw, p, rel_source_elements, stats);
    for (const auto & name : order) {
        r.choices[name] = groups.at(name).pruned.front();
    }
    finalize_recipe(r, p, rel_source_elements, stats);
    path.push_back(r);

    while (true) {
        const std::string * best_group = nullptr;
        const candidate * best_cand = nullptr;
        double best_ratio = 0.0;
        double best_gain = 0.0;
        for (const auto & name : order) {
            const auto & cur = r.choices[name];
            for (const auto & c : groups.at(name).pruned) {
                double db = c.quant_bytes - cur.quant_bytes;
                double gain = cur.loss - c.loss;
                if (db <= 0.5 || gain <= 1e-18) {
                    continue;
                }
                if (r.group_bytes + db > r.budget_bytes + 0.5) {
                    continue;
                }
                double ratio = gain / db;
                if (best_cand == nullptr ||
                        ratio > best_ratio + 1e-24 ||
                        (std::abs(ratio - best_ratio) <= 1e-24 && gain > best_gain)) {
                    best_group = &name;
                    best_cand = &c;
                    best_ratio = ratio;
                    best_gain = gain;
                }
            }
        }
        if (best_cand == nullptr || best_group == nullptr) {
            break;
        }
        r.choices[*best_group] = *best_cand;
        finalize_recipe(r, p, rel_source_elements, stats);
        path.push_back(r);
    }
    return path;
}

static void add_recipe_unique(std::vector<recipe> & out, std::set<std::string> & seen, recipe r) {
    const std::string key = recipe_key(r);
    if (seen.insert(key).second) {
        out.push_back(std::move(r));
    }
}

static std::vector<recipe> local_neighbors(
        const recipe & base,
        const params & p,
        const std::vector<std::string> & order,
        const std::map<std::string, group_data> & groups,
        double rel_source_elements,
        const model_stats & stats) {
    std::vector<recipe> out;
    for (const auto & group_name : order) {
        auto cur_it = base.choices.find(group_name);
        if (cur_it == base.choices.end()) {
            continue;
        }
        for (const auto & c : groups.at(group_name).pruned) {
            if (c.qtype == cur_it->second.qtype) {
                continue;
            }
            recipe r = base;
            r.choices[group_name] = c;
            finalize_recipe(r, p, rel_source_elements, stats);
            if (!r.over_budget) {
                out.push_back(std::move(r));
            }
        }
    }
    return out;
}

static std::vector<recipe> topk_for_target(
        const std::string & solver,
        double target,
        const params & p,
        const std::vector<std::string> & order,
        const std::map<std::string, group_data> & groups,
        const std::vector<recipe> & lambda_pool,
        double rel_source_elements,
        const model_stats & stats) {
    std::vector<recipe> pool;
    std::set<std::string> seen;
    const double budget = budget_for_target(target, p, rel_source_elements, stats);

    auto consider = [&](recipe r) {
        r.solver = solver;
        r.target_bpw = target;
        r.budget_bytes = budget;
        finalize_recipe(r, p, rel_source_elements, stats);
        if (!r.over_budget) {
            add_recipe_unique(pool, seen, std::move(r));
        }
    };

    if (solver == "lambda") {
        for (const auto & r0 : lambda_pool) {
            consider(r0);
        }
        if (!pool.empty()) {
            std::sort(pool.begin(), pool.end(), [](const recipe & a, const recipe & b) {
                if (a.total_loss != b.total_loss) {
                    return a.total_loss < b.total_loss;
                }
                return a.group_bytes > b.group_bytes;
            });
            const recipe seed = pool.front();
            for (auto n : local_neighbors(seed, p, order, groups, rel_source_elements, stats)) {
                consider(std::move(n));
            }
        }
    } else {
        std::vector<recipe> path = solve_greedy_path(target, p, order, groups, rel_source_elements, stats);
        for (auto & r : path) {
            consider(std::move(r));
        }
        if (!path.empty()) {
            recipe seed = solve_greedy(target, p, order, groups, rel_source_elements, stats);
            for (auto n : local_neighbors(seed, p, order, groups, rel_source_elements, stats)) {
                consider(std::move(n));
            }
        }
    }

    std::sort(pool.begin(), pool.end(), [](const recipe & a, const recipe & b) {
        if (a.total_loss != b.total_loss) {
            return a.total_loss < b.total_loss;
        }
        if (a.group_bytes != b.group_bytes) {
            return a.group_bytes > b.group_bytes;
        }
        return recipe_key(a) < recipe_key(b);
    });
    if ((int) pool.size() > p.top_k) {
        pool.resize((size_t) p.top_k);
    }
    return pool;
}

static std::string bpw_tag(double v) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << v;
    std::string s = ss.str();
    std::replace(s.begin(), s.end(), '.', '_');
    return s;
}

static void write_recipe_json(const fs::path & path, const recipe & r, const params & p, bool diagnostic_only) {
    json j;
    j["name"] = "moq_" + r.solver + "_bpw_" + bpw_tag(r.target_bpw);
    j["target_bpw"] = r.target_bpw;
    j["estimated_bpw"] = r.estimated_bpw;
    j["relative_group_bpw"] = r.relative_group_bpw;
    j["absolute_model_bpw"] = r.absolute_model_bpw;
    j["solver"] = r.solver;
    j["loss"] = r.total_loss;
    j["predicted_ppl"] = r.predicted_ppl;
    j["predicted_mean_kld"] = r.predicted_mean_kld;
    j["predicted_p99_kld"] = r.predicted_p99_kld;
    j["predicted_p999_kld"] = r.predicted_p999_kld;
    j["predicted_max_kld"] = r.predicted_max_kld;
    j["group_bytes"] = (uint64_t) std::llround(r.group_bytes);
    j["covered_bytes"] = (uint64_t) std::llround(r.covered_bytes);
    j["total_bytes"] = (uint64_t) std::llround(r.total_bytes);
    j["fixed_bytes"] = r.fixed_bytes;
    j["bpw_mode"] = !p.model.empty() ? "absolute_model_gguf" : (p.has_total_params ? "absolute_total_params" : "relative_group_only");
    j["over_budget"] = r.over_budget;
    j["diagnostic_only"] = diagnostic_only;
    j["auto_tail_guard"] = p.auto_tail_guard;
    j["guard_metadata"] = {
        {"mode", p.auto_tail_guard ? "auto_tail_guard" : "none"},
        {"metric", p.tail_guard_metric},
        {"target_ratio", p.tail_guard_target_ratio},
        {"absolute_p999", p.tail_guard_absolute_p999},
        {"absolute_max", p.tail_guard_absolute_max},
        {"min_upgrade_gain", p.tail_guard_min_upgrade_gain},
    };
    j["groups"] = json::object();
    for (const auto & kv : r.choices) {
        j["groups"][kv.first] = kv.second.qtype;
    }
    std::ofstream out(path);
    out << std::setw(2) << j << "\n";
}

static void write_recipe_txt(const fs::path & path, const recipe & r, const params & p, bool diagnostic_only) {
    std::ofstream out(path);
    out << "MoQ recipe\n\n";
    out << "solver: " << r.solver << "\n";
    out << "target_bpw: " << r.target_bpw << "\n";
    out << "estimated_bpw: " << r.estimated_bpw << "\n";
    out << "relative_group_bpw: " << r.relative_group_bpw << "\n";
    out << "absolute_model_bpw: " << r.absolute_model_bpw << "\n";
    out << "bpw_mode: " << (!p.model.empty() ? "absolute_model_gguf" : (p.has_total_params ? "absolute_total_params" : "relative_group_only")) << "\n";
    out << "group_bytes: " << std::llround(r.group_bytes) << "\n";
    out << "covered_bytes: " << std::llround(r.covered_bytes) << "\n";
    out << "fixed_bytes: " << std::llround(r.fixed_bytes) << "\n";
    out << "total_bytes: " << std::llround(r.total_bytes) << "\n";
    out << "loss: " << r.total_loss << "\n";
    out << "predicted_ppl: " << r.predicted_ppl << "\n";
    out << "predicted_mean_kld: " << r.predicted_mean_kld << "\n";
    out << "predicted_max_kld: " << r.predicted_max_kld << "\n";
    out << "predicted_p999_kld: " << r.predicted_p999_kld << "\n";
    out << "over_budget: " << (r.over_budget ? "true" : "false") << "\n\n";
    out << "diagnostic_only: " << (diagnostic_only ? "true" : "false") << "\n\n";
    out << "auto_tail_guard: " << (p.auto_tail_guard ? "true" : "false") << "\n\n";
    out << "group,qtype,quant_bytes,bpw_delta,loss,mean_kld,p999_kld\n";
    for (const auto & kv : r.choices) {
        const auto & c = kv.second;
        out << kv.first << ',' << c.qtype << ','
            << std::llround(c.quant_bytes) << ','
            << c.bpw_delta << ','
            << c.loss << ','
            << c.mean_kld << ','
            << c.p999_kld << "\n";
    }
}

static int recipe_qtype_count(const recipe & r, const std::string & qtype) {
    int count = 0;
    for (const auto & kv : r.choices) {
        count += kv.second.qtype == qtype ? 1 : 0;
    }
    return count;
}

static ggml_type qtype_to_ggml_type(const std::string & qtype) {
    static const std::map<std::string, ggml_type> m = {
        {"Q2_K", GGML_TYPE_Q2_K},
        {"Q3_K", GGML_TYPE_Q3_K},
        {"Q4_K", GGML_TYPE_Q4_K},
        {"Q5_K", GGML_TYPE_Q5_K},
        {"Q6_K", GGML_TYPE_Q6_K},
        {"Q8_0", GGML_TYPE_Q8_0},
        {"IQ1_S", GGML_TYPE_IQ1_S},
        {"IQ1_M", GGML_TYPE_IQ1_M},
        {"IQ2_XXS", GGML_TYPE_IQ2_XXS},
        {"IQ2_XS", GGML_TYPE_IQ2_XS},
        {"IQ2_S", GGML_TYPE_IQ2_S},
        {"IQ3_XXS", GGML_TYPE_IQ3_XXS},
        {"IQ3_S", GGML_TYPE_IQ3_S},
        {"IQ4_NL", GGML_TYPE_IQ4_NL},
        {"IQ4_XS", GGML_TYPE_IQ4_XS},
        {"Q4_0", GGML_TYPE_Q4_0},
        {"Q4_1", GGML_TYPE_Q4_1},
        {"Q5_0", GGML_TYPE_Q5_0},
        {"Q5_1", GGML_TYPE_Q5_1},
        {"F16", GGML_TYPE_F16},
        {"BF16", GGML_TYPE_BF16},
        {"F32", GGML_TYPE_F32},
    };
    auto it = m.find(qtype);
    return it == m.end() ? GGML_TYPE_COUNT : it->second;
}

static int64_t tensor_quant_bytes(const tensor_stat & ts, const std::string & qtype) {
    const ggml_type type = qtype_to_ggml_type(qtype);
    if (type == GGML_TYPE_COUNT) {
        return 0;
    }
    if (ts.shape.empty() || ts.params <= 0.0) {
        return 0;
    }
    if (type == GGML_TYPE_F32) {
        return (int64_t) std::llround(ts.params * 4.0);
    }
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        return (int64_t) std::llround(ts.params * 2.0);
    }
    const int64_t ne0 = ts.shape[0];
    if (ne0 <= 0) {
        return 0;
    }
    const int64_t rows = (int64_t) std::llround(ts.params / (double) ne0);
    return (int64_t) ggml_row_size(type, ne0) * rows;
}

static void write_tensor_type_export(
        const fs::path & base_path,
        const recipe & r,
        const std::vector<group_def> & defs,
        const model_stats & stats) {
    std::map<std::string, const group_def *> group_lookup;
    for (const auto & g : defs) {
        group_lookup[g.name] = &g;
    }

    struct assignment {
        std::string tensor;
        std::string group;
        std::string qtype;
        int64_t source_bytes = 0;
        int64_t quant_bytes = 0;
    };

    std::vector<assignment> assignments;
    std::set<std::string> covered;
    std::set<std::string> duplicates;
    std::vector<std::string> missing;
    int64_t estimated_total_bytes = (int64_t) std::llround(r.fixed_bytes);

    for (const auto & choice : r.choices) {
        auto git = group_lookup.find(choice.first);
        if (git == group_lookup.end()) {
            missing.push_back("missing_group:" + choice.first);
            continue;
        }
        for (const auto & tensor_name : git->second->tensors) {
            if (!covered.insert(tensor_name).second) {
                duplicates.insert(tensor_name);
                continue;
            }
            auto tit = stats.tensors.find(tensor_name);
            if (tit == stats.tensors.end()) {
                missing.push_back(tensor_name);
                continue;
            }
            assignment a;
            a.tensor = tensor_name;
            a.group = choice.first;
            a.qtype = choice.second.qtype;
            a.source_bytes = (int64_t) std::llround(tit->second.params * 2.0);
            a.quant_bytes = tensor_quant_bytes(tit->second, choice.second.qtype);
            estimated_total_bytes += a.quant_bytes;
            assignments.push_back(std::move(a));
        }
    }

    {
        std::ofstream out(base_path.string() + ".tensor_types.csv");
        out << "tensor_name,group,qtype,source_bytes,quant_bytes\n";
        for (const auto & a : assignments) {
            out << '"' << a.tensor << '"' << ','
                << a.group << ','
                << a.qtype << ','
                << a.source_bytes << ','
                << a.quant_bytes << "\n";
        }
    }

    {
        json j;
        j["recipe"] = base_path.filename().string();
        j["absolute_model_bpw"] = r.absolute_model_bpw;
        j["relative_group_bpw"] = r.relative_group_bpw;
        j["tensors"] = json::object();
        for (const auto & a : assignments) {
            j["tensors"][a.tensor] = a.qtype;
        }
        std::ofstream out(base_path.string() + ".tensor_types.json");
        out << std::setw(2) << j << "\n";
    }

    {
        std::ofstream out(base_path.string() + ".tensor_types.txt");
        for (const auto & a : assignments) {
            out << a.tensor << ' ' << a.qtype << "\n";
        }
    }

    int fixed_tensors = 0;
    {
        std::ofstream out(base_path.string() + ".fixed_tensors.csv");
        out << "tensor_name,type,n_elements,bytes,reason\n";
        for (const auto & kv : stats.tensors) {
            if (covered.find(kv.first) != covered.end()) {
                continue;
            }
            fixed_tensors++;
            out << '"' << kv.first << '"' << ','
                << ggml_type_name(kv.second.type) << ','
                << (int64_t) std::llround(kv.second.params) << ','
                << (int64_t) std::llround(kv.second.bytes) << ','
                << "fixed_uncovered\n";
        }
    }

    {
        std::ofstream out(base_path.string() + ".export_summary.txt");
        out << "recipe: " << base_path.filename().string() << "\n";
        out << "covered_tensors: " << assignments.size() << "\n";
        out << "fixed_tensors: " << fixed_tensors << "\n";
        out << "duplicate_tensors: " << duplicates.size() << "\n";
        out << "missing_tensors: " << missing.size() << "\n";
        out << "estimated_total_bytes: " << estimated_total_bytes << "\n";
        out << "absolute_model_bpw: " << r.absolute_model_bpw << "\n";
        if (!duplicates.empty()) {
            out << "\nDuplicates:\n";
            for (const auto & t : duplicates) {
                out << "  " << t << "\n";
            }
        }
        if (!missing.empty()) {
            out << "\nMissing:\n";
            for (const auto & t : missing) {
                out << "  " << t << "\n";
            }
        }
    }
}

static void write_dominated(const fs::path & path, const std::map<std::string, group_data> & groups) {
    std::ofstream out(path);
    out << "group,qtype,dominated_by,quant_bytes,loss,mean_kld,p999_kld,bpw_delta,input_pareto_dominated\n";
    for (const auto & kv : groups) {
        for (const auto & c : kv.second.all) {
            if (!c.dominated) {
                continue;
            }
            out << c.group << ',' << c.qtype << ',' << c.dominated_by << ','
                << c.quant_bytes << ',' << c.loss << ',' << c.mean_kld << ','
                << c.p999_kld << ',' << c.bpw_delta << ','
                << (c.input_dominated ? "true" : "false") << "\n";
        }
    }
}

static void write_frontier(const fs::path & path, std::vector<recipe> recipes) {
    std::sort(recipes.begin(), recipes.end(), [](const recipe & a, const recipe & b) {
        if (a.group_bytes != b.group_bytes) {
            return a.group_bytes < b.group_bytes;
        }
        return a.total_loss < b.total_loss;
    });
    std::ofstream out(path);
    out << "source_solver,estimated_bpw,total_bytes,group_bytes,total_loss,num_groups,recipe_key,pareto_dominated\n";
    double best_loss = std::numeric_limits<double>::infinity();
    for (const auto & r : recipes) {
        bool dominated = r.total_loss >= best_loss - 1e-15;
        best_loss = std::min(best_loss, r.total_loss);
        out << r.solver << ',' << r.estimated_bpw << ','
            << std::llround(r.total_bytes) << ','
            << std::llround(r.group_bytes) << ','
            << r.total_loss << ',' << r.choices.size() << ",\""
            << recipe_key(r) << "\"," << (dominated ? "true" : "false") << "\n";
    }
}

static bool group_contains(const std::string & group, const std::string & needle) {
    return group.find(needle) != std::string::npos;
}

static bool low_bit(const candidate & c) {
    return c.bpw_delta > 0.0 && c.bpw_delta <= 4.0;
}

static void write_pattern_validation(
        const fs::path & path,
        const std::vector<recipe> & selected,
        const std::vector<std::string> & order,
        const std::map<std::string, group_data> & groups) {
    std::ofstream out(path);
    out << "MoQ solver pattern validation\n\n";

    std::map<std::string, candidate> smallest;
    for (const auto & name : order) {
        smallest[name] = groups.at(name).pruned.front();
    }

    out << "1. Earliest upgraded groups:\n";
    int listed = 0;
    for (const auto & name : order) {
        double first = std::numeric_limits<double>::infinity();
        std::string qtype;
        for (const auto & r : selected) {
            auto it = r.choices.find(name);
            if (it != r.choices.end() && it->second.quant_bytes > smallest[name].quant_bytes + 0.5) {
                if (r.target_bpw < first) {
                    first = r.target_bpw;
                    qtype = it->second.qtype;
                }
            }
        }
        if (std::isfinite(first) && listed++ < 20) {
            out << "  " << name << " first_upgrades_at_bpw=" << first << " qtype=" << qtype << "\n";
        }
    }
    if (listed == 0) {
        out << "  No upgrades observed within requested targets.\n";
    }

    out << "\n2. Groups still low precision at the lowest BPW:\n";
    if (!selected.empty()) {
        const recipe * low = &*std::min_element(selected.begin(), selected.end(), [](const recipe & a, const recipe & b) {
            return a.target_bpw < b.target_bpw;
        });
        int n = 0;
        for (const auto & kv : low->choices) {
            if (low_bit(kv.second) && n++ < 30) {
                out << "  " << kv.first << "=" << kv.second.qtype << " bpw=" << kv.second.bpw_delta << "\n";
            }
        }
    }

    out << "\n3. Q5_K attention/FFN sweet spot:\n";
    int q5_attn = 0, attn = 0, q5_ffn = 0, ffn = 0;
    for (const auto & r : selected) {
        for (const auto & kv : r.choices) {
            bool is_attn = group_contains(kv.first, "attn");
            bool is_ffn  = group_contains(kv.first, "ffn");
            if (is_attn) {
                attn++;
                q5_attn += kv.second.qtype == "Q5_K" ? 1 : 0;
            }
            if (is_ffn) {
                ffn++;
                q5_ffn += kv.second.qtype == "Q5_K" ? 1 : 0;
            }
        }
    }
    out << "  attention Q5_K selections: " << q5_attn << "/" << attn << "\n";
    out << "  FFN Q5_K selections: " << q5_ffn << "/" << ffn << "\n";

    out << "\n4. Q6_K/Q8_0 dominated counts:\n";
    std::map<std::string, std::pair<int, int>> dom;
    for (const auto & kv : groups) {
        for (const auto & c : kv.second.all) {
            if (c.qtype == "Q6_K" || c.qtype == "Q8_0") {
                dom[c.qtype].second++;
                dom[c.qtype].first += c.dominated ? 1 : 0;
            }
        }
    }
    for (const auto & kv : dom) {
        out << "  " << kv.first << " dominated=" << kv.second.first << "/" << kv.second.second << "\n";
    }

    out << "\n5. ssm_beta tail risk:\n";
    int ssm_low_tail = 0;
    int ssm_low = 0;
    for (const auto & kv : groups) {
        if (!group_contains(kv.first, "ssm_beta")) {
            continue;
        }
        double best_p999 = std::numeric_limits<double>::infinity();
        for (const auto & c : kv.second.all) {
            best_p999 = std::min(best_p999, c.p999_kld);
        }
        for (const auto & c : kv.second.all) {
            if (low_bit(c)) {
                ssm_low++;
                if (c.p999_kld > best_p999 * 2.0 && c.p999_kld > 1e-4) {
                    ssm_low_tail++;
                }
            }
        }
    }
    out << "  low-bit ssm_beta candidates with elevated p999 tail: " << ssm_low_tail << "/" << ssm_low << "\n";

    out << "\n6. Greedy vs lambda closeness:\n";
    std::map<double, const recipe *> lambda_by_target;
    std::map<double, const recipe *> greedy_by_target;
    for (const auto & r : selected) {
        if (r.solver == "lambda") {
            lambda_by_target[r.target_bpw] = &r;
        } else if (r.solver == "greedy") {
            greedy_by_target[r.target_bpw] = &r;
        }
    }
    for (const auto & kv : lambda_by_target) {
        auto it = greedy_by_target.find(kv.first);
        if (it == greedy_by_target.end()) {
            continue;
        }
        const auto * l = kv.second;
        const auto * g = it->second;
        int same = 0;
        for (const auto & ch : l->choices) {
            auto git = g->choices.find(ch.first);
            same += git != g->choices.end() && git->second.qtype == ch.second.qtype ? 1 : 0;
        }
        out << "  target_bpw=" << kv.first
            << " same_groups=" << same << "/" << l->choices.size()
            << " lambda_loss=" << l->total_loss
            << " greedy_loss=" << g->total_loss
            << " lambda_bpw=" << l->estimated_bpw
            << " greedy_bpw=" << g->estimated_bpw << "\n";
    }
}

int main(int argc, char ** argv) {
    try {
        params p = parse_args(argc, argv);
        fs::create_directories(p.out_dir);

        auto group_defs = read_group_defs(p.groups);
        auto order = group_order(group_defs);
        auto groups = read_elasticity(p);
        const group_overlap_report overlap = analyze_group_overlap(group_defs, groups);
        write_group_overlap_report(fs::path(p.out_dir) / "group_overlap_report.txt", overlap);

        if (overlap.has_overlap && !p.allow_overlap_diagnostic) {
            throw std::runtime_error("input groups overlap; see group_overlap_report.txt or use --allow-overlap-diagnostic");
        }
        if (!overlap.groups_missing_elasticity.empty()) {
            std::ostringstream ss;
            ss << "groups missing from elasticity table:";
            for (const auto & m : overlap.groups_missing_elasticity) {
                ss << ' ' << m;
            }
            throw std::runtime_error(ss.str());
        }
        const bool diagnostic_only = overlap.has_overlap;

        pareto_prune(groups);
        const tail_guard_result tail_guard = apply_auto_tail_guard(groups, p);
        write_dominated(fs::path(p.out_dir) / "dominated_candidates.csv", groups);
        write_auto_tail_guard_report(fs::path(p.out_dir) / "auto_tail_guard_report.csv", tail_guard);
        write_auto_tail_guard_json(fs::path(p.out_dir) / "auto_tail_guard.json", tail_guard, p);

        const double rel_source_elements = source_elements_total(order, groups);
        const model_stats stats = read_model_stats(p.model, group_defs, groups);
        auto lambda_pool = build_lambda_pool(p, order, groups, rel_source_elements, stats);
        std::vector<recipe> selected;
        std::vector<recipe> emitted_topk;
        std::vector<recipe> frontier_pool = lambda_pool;

        std::ofstream summary_csv(fs::path(p.out_dir) / "solver_summary.csv");
        summary_csv << "solver,target_bpw,estimated_bpw,relative_group_bpw,absolute_model_bpw,total_bytes,fixed_bytes,covered_bytes,"
            << "predicted_ppl,predicted_mean_kld,predicted_max_kld,predicted_p999_kld,total_loss,num_groups,q8_0_selected_count,recipe_file\n";
        std::ofstream topk_csv;
        if (p.emit_topk_recipes) {
            topk_csv.open(fs::path(p.out_dir) / "solver_topk_summary.csv");
            topk_csv << "solver,target_bpw,rank,estimated_bpw,relative_group_bpw,absolute_model_bpw,total_bytes,fixed_bytes,covered_bytes,"
                << "predicted_ppl,predicted_mean_kld,predicted_max_kld,predicted_p999_kld,total_loss,num_groups,q8_0_selected_count,recipe_file\n";
        }

        for (double target : p.target_bpw) {
            for (const auto & solver : p.solvers) {
                recipe r;
                if (solver == "lambda") {
                    r = choose_for_budget(lambda_pool, target, p, rel_source_elements, stats);
                    r.solver = "lambda";
                } else {
                    r = solve_greedy(target, p, order, groups, rel_source_elements, stats);
                }
                r.target_bpw = target;
                r.budget_bytes = budget_for_target(target, p, rel_source_elements, stats);
                finalize_recipe(r, p, rel_source_elements, stats);
                selected.push_back(r);
                frontier_pool.push_back(r);

                std::string base = "recipe_" + solver + "_bpw_" + bpw_tag(target);
                fs::path json_file = fs::path(p.out_dir) / (base + ".json");
                fs::path txt_file  = fs::path(p.out_dir) / (base + ".txt");
                write_recipe_json(json_file, r, p, diagnostic_only);
                write_recipe_txt(txt_file, r, p, diagnostic_only);
                if (p.export_tensor_types && stats.available) {
                    write_tensor_type_export(fs::path(p.out_dir) / base, r, group_defs, stats);
                }
                summary_csv << solver << ',' << target << ',' << r.estimated_bpw << ','
                    << r.relative_group_bpw << ',' << r.absolute_model_bpw << ','
                    << std::llround(r.total_bytes) << ','
                    << std::llround(r.fixed_bytes) << ','
                    << std::llround(r.covered_bytes) << ','
                    << r.predicted_ppl << ','
                    << r.predicted_mean_kld << ','
                    << r.predicted_max_kld << ','
                    << r.predicted_p999_kld << ','
                    << r.total_loss << ','
                    << r.choices.size() << ','
                    << recipe_qtype_count(r, "Q8_0") << ','
                    << json_file.filename().string() << "\n";

                if (p.emit_topk_recipes) {
                    std::vector<recipe> topk = topk_for_target(
                            solver, target, p, order, groups, lambda_pool, rel_source_elements, stats);
                    for (size_t ir = 0; ir < topk.size(); ++ir) {
                        std::ostringstream rank_ss;
                        rank_ss << std::setw(2) << std::setfill('0') << (ir + 1);
                        std::string top_base = "recipe_" + solver + "_bpw_" + bpw_tag(target) + "_rank_" + rank_ss.str();
                        fs::path top_json = fs::path(p.out_dir) / (top_base + ".json");
                        fs::path top_txt  = fs::path(p.out_dir) / (top_base + ".txt");
                        write_recipe_json(top_json, topk[ir], p, diagnostic_only);
                        write_recipe_txt(top_txt, topk[ir], p, diagnostic_only);
                        if (p.export_tensor_types && stats.available) {
                            write_tensor_type_export(fs::path(p.out_dir) / top_base, topk[ir], group_defs, stats);
                        }
                        emitted_topk.push_back(topk[ir]);
                        topk_csv << solver << ',' << target << ',' << (ir + 1) << ','
                            << topk[ir].estimated_bpw << ','
                            << topk[ir].relative_group_bpw << ','
                            << topk[ir].absolute_model_bpw << ','
                            << std::llround(topk[ir].total_bytes) << ','
                            << std::llround(topk[ir].fixed_bytes) << ','
                            << std::llround(topk[ir].covered_bytes) << ','
                            << topk[ir].predicted_ppl << ','
                            << topk[ir].predicted_mean_kld << ','
                            << topk[ir].predicted_max_kld << ','
                            << topk[ir].predicted_p999_kld << ','
                            << topk[ir].total_loss << ','
                            << topk[ir].choices.size() << ','
                            << recipe_qtype_count(topk[ir], "Q8_0") << ','
                            << top_json.filename().string() << "\n";
                    }
                }
            }
        }

        write_frontier(fs::path(p.out_dir) / "pareto_frontier.csv", frontier_pool);
        write_pattern_validation(fs::path(p.out_dir) / "moq_solver_pattern_validation.txt", selected, order, groups);

        {
            std::ofstream out(fs::path(p.out_dir) / "solver_summary.txt");
            out << "MoQ solver summary\n\n";
            out << "Elasticity: " << p.elasticity << "\n";
            out << "Groups: " << p.groups << "\n";
            out << "Groups solved: " << order.size() << "\n";
            size_t n_all = 0, n_pruned = 0, n_dom = 0;
            for (const auto & kv : groups) {
                n_all += kv.second.all.size();
                n_pruned += kv.second.pruned.size();
                for (const auto & c : kv.second.all) {
                    n_dom += c.dominated ? 1 : 0;
                }
            }
            out << "Candidates: " << n_all << "\n";
            out << "Candidates after Pareto prune: " << n_pruned << "\n";
            out << "Dominated candidates: " << n_dom << "\n";
            int q8_total = 0;
            int q8_dominated = 0;
            for (const auto & kv : groups) {
                for (const auto & c : kv.second.all) {
                    if (c.qtype == "Q8_0") {
                        q8_total++;
                        q8_dominated += c.dominated ? 1 : 0;
                    }
                }
            }
            int q8_selected = 0;
            std::map<double, int> q8_selected_by_bpw;
            for (const auto & r : selected) {
                const int n = recipe_qtype_count(r, "Q8_0");
                q8_selected += n;
                q8_selected_by_bpw[r.target_bpw] += n;
            }
            int q8_topk_selected = 0;
            std::map<double, int> q8_topk_selected_by_bpw;
            for (const auto & r : emitted_topk) {
                const int n = recipe_qtype_count(r, "Q8_0");
                q8_topk_selected += n;
                q8_topk_selected_by_bpw[r.target_bpw] += n;
            }
            out << "q8_0_dominated_count: " << q8_dominated << "/" << q8_total << "\n";
            out << "q8_0_selected_count: " << q8_selected << "\n";
            out << "q8_0_selected_count_topk: " << q8_topk_selected << "\n";
            out << "q8_0_selected_by_bpw:\n";
            for (const auto & kv : q8_selected_by_bpw) {
                out << "  " << kv.first << ": " << kv.second << "\n";
            }
            if (p.emit_topk_recipes) {
                out << "q8_0_selected_by_bpw_topk:\n";
                for (const auto & kv : q8_topk_selected_by_bpw) {
                    out << "  " << kv.first << ": " << kv.second << "\n";
                }
            }
            out << "Diagnostic only: " << (diagnostic_only ? "true" : "false") << "\n";
            out << "Overlapped tensors: " << overlap.overlapped_tensors.size() << "\n";
            out << "Auto tail guard: " << (p.auto_tail_guard ? "true" : "false") << "\n";
            out << "Auto tail guarded groups: " << tail_guard.groups.size() << "\n";
            if (p.auto_tail_guard) {
                int forbid_count = 0;
                for (const auto & kv : tail_guard.groups) {
                    forbid_count += (int) kv.second.forbid_qtypes.size();
                }
                out << "Auto tail forbidden candidates: " << forbid_count << "\n";
                out << "Auto tail guard report: auto_tail_guard_report.csv\n";
                out << "Auto tail guard json: auto_tail_guard.json\n";
            }
            out << "BPW mode: " << (stats.available ? "absolute_model_gguf" : (p.has_total_params ? "absolute_total_params" : "relative_group_only")) << "\n";
            if (diagnostic_only) {
                out << "Warning: input groups overlap; generated recipes are diagnostic_only and are not directly applicable.\n";
            }
            if (stats.available) {
                out << "Model: " << stats.path << "\n";
                out << "total_params: " << std::fixed << std::setprecision(0) << stats.total_params << "\n";
                out << "total_tensor_bytes: " << std::fixed << std::setprecision(0) << stats.total_tensor_bytes << "\n";
                out << "covered_params: " << std::fixed << std::setprecision(0) << stats.covered_params << "\n";
                out << "covered_source_bytes: " << std::fixed << std::setprecision(0) << stats.covered_source_bytes << "\n";
                out << "covered_model_bytes: " << std::fixed << std::setprecision(0) << stats.covered_model_bytes << "\n";
                out << "fixed_uncovered_bytes: " << std::fixed << std::setprecision(0) << stats.fixed_uncovered_bytes << "\n";
                if (!stats.missing_group_tensors.empty()) {
                    out << "Warning: group tensors missing from model table: " << stats.missing_group_tensors.size() << "\n";
                }
                out << std::defaultfloat << std::setprecision(6);
            } else if (!p.has_total_params) {
                out << "Note: --total-params was not provided; estimated_bpw is relative to the summed source elements of the listed groups, not whole-model BPW.\n";
            }
            if (!stats.available && !p.has_fixed_bytes) {
                out << "Note: --fixed-bytes was not provided; fixed non-swept model bytes are treated as zero.\n";
            }
            out << "Relative source elements: " << rel_source_elements << "\n";
            out << "Top-K recipes: " << (p.emit_topk_recipes ? p.top_k : 0) << "\n";
            out << "Loss: " << p.loss_mean_weight << "*mean_kld + "
                << p.loss_p999_weight << "*p999_kld + "
                << p.loss_p99_weight << "*p99_kld + "
                << p.loss_ppl_weight << "*ppl + "
                << p.loss_max_weight << "*max_kld\n\n";
            out << "Recipes are listed in solver_summary.csv; JSON and txt recipe files are in this directory.\n";
        }

        std::cout << "wrote MoQ solver outputs to " << p.out_dir << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
