#include "ppq/acvp.h"

#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "ppq/kv_vectors.h"

namespace ppq {

std::vector<AcvpWorkload> load_acvp_sigver(const std::string &path) {
    std::vector<AcvpWorkload> out;
    std::map<std::pair<int, Interface>, std::size_t> group_of;
    // Keyed by the pk bytes as a string (GCC 13 warns falsely on map<vector<uint8_t>>)
    std::vector<std::map<std::string, std::uint32_t>> key_of;

    for (const KvRecord &r : load_kv_records(path)) {
        const ParamSet *params = find_param_set(static_cast<int>(r.integer("params")));
        const auto iface = parse_interface(r.str("interface"));
        const auto verdict = parse_verdict(r.str("verdict"));
        if (params == nullptr || !iface || !verdict) {
            throw std::runtime_error(r.where() + ": bad params, interface or verdict");
        }

        const auto [it, added] = group_of.try_emplace({params->id, *iface}, out.size());
        if (added) {
            AcvpWorkload g;
            g.workload.params = *params;
            g.workload.iface = *iface;
            out.push_back(std::move(g));
            key_of.emplace_back();
        }
        AcvpWorkload &g = out[it->second];

        const std::vector<std::uint8_t> pk = r.bytes("pk");
        const auto [key, new_key] = key_of[it->second].try_emplace(
            std::string(pk.begin(), pk.end()), static_cast<std::uint32_t>(g.workload.key_count()));
        if (new_key) {
            g.workload.add_key(pk);
        }

        const std::vector<std::uint8_t> msg = r.bytes(*iface == Interface::Mu ? "mu" : "msg");
        const std::vector<std::uint8_t> ctx =
            *iface == Interface::External ? r.bytes("ctx") : std::vector<std::uint8_t>{};

        try {
            g.workload.add_signature(key->second, r.bytes("sig"), msg, ctx, *verdict);
        } catch (const std::invalid_argument &e) {
            throw std::runtime_error(r.where() + ": " + e.what());
        }
        g.tc_ids.push_back(r.integer("tcId"));
    }

    for (const AcvpWorkload &g : out) {
        g.workload.validate();
    }
    return out;
}

} // namespace ppq
