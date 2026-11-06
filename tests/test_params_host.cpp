/// Parameter sets against FIPS 204 Tables 1 and 2, written out independently of params.h

#include <cstring>

#include "check.h"
#include "ppq/params.h"

namespace {

struct Expected {
    int id;
    const char *name;
    int k, l, eta, tau, lambda;
    std::int32_t gamma1, gamma2;
    int omega;
    std::int32_t beta;
    std::size_t pk, sk, sig;
    std::size_t ctilde, w1;
};

// FIPS 204 Tables 1 and 2; ctilde = lambda / 4, w1Encode = k * 32 * w1 bits
constexpr Expected TABLE[] = {
    {44, "ML-DSA-44", 4, 4, 2, 39, 128, 1 << 17, 95232, 80, 78, 1312, 2560, 2420, 32, 768},
    {65, "ML-DSA-65", 6, 5, 4, 49, 192, 1 << 19, 261888, 55, 196, 1952, 4032, 3309, 48, 768},
    {87, "ML-DSA-87", 8, 7, 2, 60, 256, 1 << 19, 261888, 75, 120, 2592, 4896, 4627, 64, 1024},
};

TEST_CASE(constants_match_fips204) {
    CHECK(ppq::Q == 8380417);
    CHECK(ppq::Q == (1 << 23) - (1 << 13) + 1);
    CHECK(ppq::N == 256);
    CHECK(ppq::D == 13);
    CHECK(ppq::T1_PACKED_BYTES == 320);
}

TEST_CASE(parameter_sets_match_tables_1_and_2) {
    CHECK(std::size(ppq::PARAM_SETS) == std::size(TABLE));

    for (const Expected &e : TABLE) {
        const ppq::ParamSet *p = ppq::find_param_set(e.id);
        CHECK(p != nullptr);
        if (p == nullptr) {
            continue;
        }

        CHECK(std::strcmp(p->name, e.name) == 0);
        CHECK(p->k == e.k);
        CHECK(p->l == e.l);
        CHECK(p->eta == e.eta);
        CHECK(p->tau == e.tau);
        CHECK(p->lambda == e.lambda);
        CHECK(p->gamma1 == e.gamma1);
        CHECK(p->gamma2 == e.gamma2);
        CHECK(p->omega == e.omega);
        CHECK(p->beta == e.beta);
        CHECK(p->pk_bytes == e.pk);
        CHECK(p->sk_bytes == e.sk);
        CHECK(p->sig_bytes == e.sig);
        CHECK(p->ctilde_bytes == e.ctilde);
        CHECK(p->w1_bytes == e.w1);
    }
}

TEST_CASE(signature_layout_adds_up) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        // sigEncode: c-tilde || l packed z polynomials || omega + k hint bytes
        CHECK(p.sig_bytes == p.ctilde_bytes + static_cast<std::size_t>(p.l) * p.z_poly_bytes +
                                 static_cast<std::size_t>(p.omega + p.k));
        CHECK(p.z_poly_bytes == ppq::N * static_cast<std::size_t>(p.z_bits) / 8);
        // z coefficients lie in (-gamma1, gamma1] and are packed as gamma1 - z
        CHECK((std::int32_t{1} << p.z_bits) == 2 * p.gamma1);
        // w1 coefficients lie in [0, (q - 1) / (2 gamma2) - 1]
        CHECK((1 << p.w1_bits) > (ppq::Q - 1) / (2 * p.gamma2) - 1);
        CHECK((1 << (p.w1_bits - 1)) <= (ppq::Q - 1) / (2 * p.gamma2) - 1);
        CHECK(p.k <= ppq::MAX_K && p.l <= ppq::MAX_L);
    }
}

TEST_CASE(unknown_ids_are_rejected) {
    CHECK(ppq::find_param_set(0) == nullptr);
    CHECK(ppq::find_param_set(2) == nullptr);
    CHECK(ppq::find_param_set(128) == nullptr);
}

TEST_CASE(bit_length_matches_definition) {
    CHECK(ppq::bit_length(0) == 0);
    CHECK(ppq::bit_length(1) == 1);
    CHECK(ppq::bit_length(4) == 3);
    CHECK(ppq::bit_length(43) == 6);
    CHECK(ppq::bit_length(15) == 4);
    CHECK(ppq::bit_length(0xFFFFFFFFu) == 32);
}

} // namespace

int main() { return check::run_all(); }
