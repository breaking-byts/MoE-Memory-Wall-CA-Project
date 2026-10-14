#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <algorithm>

struct TileConfig {
  int l1_tm;
  int l1_tn;
  int l1_tk;
  int l2_tm;
  int l2_tn;
  int l2_tk;
};

// Determine optimal L1/L2 tile sizes
inline TileConfig getOptimalTiles(double m, double n, double k,
                           int dbyte,
                           double l1_cache_size,
                           double l2_cache_size) {
    std::vector<int> trange = {8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512};
    int l1_tk = 32;
    double min_cost = std::numeric_limits<double>::max();
    int opt_l1_tm = trange[0], opt_l1_tn = trange[0];

    // L1 tile search
    for (int tm : trange) {
        for (int tn : trange) {
            int tm_eff = std::min<int>(m, tm);
            int tn_eff = std::min<int>(n, tn);
            double required = (double)(tm_eff + tn_eff) * l1_tk * dbyte
                              + (double)tm_eff * tn_eff * dbyte;
            if (required > l1_cache_size) continue;
            // L2 access estimate
            double l2_access = (std::ceil(n / (double)tn_eff) * m * k
                              + std::ceil(m / (double)tm_eff) * n * k
                              + m * n) * dbyte;
            double cost = l2_access;
            if (cost < min_cost) {
                min_cost = cost;
                opt_l1_tm = tm_eff;
                opt_l1_tn = tn_eff;
            }
        }
    }

    // L2 tile search
    int l2_tk = std::max(1, (int)(k / 64));
    double min_access = std::numeric_limits<double>::max();
    int opt_l2_tm = opt_l1_tm, opt_l2_tn = opt_l1_tn;

    std::vector<int> tm2_cands;
    int max_i = std::max<int>(1, (int)(m / opt_l1_tm));
    for (int i = 1; i <= max_i; ++i) {
        int tm2 = std::min<int>(m, opt_l1_tm * i);
        tm2_cands.push_back(tm2);
    }
    tm2_cands.push_back((int)m);

    std::vector<int> tn2_cands;
    int max_j = std::max<int>(1, (int)(n / opt_l1_tn));
    for (int j = 1; j <= max_j; ++j) {
        int tn2 = std::min<int>((int)n, opt_l1_tn * j);
        tn2_cands.push_back(tn2);
    }
    tn2_cands.push_back(n);

    for (int tm2 : tm2_cands) {
        for (int tn2 : tn2_cands) {
            double required2 = (double)(tm2 + tn2) * l2_tk * dbyte + (double)tm2 * tn2 * dbyte;
            if (required2 > l2_cache_size && !(tm2 == opt_l1_tm && tn2 == opt_l1_tn)) continue;
            double access = (std::ceil(m / (double)tm2) * n * k
                           + std::ceil(n / (double)tn2) * m * k
                           + m * n) * dbyte;
            if (access < min_access) {
                min_access = access;
                opt_l2_tm = tm2;
                opt_l2_tn = tn2;
            }
        }
    }

    return {opt_l1_tm, opt_l1_tn, l1_tk, opt_l2_tm, opt_l2_tn, l2_tk};
}

// Compute DRAM, L2, and L1 traffic (in bytes)
inline std::array<double, 3> computeCacheTraffic(double m, double n, double k,
                                          int dbyte,
                                          const TileConfig &tiles) {
    // DRAM traffic using L2 tile
    double dram = (std::ceil(m / (double)tiles.l2_tm) * n * k
                 + std::ceil(n / (double)tiles.l2_tn) * m * k
                 + m * n) * dbyte;
    // L2 traffic using L1 tile
    double l2  = (std::ceil(m / (double)tiles.l1_tm) * n * k
                 + std::ceil(n / (double)tiles.l1_tn) * m * k
                 + m * n) * dbyte;
    // L1 traffic using small register tiles
    const int reg_tm = 16, reg_tn = 16;
    double l1 = (std::ceil(m / (double)reg_tm) * n * k
               + std::ceil(n / (double)reg_tn) * m * k
               + m * n) * dbyte;
    return {dram, l2, l1};
}