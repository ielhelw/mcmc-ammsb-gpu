#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <thread>
#include <omp.h>
#include "cuckoo.h"

using namespace std;

void test(const uint32_t N, const std::vector<uint32_t>& vals, uint32_t slots) {
  cout << "slots = " << slots << endl;
  mcmc::CuckooSet h(vals.size(), slots);
  auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
  for (size_t i = 0; i < vals.size(); ++i) {
    if (!h.Insert(vals[i])) {
      cout << "Failed to insert" << endl;
      abort();
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  cout << "OK " << (t2-t1).count()
#ifndef NDEBUG
    << ", max_displacements=" << h.displacements_max()
    << ", displacements_count=" << h.displacements_count()
    << ", displacements_avg=" << h.displacements_avg()
#endif
    << endl;
  for (auto it = vals.begin(); it != vals.end(); ++it) {
    if (!h.Has(*it)) {
      cout << "Failed to find: " << *it << endl;
      abort();
    }
  }
}

int main() {
  cout << omp_get_num_threads() << ", " << omp_get_max_threads() << endl;
  const uint32_t N = static_cast<uint32_t>(65e6);
  std::vector<uint32_t> vals(N);
  {
    std::unordered_set<uint32_t> set(N);
    for (uint32_t i = 0; i < N; ++i) {
      uint32_t k;
      while (set.find(k = rand()) != set.end()) {}
      set.insert(k);
      vals[i] = k;
    }
  }
  omp_set_num_threads(16);
  test(N, vals, 4);
  test(N, vals, 8);
  test(N, vals, 16);
  omp_set_num_threads(8);
  test(N, vals, 4);
  test(N, vals, 8);
  test(N, vals, 16);
  omp_set_num_threads(4);
  test(N, vals, 4);
  test(N, vals, 8);
  test(N, vals, 16);

  cout << "DONE" << endl;
  return 0;
}

