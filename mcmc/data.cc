#include "mcmc/data.h"

#include <cmath>
#include <fstream>

#include <glog/logging.h>

namespace mcmc {

bool GetUniqueEdgesFromFile(const std::string& filename,
                            std::vector<Edge>* vals) {
  std::ifstream in(filename);
  std::string line;
  // skip first 4 lines
  for (int i = 0; i < 4; ++i) std::getline(in, line);
  do {
    uint64_t a, b, x, y;
    in >> a >> b;
    if (!in.eof()) {
      x = std::min(a, b);
      y = std::max(a, b);
      vals->push_back((x << 32) | y);
    }
  } while (in.good());
  if (in.bad()) {
    LOG(ERROR) << "Error reading file " << filename;
    return false;
  }
  std::sort(vals->begin(), vals->end());
  // squeeze out duplicates
  auto end = std::unique(vals->begin(), vals->end());
  vals->resize(end - vals->begin());
  // shuffle again
  std::random_shuffle(vals->begin(), vals->end());
  return true;
}

bool GenerateSetsFromEdges(const std::vector<Edge>& vals, double heldout_ratio,
                           std::unique_ptr<Set>* training,
                           std::unique_ptr<Set>* heldout) {
  size_t training_len =
      static_cast<size_t>(std::ceil((1 - heldout_ratio) * vals.size()));
  size_t heldout_len = vals.size() - training_len;
  if (heldout_len > 0) {
    heldout->reset(new Set(heldout_len));
    for (auto it = vals.begin(); it != vals.begin() + heldout_len; ++it) {
      if (!(*heldout)->Insert(*it)) {
        LOG(ERROR) << "Failed to insert into heldout set";
        heldout->reset();
        return false;
      }
    }
  }
  training->reset(new Set(training_len));
  for (auto it = vals.begin() + heldout_len; it != vals.end(); ++it) {
    if (!(*training)->Insert(*it)) {
      LOG(ERROR) << "Failed to insert into training set";
      training->reset();
      if (heldout_len > 0) {
        heldout->reset();
      }
      return false;
    }
  }
  return true;
}

bool GenerateSetsFromFile(const std::string& filename, double heldout_ratio,
                          std::unique_ptr<Set>* training,
                          std::unique_ptr<Set>* heldout) {
  LOG(INFO) << "Going to generate sets from " << filename
            << " with held-out ratio " << heldout_ratio;
  std::vector<Edge> vals;
  if (!GetUniqueEdgesFromFile(filename, &vals)) return false;
  if (!GenerateSetsFromEdges(vals, heldout_ratio, training, heldout))
    return false;
  return true;
}

}  // namespace mcmc
