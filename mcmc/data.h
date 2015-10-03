#ifndef __MCMC_DATA_H__
#define __MCMC_DATA_H__

#include <memory>
#include <vector>

#include "mcmc/cuckoo.h"

namespace mcmc {

// Drop in mcmc::cuckoo set implementation
using namespace mcmc::cuckoo;

bool GetUniqueEdgesFromFile(const std::string& filename,
                            std::vector<Edge>* vals);

bool GenerateSetsFromEdges(const std::vector<Edge>& vals, double heldout_ratio,
                           std::unique_ptr<Set>* training,
                           std::unique_ptr<Set>* heldout);

bool GenerateSetsFromFile(const std::string& filename, double heldout_ratio,
                          std::unique_ptr<Set>* training,
                          std::unique_ptr<Set>* heldout);

}  // namespace mcmc

#endif  // __MCMC_DATA_H__
