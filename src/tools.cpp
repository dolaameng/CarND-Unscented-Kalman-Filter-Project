#include <iostream>
#include <cassert>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  assert(estimations.size() == ground_truth.size());
  auto n_vectors = estimations.size();
  assert(n_vectors > 0);
  assert(estimations[0].size() == ground_truth[0].size());
  auto dim_vector = estimations[0].size();

  VectorXd rmse(dim_vector); rmse.fill(0);
  for (auto i = 0; i < n_vectors; ++i) {
  	VectorXd dist_sqr = (estimations[i] - ground_truth[i]).array().square();
  	rmse += dist_sqr;
  }
  rmse = (rmse / n_vectors).array().sqrt();
  return rmse;
}
