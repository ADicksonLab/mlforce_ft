/* -------------------------------------------------------------------------- *
 *                                 OpenMM-NN                                    *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "PyTorchForce.h"
#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>
#include <vector>

using namespace PyTorchPlugin;
using namespace OpenMM;
using std::vector;

extern "C" void registerPyTorchSerializationProxies();

void testSerialization() {
	// Create a Force.
  vector<vector<vector<double>>> features={{{1.13, 1.5}, {5.0, 2.3}}, {{6.6, 0.5}, {5.0, 2.1}, {1.0, 2.8}}};  // target 0 shape: (2x2), target 1 shape: (3x2)
  vector<int> pindices={0, 1, 2};
  vector<double> weights={10000, 10000, 10000, 10000};
  double scale = 10;
  int assignFreq = 1;
  vector<vector<vector<int>>> rest_idxs={{{0,1},{0,2}},{}};
  vector<vector<double>> rest_dists={{0.1,0.5},{}};
  double rmaxDelta = 0.5;
  double restraintK = 2.0;
  vector<int> init_assign={0,1,2};
  int init_target_idx=0;
  double lambda_penalty = 10.0;

  PyTorchForce force("graph.pb", features, pindices, weights, scale, assignFreq, rest_idxs, rest_dists, rmaxDelta, restraintK,
					 init_assign, init_target_idx, lambda_penalty);

  // Serialize and then deserialize it.
  std::stringstream buffer;
  XmlSerializer::serialize<PyTorchForce>(&force, "Force", buffer);
  PyTorchForce* copy = XmlSerializer::deserialize<PyTorchForce>(buffer);

  // Compare the two forces to see if they are identical.

  PyTorchForce& force2 = *copy;
  ASSERT_EQUAL(force.getFile(), force2.getFile());
  ASSERT_EQUAL(force.getParticleIndices()[0], force2.getParticleIndices()[0]);
  ASSERT_EQUAL(force.getParticleIndices()[1], force2.getParticleIndices()[1]);
  ASSERT_EQUAL(force.usesPeriodicBoundaryConditions(), force2.usesPeriodicBoundaryConditions());
  ASSERT_EQUAL(force.getForceGroup(), force2.getForceGroup());
  ASSERT_EQUAL(force.getSignalForceWeights()[0], force2.getSignalForceWeights()[0]);
  ASSERT_EQUAL(force.getSignalForceWeights()[1], force2.getSignalForceWeights()[1]);
  ASSERT_EQUAL(force.getTargetFeatures()[0][0][0], force2.getTargetFeatures()[0][0][0]);
  ASSERT_EQUAL(force.getTargetFeatures()[0][1][0], force2.getTargetFeatures()[0][1][0]);
  ASSERT_EQUAL(force.getTargetFeatures()[1][0][0], force2.getTargetFeatures()[1][0][0]);
  ASSERT_EQUAL(force.getTargetFeatures()[1][1][1], force2.getTargetFeatures()[1][1][1]);
  ASSERT_EQUAL(force.getRestraintData().first[0][1], force2.getRestraintData().first[0][1]);
  ASSERT_EQUAL(force.getRestraintData().first[1].size(), force2.getRestraintData().first[1].size());
  ASSERT_EQUAL(force.getRestraintData().second[0][1][0], force2.getRestraintData().second[0][1][0]);
  ASSERT_EQUAL(force.getRestraintData().second[0][1][1], force2.getRestraintData().second[0][1][1]);
  ASSERT_EQUAL(force.getRestraintData().second[1].size(), force2.getRestraintData().second[1].size());
  ASSERT_EQUAL(force.getRestraintParams().first, force2.getRestraintParams().first);
  ASSERT_EQUAL(force.getRestraintParams().second, force2.getRestraintParams().second);
  ASSERT_EQUAL(force.getInitialAssignment().first, force2.getInitialAssignment().first);
  ASSERT_EQUAL(force.getInitialAssignment().second[0], force2.getInitialAssignment().second[0]);
  ASSERT_EQUAL(force.getInitialAssignment().second[1], force2.getInitialAssignment().second[1]);
  ASSERT_EQUAL(force.getLambdaMismatchPenalty(), force2.getLambdaMismatchPenalty());

}

int main() {
	try {
	registerPyTorchSerializationProxies();
	testSerialization();
	}
	catch(const std::exception& e) {
	  std::cout << "exception: " << e.what() << std::endl;
	return 1;
	}
	std::cout << "Done" << std::endl;
	return 0;
}
