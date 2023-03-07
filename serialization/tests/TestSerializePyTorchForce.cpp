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

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

extern "C" void registerPyTorchSerializationProxies();

void testSerialization() {
	// Create a Force.
  	std::vector<vector<double>> features={{1.13, 1.5}, {5.0, 2.3}};
	std::vector<int> pindices={0, 1};
	std::vector<double> weights={0.1, 0.2};
	double scale = 10;
	int assignFreq = 1;
	std::vector<vector<int>> rest_idxs={{0,1}};
	std::vector<double> rest_dists={0.1};
	double rmaxDelta = 0.5;
	double restraintK = 1000;
	std::vector<int> init_assign={0,1};
	PyTorchForce force("graph.pb", features, pindices, weights, scale, assignFreq, rest_idxs, rest_dists, rmaxDelta, restraintK, init_assign);

	// Serialize and then deserialize it.

	stringstream buffer;
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
	ASSERT_EQUAL(force.getTargetFeatures()[0][0], force2.getTargetFeatures()[0][0]);
	ASSERT_EQUAL(force.getTargetFeatures()[0][1], force2.getTargetFeatures()[0][1]);
	ASSERT_EQUAL(force.getTargetFeatures()[1][0], force2.getTargetFeatures()[1][0]);
	ASSERT_EQUAL(force.getTargetFeatures()[1][1], force2.getTargetFeatures()[1][1]);
	ASSERT_EQUAL(force.getRestraintIndices()[0][0], force2.getRestraintIndices()[0][0]);
	ASSERT_EQUAL(force.getRestraintIndices()[0][1], force2.getRestraintIndices()[0][1]);
	ASSERT_EQUAL(force.getRestraintDistances()[0], force2.getRestraintDistances()[0]);
	ASSERT_EQUAL(force.getRestraintParams()[0], force2.getRestraintParams()[0]);
	ASSERT_EQUAL(force.getRestraintParams()[1], force2.getRestraintParams()[1]);
	ASSERT_EQUAL(force.getInitialAssignment()[0], force2.getInitialAssignment()[0]);
	ASSERT_EQUAL(force.getInitialAssignment()[1], force2.getInitialAssignment()[1]);

}

int main() {
	try {
	registerPyTorchSerializationProxies();
	testSerialization();
	}
	catch(const exception& e) {
	cout << "exception: " << e.what() << endl;
	return 1;
	}
	cout << "Done" << endl;
	return 0;
}
