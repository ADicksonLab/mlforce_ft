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
