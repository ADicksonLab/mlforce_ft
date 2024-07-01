/**
 * This tests the ability of the Cuda implementation of PyTorchForce to
 * accept an initial set of assignments, and to control the assignment
 * updates with the AssignFreq parameter.
 */

#include "PyTorchForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/CustomNonbondedForce.h"
#include "sfmt/SFMT.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerPyTorchCudaKernelFactories();

void testForceFixedAssignments() {
	// Create a random cloud of particles.

	const int numParticles = 10;
	System system;
	vector<Vec3> positions(numParticles);
	OpenMM_SFMT::SFMT sfmt;
	init_gen_rand(0, sfmt);
	std::vector<int> pindices;
	std::vector<int> init_a;

	for (int i = 0; i < numParticles; i++) {
	  system.addParticle(1.0);
	  positions[i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*10;
	  pindices.push_back(i);
	  init_a.push_back(i);
	}

	// Initialize target features as zero vectors
	std::vector<vector<double>> features(numParticles, std::vector<double>(180));
	
	std::vector<double> sf_weights={10000,10000,10000,10000};
	double scale = 10.0;
	int assignFreq = 1;
	std::vector<std::vector<int>> rest_idxs {{0,1}};
	std::vector<double> rest_dists {0.1};
	double rest_rmax_delta = 0.5;
	double rest_k = 0;
	
	PyTorchForce* force = new PyTorchForce("tests/ani_model_cpu.pt", features, pindices, sf_weights, scale, assignFreq, rest_idxs, rest_dists, rest_rmax_delta, rest_k, init_a);
	system.addForce(force);

	CustomNonbondedForce* cnb_force = new CustomNonbondedForce("epsilon*(sigma/r)^12;sigma=0.5*(sigma1+sigma2);epsilon=sqrt(epsilon1*epsilon2)");
	cnb_force->addPerParticleParameter("sigma");
	cnb_force->addPerParticleParameter("epsilon");

	vector<double> param = {0.02, 2.0};
	char varname[50];
	int n;
	for (int i = 0; i < numParticles; i++) {
	  cnb_force->addParticle(param);
	  
	  n = sprintf(varname, "charge_g%d", i);
	  cnb_force->addGlobalParameter(varname, -1.2);

	  n = sprintf(varname, "sigma_g%d", i);
	  cnb_force->addGlobalParameter(varname, 0.023);

	  n = sprintf(varname, "epsilon_g%d", i);
	  cnb_force->addGlobalParameter(varname, 1.0);

	  n = sprintf(varname, "lambda_g%d", i);
	  cnb_force->addGlobalParameter(varname, 1.0);

	  n = sprintf(varname, "assignment_g%d", i);
	  cnb_force->addGlobalParameter(varname, -1);

	}
	system.addForce(cnb_force);

	// Compute the forces and energy.

	VerletIntegrator integ(1.0);
	Platform& platform = Platform::getPlatformByName("CUDA");
	Context context(system, integ, platform);
	context.setPositions(positions);
	State state = context.getState(State::Energy | State::Forces);

	// Read out the assignments, assert they are equal to init_a

	map<string,double> props = context.getParameters();
	for (int i = 0; i < numParticles; i++) {
	  n = sprintf(varname, "assignment_g%d", i);
	  ASSERT_EQUAL(props[varname], init_a[i]);
	}
	
}

int main(int argc, char* argv[]) {
	try {
	registerPyTorchCudaKernelFactories();
	if (argc > 1)
	  Platform::getPlatformByName("CUDA").setPropertyDefaultValue("Precision", string(argv[1]));

	testForceFixedAssignments();

	}
	catch(const std::exception& e) {
	std::cout << "exception: " << e.what() << std::endl;
	return 1;
	}
	std::cout << "Done" << std::endl;
	return 0;
}
