/**
 * This tests the Reference implementation of PyTorchForce.
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

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerPyTorchReferenceKernelFactories();

void testForce() {
	// Create a random cloud of particles.

	const int numParticles = 3;
	System system;
	vector<Vec3> positions(numParticles);
	OpenMM_SFMT::SFMT sfmt;
	init_gen_rand(0, sfmt);
	for (int i = 0; i < numParticles; i++) {
	  system.addParticle(1.0);
	  positions[i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*10;
	}
	std::vector<int> pindices={0, 1, 2};
	std::vector<double> weights={1,1,1,1};
	double scale = 3084.0973724679484;
	double offset = -1421.63869806774;
	
	PyTorchForceE2E* force = new PyTorchForceE2E("tests/test_script.pt", pindices, weights, scale, offset);
	system.addForce(force);

	CustomNonbondedForce* cnb_force = new CustomNonbondedForce("epsilon*(sigma/r)^12;sigma=0.5*(sigma1+sigma2);epsilon=sqrt(epsilon1*epsilon2)");
	cnb_force->addPerParticleParameter("sigma");
	cnb_force->addPerParticleParameter("epsilon");

	vector<double> param = {0.02, 2.0};
	for (int i = 0; i < numParticles; i++) {
	  cnb_force->addParticle(param);
	}
	cnb_force->addGlobalParameter("charge_g0", -1.2);
	cnb_force->addGlobalParameter("charge_g1", 0.5);
	cnb_force->addGlobalParameter("charge_g2", 0.51);
	cnb_force->addGlobalParameter("sigma_g0", 0.023);
	cnb_force->addGlobalParameter("sigma_g1", 0.1);
	cnb_force->addGlobalParameter("sigma_g2", 0.11);
	cnb_force->addGlobalParameter("epsilon_g0", 0.03);
	cnb_force->addGlobalParameter("epsilon_g1", 1.5);
	cnb_force->addGlobalParameter("epsilon_g2", 1.51);
	cnb_force->addGlobalParameter("lambda_g0", 1);
	cnb_force->addGlobalParameter("lambda_g1", 1);
	cnb_force->addGlobalParameter("lambda_g2", 1);

	system.addForce(cnb_force);

	// Compute the forces and energy.

	VerletIntegrator integ(1.0);
	Platform& platform = Platform::getPlatformByName("Reference");
	Context context(system, integ, platform);
	context.setPositions(positions);
	State state = context.getState(State::Energy | State::Forces);
		
}


int main() {
	try {
	registerPyTorchReferenceKernelFactories();
	testForce();
	}
	catch(const std::exception& e) {
	std::cout << "exception: " << e.what() << std::endl;
	return 1;
	}
	std::cout << "Done" << std::endl;
	return 0;
}