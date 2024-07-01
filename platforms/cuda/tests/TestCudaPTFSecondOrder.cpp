/**
 * This tests the second order restraints that can be incorporated into
 * a PyTorchForce object.  Specifically, it tests the following:
 * - passing an empty list should not change the energy or the forces, compared with a force strength of zero
 * - a restraint with particles at the potential energy minimum should not change the energy, or the forces
 * - a restraint with particles away from the minimum *should* change the energy, and the forces in a predictable way
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

extern "C" OPENMM_EXPORT void registerPyTorchCudaKernelFactories();

double TOL=1e-6;


CustomNonbondedForce* new_cnb_force() {
  CustomNonbondedForce* cnb_force = new CustomNonbondedForce("epsilon*(sigma/r)^12;sigma=0.5*(sigma1+sigma2);epsilon=sqrt(epsilon1*epsilon2)");
  cnb_force->addPerParticleParameter("sigma");
  cnb_force->addPerParticleParameter("epsilon");

  vector<double> param = {0.02, 2.0};
  for (int i = 0; i < 2; i++) {
	cnb_force->addParticle(param);
  }
  cnb_force->addGlobalParameter("charge_g0", -1.2);
  cnb_force->addGlobalParameter("charge_g1", 0.5);
  cnb_force->addGlobalParameter("sigma_g0", 0.023);
  cnb_force->addGlobalParameter("sigma_g1", 0.1);
  cnb_force->addGlobalParameter("epsilon_g0", 0.03);
  cnb_force->addGlobalParameter("epsilon_g1", 1.5);
  cnb_force->addGlobalParameter("lambda_g0", 1);
  cnb_force->addGlobalParameter("lambda_g1", 1);
  cnb_force->addGlobalParameter("assignment_g0", 0);
  cnb_force->addGlobalParameter("assignment_g1", 1);

  return cnb_force;
}

void testForceEmptyVsOff() {
	// Create a random cloud of particles.

	const int numParticles = 2;
	System system1, system2;
	vector<Vec3> positions(numParticles);
	OpenMM_SFMT::SFMT sfmt;
	init_gen_rand(0, sfmt);
	for (int i = 0; i < numParticles; i++) {
	  system1.addParticle(1.0);
	  system2.addParticle(1.0);
	  positions[i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*10;
	}
	std::vector<vector<double>> features(2, std::vector<double>(180));
	std::vector<int> pindices={0, 1};
	std::vector<double> sf_weights={10000,10000,10000,10000};
	double scale = 10.0;
	
	int assignFreq = 1;
	std::vector<std::vector<int>> rest_idxs1 {{0,1}};
	std::vector<std::vector<int>> rest_idxs2 {{}};
	std::vector<double> rest_dists1 {1.0};
	std::vector<double> rest_dists2 {};
	double rest_rmax_delta = 0.5;
	double rest_k1 = 0.0;
	double rest_k2 = 1000.0;
	std::vector<int> init_a={0,1};

	PyTorchForce* force1 = new PyTorchForce("tests/ani_model_cpu.pt", features, pindices, sf_weights, scale, assignFreq, rest_idxs1, rest_dists1, rest_rmax_delta, rest_k1, init_a);
	PyTorchForce* force2 = new PyTorchForce("tests/ani_model_cpu.pt", features, pindices, sf_weights, scale, assignFreq, rest_idxs2, rest_dists2, rest_rmax_delta, rest_k2, init_a);
	system1.addForce(force1);
	system2.addForce(force2);

	system1.addForce(new_cnb_force());
	system2.addForce(new_cnb_force());

	// Compute the forces and energy.

	VerletIntegrator integ1(1.0);
	VerletIntegrator integ2(1.0);
	Platform& platform1 = Platform::getPlatformByName("CUDA");
	Platform& platform2 = Platform::getPlatformByName("CUDA");
	Context context1(system1, integ1, platform1);
	Context context2(system2, integ2, platform2);
	context1.setPositions(positions);
	context2.setPositions(positions);
	State state1 = context1.getState(State::Energy | State::Forces);
	State state2 = context2.getState(State::Energy | State::Forces);

	ASSERT_EQUAL(state1.getPotentialEnergy(), state2.getPotentialEnergy());
	ASSERT_EQUAL_VEC(state1.getForces()[0], state2.getForces()[0],TOL);
	ASSERT_EQUAL_VEC(state1.getForces()[1], state2.getForces()[1],TOL);
}

void testForceOffVsMinimum() {
	// Create a random cloud of particles.

	const int numParticles = 2;
	System system1, system2;
	vector<Vec3> positions(numParticles);
	OpenMM_SFMT::SFMT sfmt;
	init_gen_rand(0, sfmt);
	for (int i = 0; i < numParticles; i++) {
	  system1.addParticle(1.0);
	  system2.addParticle(1.0);
	}
	positions[0] = Vec3(0.0,0.0,0.0);
	positions[1] = Vec3(0.0,0.0,0.5);
	
	std::vector<vector<double>> features(2, std::vector<double>(180));
	std::vector<int> pindices={0, 1};
	std::vector<double> sf_weights={10000,10000,10000,10000};
	double scale = 10.0;
	
	int assignFreq = 1;
	std::vector<std::vector<int>> rest_idxs1 {{0,1}};
	std::vector<std::vector<int>> rest_idxs2 {{0,1}};
	std::vector<double> rest_dists1 {1.0};
	std::vector<double> rest_dists2 {0.5};
	double rest_rmax_delta = 0.5;
	double rest_k1 = 0.0;
	double rest_k2 = 1000.0;
	std::vector<int> init_a={0,1};
	
	PyTorchForce* force1 = new PyTorchForce("tests/ani_model_cpu.pt", features, pindices, sf_weights, scale, assignFreq, rest_idxs1, rest_dists1, rest_rmax_delta, rest_k1, init_a);
	PyTorchForce* force2 = new PyTorchForce("tests/ani_model_cpu.pt", features, pindices, sf_weights, scale, assignFreq, rest_idxs2, rest_dists2, rest_rmax_delta, rest_k2, init_a);
	system1.addForce(force1);
	system2.addForce(force2);

	system1.addForce(new_cnb_force());
	system2.addForce(new_cnb_force());

	// Compute the forces and energy.

	VerletIntegrator integ1(1.0);
	VerletIntegrator integ2(1.0);
	Platform& platform1 = Platform::getPlatformByName("CUDA");
	Platform& platform2 = Platform::getPlatformByName("CUDA");
	Context context1(system1, integ1, platform1);
	Context context2(system2, integ2, platform2);
	context1.setPositions(positions);
	context2.setPositions(positions);
	State state1 = context1.getState(State::Energy | State::Forces);
	State state2 = context2.getState(State::Energy | State::Forces);

	ASSERT_EQUAL(state1.getPotentialEnergy(), state2.getPotentialEnergy());
	ASSERT_EQUAL_VEC(state1.getForces()[0], state2.getForces()[0],TOL);
	ASSERT_EQUAL_VEC(state1.getForces()[1], state2.getForces()[1],TOL);
}

void testRestEnergyAndForce() {
	// Create a random cloud of particles.

	const int numParticles = 2;
	System system1, system2;
	vector<Vec3> positions(numParticles);
	OpenMM_SFMT::SFMT sfmt;
	init_gen_rand(0, sfmt);
	for (int i = 0; i < numParticles; i++) {
	  system1.addParticle(1.0);
	  system2.addParticle(1.0);
	}
	positions[0] = Vec3(0.0,0.0,0.0);
	positions[1] = Vec3(0.0,0.0,1.5);
	
	std::vector<vector<double>> features(2, std::vector<double>(180));
	std::vector<int> pindices={0, 1};
	std::vector<double> sf_weights={10000,10000,10000,10000};
	double scale = 10.0;
	
	int assignFreq = -1;
	std::vector<std::vector<int>> rest_idxs1 {{0,1}};
	std::vector<std::vector<int>> rest_idxs2 {{0,1}};
	std::vector<double> rest_dists1 {1.0};
	std::vector<double> rest_dists2 {1.0};
	double rest_rmax_delta = 1.0;
	double rest_k1 = 0.0;    // system1 is off
	double rest_k2 = 1000.0; // system2 is on
	std::vector<int> init_a={0,1};
	
	PyTorchForce* force1 = new PyTorchForce("tests/ani_model_cpu.pt", features, pindices, sf_weights, scale, assignFreq, rest_idxs1, rest_dists1, rest_rmax_delta, rest_k1, init_a);
	PyTorchForce* force2 = new PyTorchForce("tests/ani_model_cpu.pt", features, pindices, sf_weights, scale, assignFreq, rest_idxs2, rest_dists2, rest_rmax_delta, rest_k2, init_a);
	system1.addForce(force1);
	system2.addForce(force2);

	system1.addForce(new_cnb_force());
	system2.addForce(new_cnb_force());

	// Compute the forces and energy.

	VerletIntegrator integ1(1.0);
	Platform& platform1 = Platform::getPlatformByName("CUDA");
	Context context1(system1, integ1, platform1);
	context1.setPositions(positions);
	State state1 = context1.getState(State::Energy | State::Forces);
	
	VerletIntegrator integ2(1.0);
	Platform& platform2 = Platform::getPlatformByName("CUDA");
	Context context2(system2, integ2, platform2);
	context2.setPositions(positions);
	State state2 = context2.getState(State::Energy | State::Forces);

	double tgt_energy = 0.5*scale*rest_k2*(0.5*0.5);
	ASSERT(state2.getPotentialEnergy() - state1.getPotentialEnergy() - tgt_energy < TOL);
	vector<Vec3> f1 = state1.getForces();
	vector<Vec3> f2 = state2.getForces();
	Vec3 p0_diff = f2[0]-f1[0];
	Vec3 p1_diff = f2[1]-f1[0];

	// x and y components of the forces on both particles should be zero
	ASSERT(abs(p0_diff[0]) < TOL);
	ASSERT(abs(p0_diff[1]) < TOL);
	ASSERT(abs(p1_diff[0]) < TOL);
	ASSERT(abs(p1_diff[1]) < TOL);

	// z component should be scale*rest_k2*(dz-r0)
	double tgt_force_z = scale*rest_k2*(0.5);
	ASSERT(abs(p0_diff[2] - tgt_force_z) < TOL);  // p0 force should be positive
	ASSERT(abs(p1_diff[2] + tgt_force_z) < TOL);  // p1 force should be negative
}

int main(int argc, char* argv[]) {
  try {
	registerPyTorchCudaKernelFactories();
	if (argc > 1)
	  Platform::getPlatformByName("CUDA").setPropertyDefaultValue("Precision", string(argv[1]));

	testForceEmptyVsOff(); // expensive and redundant with reference tests
	testForceOffVsMinimum(); // expensive and redundant with reference tests
	testRestEnergyAndForce();
	}
	catch(const std::exception& e) {
	std::cout << "exception: " << e.what() << std::endl;
	return 1;
	}
	std::cout << "Done" << std::endl;
	return 0;
}
