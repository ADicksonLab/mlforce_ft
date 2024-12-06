/**
 * This tests the CUDA implementation of TorchMLForce.
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

void testForce() {
  // Create a random cloud of particles.

  const int numParticles = 19;
  System system;
  vector<Vec3> positions(numParticles);
  OpenMM_SFMT::SFMT sfmt;
  init_gen_rand(0, sfmt);
  for (int i = 0; i < numParticles; i++) {
	system.addParticle(1.0);
  }
  positions = {Vec3(-0.9441, -1.2828, -0.7263),
			   Vec3(-1.7267, -1.6928, -1.1089),
			   Vec3(-1.3489, -0.4082,  0.3210),
			   Vec3(-2.5852,  0.2813, -0.0262),
			   Vec3(-3.6063,  0.7968, -0.3789),
			   Vec3(-4.4977,  1.2764, -0.6551),
			   Vec3(-1.5339, -1.0026,  1.2359),
			   Vec3(-0.2444,  0.6217,  0.6019),
			   Vec3( 0.9788,  0.0310,  1.3028),
			   Vec3( 1.8110, -0.8513,  0.3748),
			   Vec3( 2.6200, -0.0674, -0.6114),
			   Vec3( 2.6793,  1.1275, -0.6919),
			   Vec3( 3.2028, -0.7124, -1.3020),
			   Vec3( 1.1619, -1.5222, -0.1924),
			   Vec3( 2.5060, -1.4750,  0.9441),
			   Vec3( 1.6041,  0.8539,  1.6548),
			   Vec3(0.6576, -0.5484,  2.1714),
			   Vec3(-0.6750,  1.4037,  1.2307),
			   Vec3(0.0583,  1.0726, -0.3434)};
  
  std::vector<int> pindices={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  std::vector<double> weights={1,1,1,1};
  double scale = 1.0;
  bool useAttr = false;

  torch::TensorOptions options_float = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
  torch::TensorOptions options_int = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);
	
  auto t_sample = torch::tensor({0.5}, options_float);
  //auto sigma_sample = torch::rand({5000}, options_float);
  auto atom_type_sample = torch::tensor({8, 1, 6, 6, 6, 1, 1, 6, 6, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1 }, options_int);
  auto edge_index_sample = torch::tensor({
										  { 0,  0,  1,  2,  2,  2,  2,  3,  3,  4,  4,  5,  6,  7,  7,  7,  7,  8,
											8,  8,  8,  9,  9,  9,  9, 10, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18},
										  { 1,  2,  0,  0,  3,  6,  7,  2,  4,  3,  5,  4,  2,  2,  8, 17, 18,  7,
											9, 15, 16,  8, 10, 13, 14,  9, 11, 12, 10, 10,  9,  9,  8,  8,  7, 7 }
	}, options_int);
  
  auto edge_type_sample = torch::tensor({1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
										 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1}, options_int);

  auto batch_sample = torch::tensor({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, options_int);

  std::vector<torch::Tensor> fixedInputs;
  fixedInputs.push_back(t_sample) ;
  //fixedInputs.push_back(sigma_sample) ;
  fixedInputs.push_back(atom_type_sample) ;
  fixedInputs.push_back(edge_index_sample) ;
  fixedInputs.push_back(edge_type_sample) ;
  fixedInputs.push_back(batch_sample) ;
	
  PyTorchForceE2EDirect* force = new PyTorchForceE2EDirect("tests/test_scriptE2EDirect.pt", pindices, weights, scale, fixedInputs, useAttr);
  system.addForce(force);

  CustomNonbondedForce* cnb_force = new CustomNonbondedForce("epsilon*(sigma/r)^12;sigma=0.5*(sigma1+sigma2);epsilon=sqrt(epsilon1*epsilon2)");
  cnb_force->addPerParticleParameter("sigma");
  cnb_force->addPerParticleParameter("epsilon");

  vector<double> param = {0.02, 2.0};
  for (int i = 0; i < numParticles; i++) {
	cnb_force->addParticle(param);
  }
  cnb_force->addGlobalParameter("diffTime", 0.5);
  
  system.addForce(cnb_force);
	
  // Compute the forces and energy.

  VerletIntegrator integ(1.0);
  Platform& platform = Platform::getPlatformByName("CUDA");
  Context context(system, integ, platform);
  context.setParameter("diffTime",0.5);
  context.setPositions(positions);
  State state = context.getState(State::Energy | State::Forces);
		
}


int main(int argc, char* argv[]) {
	try {
		registerPyTorchCudaKernelFactories();
		if (argc > 1)
			Platform::getPlatformByName("CUDA").setPropertyDefaultValue("Precision", string(argv[1]));
		testForce();
	}
	catch(const std::exception& e) {
		std::cout << "exception: " << e.what() << std::endl;
		return 1;
	}
	std::cout << "Done" << std::endl;
	return 0;
}
