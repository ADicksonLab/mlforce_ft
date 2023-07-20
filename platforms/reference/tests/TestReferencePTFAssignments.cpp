/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
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
#include <algorithm>

using namespace PyTorchPlugin;
using namespace OpenMM;
using std::vector;
using std::map;
using std::string;

extern "C" OPENMM_EXPORT void registerPyTorchReferenceKernelFactories();

void testForceFixedAssignments() {
	// Create a random cloud of particles.

	const int numParticles = 10;
	System system;
	vector<Vec3> positions(numParticles);
	OpenMM_SFMT::SFMT sfmt;
	init_gen_rand(0, sfmt);
	vector<int> pindices;
	vector<int> init_a;

	for (int i = 0; i < numParticles; i++) {
	  system.addParticle(1.0);
	  positions[i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*10;
	  pindices.push_back(i);
	  init_a.push_back(i);
	}

	// Initialize target features as zero vectors
	vector<vector<vector<double>>> features(1, vector<vector<double>>(numParticles,vector<double>(180)));
	
	vector<double> sf_weights={10000,10000,10000,10000};
	double scale = 10.0;
	int assignFreq = -1;
	vector<vector<vector<int>>> rest_idxs {{{0,1}}};
	vector<vector<double>> rest_dists {{0.1}};
	double rest_rmax_delta = 0.5;
	double rest_k = 0;
	int targetIdx=0;
	double lambda_pen=10.0;
	
	PyTorchForce* force = new PyTorchForce("tests/ani_model_cpu.pt", features, pindices, sf_weights, scale, assignFreq, rest_idxs, rest_dists, rest_rmax_delta, rest_k, init_a, targetIdx, lambda_pen);
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
	cnb_force->addGlobalParameter("targetIdx",0);
	system.addForce(cnb_force);

	// Compute the forces and energy.

	VerletIntegrator integ(1.0);
	Platform& platform = Platform::getPlatformByName("Reference");
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

void fill_row(vector<double> & row)
{
  std::generate(row.begin(), row.end(), [](){ return double(rand())/RAND_MAX; }); 
}

void fill_matrix(vector<vector<double>> & mat)
{
  for_each(mat.begin(), mat.end(), fill_row);
}

void testForceVariableAssignments() {
	// Create a random cloud of particles.

	const int numParticles = 10;
	System system;
	vector<Vec3> positions(numParticles);
	OpenMM_SFMT::SFMT sfmt;
	init_gen_rand(0, sfmt);
	vector<int> pindices;
	vector<int> init_a;

	for (int i = 0; i < numParticles; i++) {
	  system.addParticle(1.0);
	  positions[i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*10;
	  pindices.push_back(i);
	  init_a.push_back(0);
	}

	// Initialize target features as zero vectors
	vector<vector<vector<double>>> features;
	vector<vector<double>> feat1(numParticles,vector<double>(180));
	fill_matrix(feat1);
	features.push_back(feat1);
	
	vector<double> sf_weights={10000,10000,10000,10000};
	double scale = 10.0;
	int assignFreq = 1;
	vector<vector<vector<int>>> rest_idxs {{{0,1}}};
	vector<vector<double>> rest_dists {{0.1}};
	double rest_rmax_delta = 0.5;
	double rest_k = 0;
	int targetIdx = 0;
	double lambda_pen= 0.0;
	
	PyTorchForce* force = new PyTorchForce("tests/ani_model_cpu.pt", features, pindices, sf_weights, scale, assignFreq, rest_idxs, rest_dists, rest_rmax_delta, rest_k, init_a, targetIdx, lambda_pen);
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
	cnb_force->addGlobalParameter("targetIdx",0);
	system.addForce(cnb_force);

	// Compute the forces and energy.

	VerletIntegrator integ(1.0);
	Platform& platform = Platform::getPlatformByName("Reference");
	Context context(system, integ, platform);
	context.setPositions(positions);
	State state = context.getState(State::Energy | State::Forces);

	// Read out the assignments, assert they are equal to init_a

	map<string,double> props = context.getParameters();
	vector<int> new_a;
	
	for (int i = 0; i < numParticles; i++) {
	  n = sprintf(varname, "assignment_g%d", i);
	  new_a.push_back(props[varname]);
	  ASSERT(props[varname] >= 0);
	  ASSERT(props[varname] < numParticles);
	  if (i > 0) {
		for (int j = 0; j < i; j++) {
		  ASSERT(props[varname] != new_a[j]);
		}
	  }
	}
}


int main() {
	try {
	registerPyTorchReferenceKernelFactories();
	testForceFixedAssignments();
	testForceVariableAssignments();
	}
	catch(const std::exception& e) {
	std::cout << "exception: " << e.what() << std::endl;
	return 1;
	}
	std::cout << "Done" << std::endl;
	return 0;
}
