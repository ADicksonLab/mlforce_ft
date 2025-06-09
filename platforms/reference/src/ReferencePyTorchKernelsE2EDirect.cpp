#include "ReferencePyTorchKernels.h"
#include "PyTorchForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include <cmath>
#include <assert.h>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

/**
 * @brief
 *
 * @param context
 * @return vector<Vec3>&
 */
static vector<Vec3>& extractPositions(ContextImpl& context) {
	ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
	return *((vector<Vec3>*) data->positions);
}

/**
 * @brief
 *
 * @param context
 * @return vector<Vec3>&
 */
static vector<Vec3>& extractForces(ContextImpl& context) {
	ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
	return *((vector<Vec3>*) data->forces);
}
/**
 * @brief
 *
 * @param context
 * @return Vec3*
 */
static Vec3* extractBoxVectors(ContextImpl& context) {
	ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
	return (Vec3*) data->periodicBoxVectors;
}

/**
 * @brief
 *
 * @param context
 * @return map<string, double>&
 */
static map<string, double>& extractEnergyParameterDerivatives(ContextImpl& context) {
	ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
	return *((map<string, double>*) data->energyParameterDerivatives);
}


/**
 * @brief
 *
 * @param context
 * @param numParticles
 * @return std::vector<double>
 */
static std::vector<double> extractContextVariables(ContextImpl& context, int numParticles) {
	std::vector<double> signals;
	string name;
	for (int i=0; i < numParticles; i++) {
		for (std::size_t j=0; j < PARAMETERNAMES.size(); j++) {
			signals.push_back(context.getParameter(PARAMETERNAMES[j]+std::to_string(i)));
		}
	}
	return signals;
}

ReferenceCalcPyTorchForceE2EDirectKernel::~ReferenceCalcPyTorchForceE2EDirectKernel() {
}


/**
 * @brief
 *
 * @param system
 * @param force
 * @param nnModule
 */
void ReferenceCalcPyTorchForceE2EDirectKernel::initialize(const System& system, const PyTorchForceE2EDirect& force, torch::jit::script::Module& nnModule) {
	this->nnModule = nnModule;
	nnModule.to(torch::kCPU);
	nnModule.eval();

	scale = force.getScale();
	particleIndices = force.getParticleIndices();
	signalForceWeights = force.getSignalForceWeights();
	vector<int> tmpAtomTypes = force.getAtomTypes();
	vector<vector<int>> tmpEdgeIdxs = force.getEdgeIndices();
	vector<int> tmpEdgeTypes = force.getEdgeTypes();
	useAttr = force.getUseAttr();
	usePeriodic = force.usesPeriodicBoundaryConditions();
	
	int n_edges = tmpEdgeTypes.size();
	int numGhostParticles = particleIndices.size();
	assert(tmpAtomTypes.size() == numGhostParticles);
	assert(tmpEdgeIdxs.size() == 2);
	assert(tmpEdgeIdxs[0].size() == n_edges);
	assert(tmpEdgeIdxs[1].size() == n_edges);

	options_float = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
	options_int = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);

	// define tensors used for model inputs
	torch::Tensor atom_types_tensor = torch::empty({static_cast<int64_t>(numGhostParticles)}, options_int);
	auto at_acc = atom_types_tensor.accessor<int64_t, 1>();
	
	//Copy data to the tensor
	for (int i = 0; i < numGhostParticles; i++) {
	  at_acc[i] = tmpAtomTypes[i];
	}

	torch::Tensor edge_idxs_tensor = torch::empty({2, static_cast<int64_t>(n_edges)}, options_int);
	auto edge_acc = edge_idxs_tensor.accessor<int64_t, 2>();
	torch::Tensor edge_types_tensor = torch::empty({static_cast<int64_t>(n_edges)}, options_int);
	auto et_acc = edge_types_tensor.accessor<int64_t, 1>();

	//Copy data to the tensors
	for (int i = 0; i < n_edges; i++) {
	  edge_acc[0][i] = tmpEdgeIdxs[0][i];
	  edge_acc[1][i] = tmpEdgeIdxs[1][i];
	  et_acc[i] = tmpEdgeTypes[i];
	}

	torch::Tensor batch = torch::zeros({numGhostParticles}, options_int);

	//                             |------------- fixedInputs ---------|
	// nnInputs = {[attr], pos, t, atomTypes, edgeIdxs, edgeTypes, batch}
	fixedInputs = {atom_types_tensor, edge_idxs_tensor, edge_types_tensor, batch};
	
	if (usePeriodic) {
	  int64_t boxVectorsDims[] = {3, 3};
	  boxVectorsTensor = torch::zeros(boxVectorsDims);
	  boxVectorsTensor = boxVectorsTensor.to(torch::kFloat32);
	}

}

/**
 * @brief
 *
 * @param context
 * @param includeForces
 * @param includeEnergy
 * @return double
 */
double ReferenceCalcPyTorchForceE2EDirectKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {


  double useGlobal = context.getParameter("use_global");
  torch::Tensor useGlobalTensor = torch::ones({1}, options_float) * useGlobal;

  // Get the  positions from the context (previous step)
	vector<Vec3>& MDPositions = extractPositions(context);
	vector<Vec3>& MDForce = extractForces(context);

	int numGhostParticles = particleIndices.size();
	
	torch::Tensor positionsTensor = torch::empty({numGhostParticles, 3}, options_float.requires_grad(true));

	auto positions = positionsTensor.accessor<float, 2>();
	//Copy positions to the tensor
	for (int i = 0; i < numGhostParticles; i++) {
	  positions[i][0] = MDPositions[particleIndices[i]][0]*10; 
	  positions[i][1] = MDPositions[particleIndices[i]][1]*10;
	  positions[i][2] = MDPositions[particleIndices[i]][2]*10;
	}

	torch::Tensor signalsTensor = torch::empty({numGhostParticles, 4}, options_float.requires_grad(true));
	
	vector<torch::jit::IValue> nnInputs = {};
	if (useAttr) {
	  // if using signals, pass them first
	  std::vector<double> globalVariables = extractContextVariables(context, numGhostParticles);

	  auto signals = signalsTensor.accessor<float, 2>();
	  for (int i = 0; i < numGhostParticles; i++) {
		for (int j = 0; j < 4; j++) {
		  signals[i][j] = globalVariables[4*i + j];
		}
	  }
	
	  nnInputs.push_back(signalsTensor);
	}

	torch::Tensor timTensor = torch::ones({1}, options_float);
	
	nnInputs.push_back(positionsTensor);
	nnInputs.push_back(timTensor);
	for ( auto &ten : fixedInputs ) {
	  nnInputs.push_back(ten);
	}
	nnInputs.push_back(useGlobalTensor);


	auto get_diffusion_noise = nnModule.get_method("get_diffusion_noise");

	//std::cout << "get_diffusion_noise device:" << get_diffusion_noise.device();
	torch::Tensor noise = scale*get_diffusion_noise(nnInputs).toTensor();

	
	// get forces on positions as before
	if (includeForces) {

	  auto NNForce = noise.accessor<float, 2>();
		
	  for (int i = 0; i < numGhostParticles; i++) {
		MDForce[particleIndices[i]][0] += double(NNForce[i][0]);
		MDForce[particleIndices[i]][1] += double(NNForce[i][1]);
		MDForce[particleIndices[i]][2] += double(NNForce[i][2]);
	  }

	  if (useAttr) {
		// update the global variables derivatives
		map<string, double>& energyParamDerivs = extractEnergyParameterDerivatives(context);

		for (int i = 0; i < numGhostParticles; i++) {
		  for (int j=0; j<4; j++) { 
			energyParamDerivs[PARAMETERNAMES[j]+std::to_string(i)] += double(NNForce[i][j+3])*signalForceWeights[j];
		  }
		}
		
	  }
	}
	return 0.0; // E2EDirect only updates forces, there is no energy
  }
