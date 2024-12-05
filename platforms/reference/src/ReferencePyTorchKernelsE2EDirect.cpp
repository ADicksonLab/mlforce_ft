#include "ReferencePyTorchKernels.h"
#include "PyTorchForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include <cmath>

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
	vector<torch::Tensor> tmpFixedInputs = force.getFixedInputs();
	useAttr = force.getUseAttr();

	beta_start = 1.0e-7;
	beta_end = 2.0e-3;
	
	// 	Assume:
	// fixedInputs[0] : atomtype (int, N)
	// fixedInputs[1] : edgeIndex (int, 2, Ne)
	// fixedInputs[2] : edgeType (int, Ne)
	// fixedInputs[3] : batch (int, N)

	// nnInputs = {[attr], pos, t, *fixedInputs}

	
	usePeriodic = force.usesPeriodicBoundaryConditions();
	int numGhostParticles = particleIndices.size();

	if (usePeriodic) {
	  int64_t boxVectorsDims[] = {3, 3};
	  boxVectorsTensor = torch::zeros(boxVectorsDims);
	  boxVectorsTensor = boxVectorsTensor.to(torch::kFloat32);
	}

	options_float = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
	options_int = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);

	fixedInputs = {};
	fixedInputs.push_back(tmpFixedInputs[0].to(options_float));	
	fixedInputs.push_back(tmpFixedInputs[1].to(options_int));
	fixedInputs.push_back(tmpFixedInputs[2].to(options_int));
	fixedInputs.push_back(tmpFixedInputs[3].to(options_int));
	fixedInputs.push_back(tmpFixedInputs[4].to(options_int));

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

  // Get the current diffusion time from the context
  double tim = context.getParameter("diffTime");
  double beta = beta_start + (beta_end-beta_start)/(1.0 + exp(-tim));

  // Get the  positions from the context (previous step)
	vector<Vec3>& MDPositions = extractPositions(context);
	vector<Vec3>& MDForce = extractForces(context);

	int numGhostParticles = particleIndices.size();
	
	torch::Tensor positionsTensor = torch::empty({numGhostParticles, 3}, options_float.requires_grad(true));

	auto positions = positionsTensor.accessor<float, 2>();
	//Copy positions to the tensor
	for (int i = 0; i < numGhostParticles; i++) {
		positions[i][0] = MDPositions[particleIndices[i]][0];
		positions[i][1] = MDPositions[particleIndices[i]][1];
		positions[i][2] = MDPositions[particleIndices[i]][2];
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
	
	nnInputs.push_back(positionsTensor);
	for ( auto &ten : fixedInputs ) {
	  nnInputs.push_back(ten);
	}

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
