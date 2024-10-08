#include "ReferencePyTorchKernels.h"
#include "PyTorchForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"

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


ReferenceCalcPyTorchForceE2EKernel::~ReferenceCalcPyTorchForceE2EKernel() {
}


/**
 * @brief
 *
 * @param system
 * @param force
 * @param nnModule
 */
void ReferenceCalcPyTorchForceE2EKernel::initialize(const System& system, const PyTorchForceE2E& force, torch::jit::script::Module& nnModule) {
	this->nnModule = nnModule;
	nnModule.to(torch::kCPU);
	nnModule.eval();

	scale = force.getScale();
	offset = force.getOffset();
	particleIndices = force.getParticleIndices();
	signalForceWeights = force.getSignalForceWeights();
	
	usePeriodic = force.usesPeriodicBoundaryConditions();
	int numGhostParticles = particleIndices.size();

	if (usePeriodic) {
	  int64_t boxVectorsDims[] = {3, 3};
	  boxVectorsTensor = torch::zeros(boxVectorsDims);
	  boxVectorsTensor = boxVectorsTensor.to(torch::kFloat32);
	}

	// prepare all-to-all edge tensors
	std::vector<vector<int64_t>> edges;
	for (int64_t i = 0; i < numGhostParticles; i++) {
	  for (int64_t j = 0; j < numGhostParticles; j++) {
		if (i != j) {
		  vector<int64_t> p = {i,j};
		  edges.push_back(p);
		}
	  }
	}
	int num_edges = edges.size();

	edge_idxs = torch::empty({2, static_cast<int64_t>(num_edges)},
							 torch::TensorOptions().dtype(torch::kInt64));
	auto edge_acc = edge_idxs.accessor<int64_t, 2>();

	//Copy indices to the tensor
	for (int i = 0; i < num_edges; i++) {
	  edge_acc[0][i] = edges[i][0];
	  edge_acc[1][i] = edges[i][1];
	}

	edge_attrs = torch::zeros({num_edges, 1}, torch::TensorOptions().dtype(torch::kFloat32));
	batch = torch::zeros({numGhostParticles}, torch::TensorOptions().dtype(torch::kInt64));

}

/**
 * @brief
 *
 * @param context
 * @param includeForces
 * @param includeEnergy
 * @return double
 */
double ReferenceCalcPyTorchForceE2EKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

	// Get the  positions from the context (previous step)
	vector<Vec3>& MDPositions = extractPositions(context);
	vector<Vec3>& MDForce = extractForces(context);

	int numGhostParticles = particleIndices.size();
	
	
	torch::Tensor positionsTensor = torch::empty({numGhostParticles, 3},
												 torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));


	auto positions = positionsTensor.accessor<float, 2>();

	//Copy positions to the tensor
	for (int i = 0; i < numGhostParticles; i++) {
		positions[i][0] = MDPositions[particleIndices[i]][0];
		positions[i][1] = MDPositions[particleIndices[i]][1];
		positions[i][2] = MDPositions[particleIndices[i]][2];
	}

	std::vector<double> globalVariables = extractContextVariables(context, numGhostParticles);

	torch::Tensor signalsTensor = torch::empty({numGhostParticles, 4},
												 torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));

	auto signals = signalsTensor.accessor<float, 2>();

	//Copy positions to the tensor
	for (int i = 0; i < numGhostParticles; i++) {
	  for (int j = 0; j < 4; j++) {
		signals[i][j] = globalVariables[4*i + j];
	  }
	}
	
	// Run the pytorch model and get the energy
	vector<torch::jit::IValue> nnInputs = {signalsTensor, positionsTensor, edge_idxs, edge_attrs, batch};

	// outputTensor : energy
	torch::Tensor outputTensor = scale*nnModule.forward(nnInputs).toTensor() + offset;
	

	// update the global variables derivatives
	map<string, double>& energyParamDerivs = extractEnergyParameterDerivatives(context);

	// get forces on positions as before
	if (includeForces) {
		outputTensor.backward();

		// check if positions have gradients
		auto forceTensor = torch::zeros_like(positionsTensor);
		auto signalDerivTensor = torch::zeros_like(signalsTensor);

		forceTensor = - positionsTensor.grad().clone().detach();
		signalDerivTensor = signalsTensor.grad().clone().detach();

		positionsTensor.grad().zero_();
		signalsTensor.grad().zero_();

		if (!(forceTensor.dtype() == torch::kFloat64))
			forceTensor = forceTensor.to(torch::kFloat64);

		if (!(signalDerivTensor.dtype() == torch::kFloat64))
			signalDerivTensor = signalDerivTensor.to(torch::kFloat64);

		auto NNForce = forceTensor.accessor<double, 2>();
		auto NNSignalDeriv = signalDerivTensor.accessor<double, 2>();

		for (int i = 0; i < numGhostParticles; i++) {
			MDForce[particleIndices[i]][0] += NNForce[i][0];
			MDForce[particleIndices[i]][1] += NNForce[i][1];
			MDForce[particleIndices[i]][2] += NNForce[i][2];

			for (int j=0; j<4; j++) { 
			  energyParamDerivs[PARAMETERNAMES[j]+std::to_string(i)] += NNSignalDeriv[i][j]*signalForceWeights[j];
			}
			
		}
	}
	return outputTensor.item<double>();
  }
