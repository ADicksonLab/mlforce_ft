#include "ReferencePyTorchKernels.h"
#include "PyTorchForce.h"
#include "Hungarian.h"
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


/**
 * @brief
 *
 * @param ptr
 * @param nRows
 * @param nCols
 * @return std::vector<std::vector<double> >
 */
static std::vector<std::vector<double> > tensorTo2DVec(double* ptr, int nRows, int nCols) {
	std::vector<std::vector<double> > distMat(nRows, std::vector<double>(nCols));
	for (int i=0; i<nRows; i++) {
		std::vector<double> vec(ptr+nCols*i, ptr+nRows*(i+1));
		distMat[i] = vec;
	}
	return distMat;
}

/**
 * @brief
 *
 * @param assignment
 * @return std::vector<int>
 */
static std::vector<int> getReverseAssignment(std::vector<int> assignment) {
	int n = assignment.size();
	std::vector<int> rev_assignment(n, -1);
	for (int i=0; i<n; i++) {
		if ((assignment[i] >= 0) && (assignment[i] < n)) {
		  rev_assignment[assignment[i]] = i;
		}
	}
	return rev_assignment;
}


ReferenceCalcPyTorchForceKernel::~ReferenceCalcPyTorchForceKernel() {
}


/**
 * @brief
 *
 * @param system
 * @param force
 * @param nnModule
 */
void ReferenceCalcPyTorchForceKernel::initialize(const System& system, const PyTorchForce& force, torch::jit::script::Module& nnModule) {
	this->nnModule = nnModule;
	nnModule.to(torch::kCPU);
	nnModule.eval();

	scale = force.getScale();
	assignFreq = force.getAssignFreq();
	particleIndices = force.getParticleIndices();
	signalForceWeights = force.getSignalForceWeights();
	targetRestraintIndices = force.getRestraintIndices();
	targetRestraintDistances = force.getRestraintDistances();
	targetRestraintParams = force.getRestraintParams();
	rmax_delta = targetRestraintParams[0];
	restraint_k = targetRestraintParams[1];

	assignment = force.getInitialAssignment();
	reverse_assignment = getReverseAssignment(assignment);
	
	numRestraints = targetRestraintDistances.size();
	for (int i = 0; i < numRestraints; i++) {
		r0sq.push_back(targetRestraintDistances[i]*targetRestraintDistances[i]);
		rmax.push_back(targetRestraintDistances[i] + rmax_delta);
		restraint_b.push_back(0.5*restraint_k*(r0sq[i] - rmax[i]*rmax[i]));
	}

	usePeriodic = force.usesPeriodicBoundaryConditions();
	int numGhostParticles = particleIndices.size();

	//get target features
	targetFeatures = force.getTargetFeatures();
	targetFeaturesTensor = torch::zeros({static_cast<int64_t>(targetFeatures.size()),
		static_cast<int64_t>(targetFeatures[0].size())},
		torch::TensorOptions().dtype(torch::kFloat64));

	for (std::size_t i = 0; i < targetFeatures.size(); i++)
		targetFeaturesTensor.slice(0, i, i+1) = torch::from_blob(targetFeatures[i].data(),
			{(long long)targetFeatures[0].size()},
			torch::TensorOptions().dtype(torch::kFloat64));

	if (usePeriodic) {
	  int64_t boxVectorsDims[] = {3, 3};
	  boxVectorsTensor = torch::zeros(boxVectorsDims);
	  boxVectorsTensor = boxVectorsTensor.to(torch::kFloat64);
	}

	step_count = 0;
}

/**
 * @brief
 *
 * @param context
 * @param includeForces
 * @param includeEnergy
 * @return double
 */
double ReferenceCalcPyTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

	// Get the  positions from the context (previous step)
	vector<Vec3>& MDPositions = extractPositions(context);
	vector<Vec3>& MDForce = extractForces(context);

	int numGhostParticles = particleIndices.size();
	
	
	torch::Tensor positionsTensor = torch::empty({numGhostParticles, 3},
		torch::TensorOptions().requires_grad(true).dtype(torch::kFloat64));


	auto positions = positionsTensor.accessor<double, 2>();

	//Copy positions to the tensor
	for (int i = 0; i < numGhostParticles; i++) {
		positions[i][0] = MDPositions[particleIndices[i]][0];
		positions[i][1] = MDPositions[particleIndices[i]][1];
		positions[i][2] = MDPositions[particleIndices[i]][2];
	}

	torch::Tensor signalsTensor = torch::zeros({numGhostParticles, 4}, torch::kFloat64);
	std::vector<double> globalVariables = extractContextVariables(context, numGhostParticles);
	signalsTensor = torch::from_blob(globalVariables.data(),
		{static_cast<int64_t>(numGhostParticles), 4}, torch::kFloat64);

	// Run the pytorch model and get the energy
	auto charges = signalsTensor.index({Slice(), 0});
	vector<torch::jit::IValue> nnInputs = {positionsTensor, charges};

	// Copy the box vector
	if (usePeriodic) {
		Vec3* box = extractBoxVectors(context);
		torch::Tensor boxVectorsTensor = torch::from_blob(box, {3, 3}, torch::kFloat64);
		nnInputs.push_back(boxVectorsTensor);
	}

	// outputTensor : attributes (ANI AEVs)
	torch::Tensor outputTensor = nnModule.forward(nnInputs).toTensor();
	

	// call Hungarian algorithm to determine mapping (and loss)
	if (assignFreq > 0) {
	  if (step_count % assignFreq == 0) {

		// concat ANI AEVS with atomic attributes [charge, sigma, epsiolon, lambda]
		torch::Tensor ghFeaturesTensor = torch::cat({outputTensor, signalsTensor}, 1);

		torch::Tensor distMatTensor = at::norm(ghFeaturesTensor.index({Slice(), None})
											   - targetFeaturesTensor, 2, 2);

		//convert it to a 2d vector
		std::vector<std::vector<double> > distMatrix = tensorTo2DVec(distMatTensor.data_ptr<double>(),
																	 numGhostParticles,
																	 static_cast<int>(targetFeaturesTensor.size(0)));

		assignment = hungAlg.Solve(distMatrix);
		reverse_assignment = getReverseAssignment(assignment);
	  }
	}
	// Save the assignments in the context variables
	for (std::size_t i=0; i<assignment.size(); i++) {
	  context.setParameter("assignment_g"+std::to_string(i), assignment[i]);
	}	
	step_count += 1;

	// reorder the targetFeaturesTensor using the mapping
	torch::Tensor reFeaturesTensor = targetFeaturesTensor.index({{torch::tensor(assignment)}}).clone();

	// determine energy using the feature loss
	torch::Tensor energyTensor = scale * torch::mse_loss(outputTensor,
		reFeaturesTensor.narrow(1, 0, outputTensor.size(1))).clone();

	// calculate force on the signals (first, clip out signals from the end of features) 
	torch::Tensor targtSignalsTensor = reFeaturesTensor.narrow(1, -4, 4);

	// update the global variables derivatives
	map<string, double>& energyParamDerivs = extractEnergyParameterDerivatives(context);
	auto targetSignalsData = targtSignalsTensor.accessor<double, 2>();
	double parameter_deriv;
	double param_energy = 0; 
	for (int i = 0; i < numGhostParticles; i++) {
		for (int j=0; j<4; j++)
		{
			parameter_deriv = signalForceWeights[j] * (globalVariables[i*4+j] - targetSignalsData[i][j]);
			// to do: need to substract out the (much smaller) energy arising from target signal discrepancies in the energyTensor
			param_energy += 0.5*(signalForceWeights[j]-1)*(globalVariables[i*4+j] - targetSignalsData[i][j])*(globalVariables[i*4+j] - targetSignalsData[i][j]);
			energyParamDerivs[PARAMETERNAMES[j]+std::to_string(i)] += parameter_deriv;
		}
	}
	
	// compute energies and forces from restraints
	double restraint_energy = 0;
	for (int i = 0; i < numRestraints; i++) {
		int g1idx = reverse_assignment[targetRestraintIndices[i][0]];
		int g2idx = reverse_assignment[targetRestraintIndices[i][1]];
		OpenMM::Vec3 r = MDPositions[g1idx] - MDPositions[g2idx];
		double rlensq = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];

		if (rlensq > r0sq[i]) {
			double rlen = sqrt(rlensq);
			if (rlen < rmax[i]) {
			    double dr = rlen-targetRestraintDistances[i];
				restraint_energy += 0.5*scale*restraint_k*dr*dr;
				if (includeForces) {
					OpenMM::Vec3 dvdx = scale*restraint_k*dr*r/rlen;
			  		for (int j = 0; j < 3; j++) {
						MDForce[g1idx][j] -= dvdx[j];
						MDForce[g2idx][j] += dvdx[j];				  
			  		}
				}
			} else {
			  restraint_energy += scale*(restraint_k*(rmax[i]-targetRestraintDistances[i])*rlen + restraint_b[i]);
			  if (includeForces) {
				OpenMM::Vec3 dvdx = scale*restraint_k*(rmax[i]-targetRestraintDistances[i])*r/rlen;
				for (int j = 0; j < 3; j++) {
				  MDForce[g1idx][j] -= dvdx[j];
				  MDForce[g2idx][j] += dvdx[j];
				}
			  }
			}
		}
	}
	
	// get forces on positions as before
	if (includeForces) {
		energyTensor.backward();

		// check if positions have gradients
		auto forceTensor = torch::zeros_like(positionsTensor);

		forceTensor = - positionsTensor.grad();
		positionsTensor.grad().zero_();
		if (!(forceTensor.dtype() == torch::kFloat64))
			forceTensor = forceTensor.to(torch::kFloat64);

		auto NNForce = forceTensor.accessor<double, 2>();
		for (int i = 0; i < numGhostParticles; i++) {
			MDForce[particleIndices[i]][0] += NNForce[i][0];
			MDForce[particleIndices[i]][1] += NNForce[i][1];
			MDForce[particleIndices[i]][2] += NNForce[i][2];
		}
	}
	return energyTensor.item<double>() + restraint_energy + param_energy;
  }
