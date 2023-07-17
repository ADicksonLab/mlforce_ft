
#include "ReferencePyTorchKernels.h"
#include "PyTorchForce.h"
#include "Hungarian.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include <algorithm>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

std::pair<int,std::vector<int>> getTarget(torch::Tensor ghFeatures, torch::Tensor lambdas, std::vector<torch::Tensor> targetFeatures, double lambda_mismatch_penalty) {

    int nTargets = targetFeatures.size(0);
    int nGhostAtoms = ghFeatures.size(0);

	// sizes of input tensors
	//
	// ghFeatures : n x nf 
	// targetFeatures[i]  : m x nf
	// distMatTensor : n x m
	// lambdas : n

	std::vector<double> costs(nTargets, 0.0);
	std::vector<std::vector<int>> allAssignments(nTargets);
	HungarianAlgorithm hungAlg;

	for (int i=0; i < nTargets; i++) {
	    int nTargetAtoms = targetFeatures[i].size(0);

		assert(nGhostAtoms >= nTargetAtoms);
	
		torch::Tensor lambdaExpand = at::tile(lambdas.unsqueeze(1),{1,nTargetAtoms});   // expand to shape (n,m)
		torch::Tensor ghExpand = at::tile(ghFeatures.unsqueeze(1),{1,nTargetAtoms,1});  // expand to shape (n,m,nf)
		torch::Tensor targetExpand = at::tile(targetFeatures[i],{nGhostAtoms,1,1});     // expand to shape (n,m,nf)
		torch::Tensor diff = ghExpand - targetExpand;

		// distmat[a][b] is the distance from ghost atom a to target atom b
		torch::Tensor distmat = (diff*diff).sum(2)*lambdaExpand;
	
		// pad with zeros
		at::Tensor distmat_nn = at::constant_pad_nd(distmat, {0, nGhostAtoms-nTargetAtoms}, 0); // pad to shape (n,n)
		
		// target lambdas (n,n) (second dimension is different)
		torch::Tensor lambda_T = torch::constant_pad_nd(torch::ones({nGhostAtoms,nTargetAtoms}), {0,nGhostAtoms-nTargetAtoms}, 0);

		// ghost lambdas (n,n) (first dimension is different)
		torch::Tensor lambda_gh = at::tile(lambdas.unsqueeze(1),{1,nGhostAtoms});
	
		// add lambdaMismatchPenalty*(lambda[i] - lambda_T[j])**2
		distmat_nn += lambda_mismatch_penalty*torch::pow(lambda_T-lambda_gh,2);
	
		//convert it to a 2d vector
		std::vector<std::vector<double> > distMatrix = tensorTo2DVec(distmat_nn.data_ptr<double>(),
																	 nGhostAtoms,nGhostAtoms);
	
		allAssignments[i] = hungAlg.Solve(distMatrix);
		at::Tensor assignment_tensor = at::tensor(allAssignments[i], at::kLong);
		costs[i] = (at::one_hot(assignment_tensor, nGhostAtoms) * distmat_nn).sum().item<double>();
	}

	// get the index corresponding to the minimum cost
	std::vector<int>::iterator result = std::min_element(costs.begin(), costs.end());
	int bestIndex = std::distance(costs.begin(), result);
	return {bestIndex, allAssignments[bestIndex]};

}

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
std::vector<double> extractContextVariables(ContextImpl& context, int numParticles) {
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
std::vector<std::vector<double> > tensorTo2DVec(double* ptr, int nRows, int nCols) {
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
std::vector<int> getReverseAssignment(std::vector<int> assignment) {
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
void ReferenceCalcPyTorchForceKernel::initialize(const System& system, const PyTorchForce& force, torch::jit::script::Module nnModule) {
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
	lambdaMismatchPenalty = force.getLambdaMismatchPenalty();
	rmax_delta = targetRestraintParams[0];
	restraint_k = targetRestraintParams[1];

	targetIdx = force.getInitialTargetIdx();
	assignment = force.getInitialAssignment();
	reverse_assignment = getReverseAssignment(assignment);
	
	numTargets = targetRestraintDistances.size();
	for (int t_idx = 0; t_idx < numTargets; t_idx++) {
  	    numRestraints.push_back(targetRestraintDistances[t_idx].size());
		std::vector<double> tmp_r0sq,tmp_rmax,tmp_restraint_b;
		for (int i = 0; i < numRestraints[t_idx]; i++) {
		    tmp_r0sq.push_back(targetRestraintDistances[t_idx][i]*targetRestraintDistances[t_idx][i]);
			tmp_rmax.push_back(targetRestraintDistances[t_idx][i] + rmax_delta);
			tmp_restraint_b.push_back(0.5*restraint_k*(r0sq[i] - rmax[i]*rmax[i]));
		}
		r0sq.push_back(tmp_r0sq);
		rmax.push_back(tmp_rmax);
		restraint_b.push_back(tmp_restraint_b);
	}

	usePeriodic = force.usesPeriodicBoundaryConditions();
	int numGhostParticles = particleIndices.size();

	//get target features
	targetFeatures = force.getTargetFeatures();
	for (int t_idx = 0; t_idx < targetFeatures.size(); t_idx++) {
	    targetFeaturesTensor = torch::zeros({static_cast<int64_t>(targetFeatures[t_idx].size()),
											 static_cast<int64_t>(targetFeatures[t_idx][0].size())},
		  torch::TensorOptions().dtype(torch::kFloat64));

		for (std::size_t i = 0; i < targetFeatures[t_idx].size(); i++)
  		    targetFeaturesTensor.slice(0, i, i+1) = torch::from_blob(targetFeatures[t_idx][i].data(),
																	 {(long long)targetFeatures[t_idx][0].size()},
																	 torch::TensorOptions().dtype(torch::kFloat64));
		allTargetFeatures.push_back(targetFeaturesTensor);
	}

	torch::Tensor signalFW_tensor = torch::from_blob(signalForceWeights.data(),
													 {static_cast<int64_t>(signalForceWeights.size())},
													 torch::kFloat64);

	// subtract 4 from targetFeatures size to get ani features size
	int nAniFeatures  = targetFeatures[0][0].size() - 4; // Number of features (except gp attributes)
	allForceWeights = torch::cat({torch::ones({nAniFeatures}, torch::kFloat64), signalFW_tensor}, 0); // all force weights (features + attributes)

	
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

	signalsTensor.requires_grad_(true);

	// Run the pytorch model and get the energy
	auto charges = signalsTensor.index({Slice(), 0});
	auto lambdas = signalsTensor.index({Slice(), 3});
	vector<torch::jit::IValue> nnInputs = {positionsTensor, charges};

	// Copy the box vector
	if (usePeriodic) {
		Vec3* box = extractBoxVectors(context);
		torch::Tensor boxVectorsTensor = torch::from_blob(box, {3, 3}, torch::kFloat64);
		nnInputs.push_back(boxVectorsTensor);
	}

	// outputTensor : attributes (ANI AEVs)
	torch::Tensor outputTensor = nnModule.forward(nnInputs).toTensor();

	torch::Tensor ghFeaturesTensor = torch::cat({outputTensor, signalsTensor}, 1); 

	// call Hungarian algorithm to determine mapping (and loss)
	if (assignFreq > 0) {
	  if (step_count % assignFreq == 0) {

		// when passing to getTarget, include the lambdas separately
		std::pair<int,std::vector<int>> result = getTarget(ghFeaturesTensor.index({Slice(),Slice(0,-1)}),lambdas,lambdaMismatchPenalty);
		targetIdx = result.first;
		assignment = result.second;

		// assignment[i]:  which target atom ghost atom i assigned to?
		// reverse_assignment[i]:  which ghost atom is assigned to target atom i?
		reverse_assignment = getReverseAssignment(assignment);

	  }
	}
	// Save the assignments in the context variables
	for (std::size_t i=0; i<assignment.size(); i++) {
	  context.setParameter("assignment_g"+std::to_string(i), assignment[i]);
	}
	context.setParameter("targetIdx",targetIdx);
	step_count += 1;

	// reorder the targetFeaturesTensor using the mapping
	int nTargetAtoms = targetFeaturesTensor[targetIdx].size(0);

	// get difference between re_gh and target (multiply by allForceWeights and lambda)
	torch::Tensor reGhFeaturesTensor = ghFeaturesTensor.index({{torch::tensor(reverse_assignment)}});
	auto reGhLambdasTensor = lambdas.index({{torch::tensor(reverse_assignment)}});

	// don't include lambda in the diff
	torch::Tensor diff = reGhFeaturesTensor.index({Slice(0,nTargetAtoms),Slice(0,-1)}) - targetFeaturesTensor[targetIdx].index({Slice(),Slice(0,-1)});
	
	// add lambda terms for unassigned atoms
	auto unassigned_lambdas = lambdas.index({{torch::gt(torch::tensor(assignment),nTargetAtoms-1)}});
	torch::Tensor energy_tmp = (diff*diff*allForceWeights*reGhLambdasTensor.index({Slice(0,nTargetAtoms)})).sum() +
	  (unassigned_lambdas*unassigned_lambdas*lambdaMismatchPenalty).sum();
	  
	torch::Tensor energyTensor = scale * energy_tmp.clone() / (ghFeaturesTensor.size(0) * ghFeaturesTensor.size(1));

	// compute energies and forces from restraints
	double restraint_energy = 0;
	for (int i = 0; i < numRestraints[targetIdx]; i++) {
		int g1idx = reverse_assignment[targetRestraintIndices[targetIdx][i][0]];
		int g2idx = reverse_assignment[targetRestraintIndices[targetIdx][i][1]];
		OpenMM::Vec3 r = MDPositions[g1idx] - MDPositions[g2idx];
		double rlensq = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];

		double rlen = sqrt(rlensq);
		if (rlen < rmax[targetIdx][i]) {
		  double dr = rlen-targetRestraintDistances[targetIdx][i];
		  restraint_energy += 0.5*scale*restraint_k*dr*dr;
		  if (includeForces) {
			OpenMM::Vec3 dvdx = scale*restraint_k*dr*r/rlen;
			for (int j = 0; j < 3; j++) {
			  MDForce[g1idx][j] -= dvdx[j];
			  MDForce[g2idx][j] += dvdx[j];				  
			}
		  }
		} else {
		  restraint_energy += scale*(restraint_k*(rmax[targetIdx][i]-targetRestraintDistances[targetIdx][i])*rlen + restraint_b[targetIdx][i]);
		  if (includeForces) {
			OpenMM::Vec3 dvdx = scale*restraint_k*(rmax[targetIdx][i]-targetRestraintDistances[targetIdx][i])*r/rlen;
			for (int j = 0; j < 3; j++) {
			  MDForce[g1idx][j] -= dvdx[j];
			  MDForce[g2idx][j] += dvdx[j];
			}
		  }
		}
	}
		
	// get forces on positions
	if (includeForces) {

	  energyTensor.backward();

	  auto forceTensor = torch::zeros_like(positionsTensor);
	  auto signalsGradTensor = torch::zeros_like(signalsTensor);

	  forceTensor = - positionsTensor.grad().clone();	 
	  signalsGradTensor = signalsTensor.grad().clone();

	  signalsTensor.grad().zero_();
	  positionsTensor.grad().zero_();
	  
	  if (!(signalsGradTensor.dtype() == torch::kFloat64))
		signalsGradTensor = signalsGradTensor.to(torch::kFloat64);
		
	  map<string, double>& energyParamDerivs = extractEnergyParameterDerivatives(context);
	  auto signalsGradData = signalsGradTensor.accessor<double, 2>();
	  for (int i = 0; i < numGhostParticles; i++) {
		for (int j=0; j<4; j++){
		  energyParamDerivs[PARAMETERNAMES[j]+std::to_string(i)] += signalsGradData[i][j];
		}
	  }

	  if (!(forceTensor.dtype() == torch::kFloat64))
		forceTensor = forceTensor.to(torch::kFloat64);
		
	  auto NNForce = forceTensor.accessor<double, 2>();
	  for (int i = 0; i < numGhostParticles; i++) {
		MDForce[particleIndices[i]][0] += NNForce[i][0];
		MDForce[particleIndices[i]][1] += NNForce[i][1];
		MDForce[particleIndices[i]][2] += NNForce[i][2];
	  }
	}

	return energyTensor.item<double>() + restraint_energy;
  }
