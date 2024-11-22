#ifndef OPENMM_PYTORCH_FORCE_H_
#define OPENMM_PYTORCH_FORCE_H_

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <string>
#include "internal/windowsExportPyTorch.h"
#include<vector>

namespace PyTorchPlugin {

/**
 * This class implements forces that are defined by user-supplied neural networks.
 * It uses the PyTorch library to perform the computations. */

class OPENMM_EXPORT_PYTORCH PyTorchForce : public OpenMM::Force {
public:
	/**
	 * Create a PyTorchForce.  The network is defined by  Pytorch and saved
	 * to a .pt or .pth file
	 *
	 * @param file   the path to the file containing the network
	 */
  PyTorchForce(const std::string& file, std::vector<std::vector<double> > targetFeatures,
			   std::vector<int> particleIndices, std::vector<double> signalForceWeights, double scale, int assignFreq,
			   std::vector<std::vector<int> > restraintIndices, std::vector<double> restraintDistances, double rmaxDelta, double restraintK, std::vector<int> initialAssignment);
	/**
	 * Get the path to the file containing the graph.
	 */
	const std::string& getFile() const;
	const double getScale() const;
	const int getAssignFreq() const;
	const std::vector<std::vector<double> > getTargetFeatures() const;
	const std::vector<int> getParticleIndices() const;
	const std::vector<double> getSignalForceWeights() const;
	/**
	 * Get the initial assignments of ghost particles to 
	 * target atoms.  
	 * 
	 * @return initialAssignment
	 */
	const std::vector<int> getInitialAssignment() const;
	/**
	 * Get the atomic indices associated with each restraint.  
	 * 
	 * @return restraintIndices
	 */
	const std::vector<std::vector<int> > getRestraintIndices() const;
	/**
	 * Get the bond distances of the restraints.  
	 * 
	 * @return restraintDistances
	 */
	const std::vector<double> getRestraintDistances() const;
	/**
	 * Get the parameters that are common to all atomic restraints.  This returns a vector
	 * with two elements: the rmaxDelta and the restraintK.  rmaxDelta is equal to rmax - r0
	 * and is used to compute rmax.  rmax is the distance beyond which the restraint potential
	 * converts to linear. restraintK is the force constant of the harmonic restraint.
	 * 
	 * @return {rmax_delta, rest_k}
	 */
	const std::vector<double> getRestraintParams() const;
	/**
	 * Set whether this force makes use of periodic boundary conditions.  If this is set
	 * to true, the TensorFlow graph must include a 3x3 tensor called "boxvectors", which
	 * is set to the current periodic box vectors.
	 */
	void setUsesPeriodicBoundaryConditions(bool periodic);
	/**
	 * Get whether this force makes use of periodic boundary conditions.
	 */
	bool usesPeriodicBoundaryConditions() const;

	/**
	 * Get the number of global parameters that the interaction depends on.
	 */
	int getNumGlobalParameters() const;
	/**
	 * Add a new global parameter that the interaction may depend on.  The default value provided to
	 * this method is the initial value of the parameter in newly created Contexts.  You can change
	 * the value at any time by calling setParameter() on the Context.
	 *
	 * @param name             the name of the parameter
	 * @param defaultValue     the default value of the parameter
	 * @return the index of the parameter that was added
	 */
	int addGlobalParameter(const std::string& name, double defaultValue);
	/**
	 * Get the name of a global parameter.
	 *
	 * @param index     the index of the parameter for which to get the name
	 * @return the parameter name
	 */
	const std::string& getGlobalParameterName(int index) const;
	/**
	 * Set the name of a global parameter.
	 *
	 * @param index          the index of the parameter for which to set the name
	 * @param name           the name of the parameter
	 */
	void setGlobalParameterName(int index, const std::string& name);
	/**
	 * Get the default value of a global parameter.
	 *
	 * @param index     the index of the parameter for which to get the default value
	 * @return the parameter default value
	 */
	double getGlobalParameterDefaultValue(int index) const;
	/**
	 * Set the default value of a global parameter.
	 *
	 * @param index          the index of the parameter for which to set the default value
	 * @param defaultValue   the default value of the parameter
	 */
	void setGlobalParameterDefaultValue(int index, double defaultValue);
protected:
	OpenMM::ForceImpl* createImpl() const;
private:
	class GlobalParameterInfo;
	std::string file;
	std::vector<std::vector<double> > targetFeatures;
	std::vector<int> particleIndices;
	std::vector<int> initialAssignment;
	std::vector<double> signalForceWeights;
	double scale;
	int assignFreq;
	std::vector<std::vector<int> > restraintIndices;
	std::vector<double> restraintDistances;
	double rmaxDelta;
	double restraintK;
	bool usePeriodic;
	std::vector<GlobalParameterInfo> globalParameters;
};

/**
 * This is an internal class used to record information about a global parameter.
 * @private
 */
class PyTorchForce::GlobalParameterInfo {
public:
    std::string name;
    double defaultValue;
    GlobalParameterInfo() {
    }
    GlobalParameterInfo(const std::string& name, double defaultValue) : name(name), defaultValue(defaultValue) {
    }
};


/**
 * This class implements forces that are defined by user-supplied neural networks.
 * It uses the PyTorch library to perform the computations. */

class OPENMM_EXPORT_PYTORCH PyTorchForceE2E : public OpenMM::Force {
public:
	/**
	 * Create a PyTorchForceE2E.  The network is defined by  Pytorch and saved
	 * to a .pt or .pth file
	 *
	 * @param file   the path to the file containing the network
	 */
  PyTorchForceE2E(const std::string& file, 
				  std::vector<int> particleIndices, std::vector<double> signalForceWeights, double scale, double offset);
	/**
	 * Get the path to the file containing the graph.
	 */
	const std::string& getFile() const;
	const double getScale() const;
	const double getOffset() const;
	const std::vector<int> getParticleIndices() const;
	const std::vector<double> getSignalForceWeights() const;
	void setUsesPeriodicBoundaryConditions(bool periodic);
	/**
	 * Get whether this force makes use of periodic boundary conditions.
	 */
	bool usesPeriodicBoundaryConditions() const;

	/**
	 * Get the number of global parameters that the interaction depends on.
	 */
	int getNumGlobalParameters() const;
	/**
	 * Add a new global parameter that the interaction may depend on.  The default value provided to
	 * this method is the initial value of the parameter in newly created Contexts.  You can change
	 * the value at any time by calling setParameter() on the Context.
	 *
	 * @param name             the name of the parameter
	 * @param defaultValue     the default value of the parameter
	 * @return the index of the parameter that was added
	 */
	int addGlobalParameter(const std::string& name, double defaultValue);
	/**
	 * Get the name of a global parameter.
	 *
	 * @param index     the index of the parameter for which to get the name
	 * @return the parameter name
	 */
	const std::string& getGlobalParameterName(int index) const;
	/**
	 * Set the name of a global parameter.
	 *
	 * @param index          the index of the parameter for which to set the name
	 * @param name           the name of the parameter
	 */
	void setGlobalParameterName(int index, const std::string& name);
	/**
	 * Get the default value of a global parameter.
	 *
	 * @param index     the index of the parameter for which to get the default value
	 * @return the parameter default value
	 */
	double getGlobalParameterDefaultValue(int index) const;
	/**
	 * Set the default value of a global parameter.
	 *
	 * @param index          the index of the parameter for which to set the default value
	 * @param defaultValue   the default value of the parameter
	 */
	void setGlobalParameterDefaultValue(int index, double defaultValue);
protected:
	OpenMM::ForceImpl* createImpl() const;
private:
	class GlobalParameterInfo;
	std::string file;
	std::vector<int> particleIndices;
	std::vector<double> signalForceWeights;
	double scale, offset;
	bool usePeriodic;
	std::vector<GlobalParameterInfo> globalParameters;
};

/**
 * This is an internal class used to record information about a global parameter.
 * @private
 */
class PyTorchForceE2E::GlobalParameterInfo {
public:
    std::string name;
    double defaultValue;
    GlobalParameterInfo() {
    }
    GlobalParameterInfo(const std::string& name, double defaultValue) : name(name), defaultValue(defaultValue) {
    }
};

/**
 * This class implements forces that are defined by user-supplied neural networks.
 * It uses the PyTorch library to perform the computations. */

class OPENMM_EXPORT_PYTORCH PyTorchForceE2EDirect : public OpenMM::Force {
public:
	/**
	 * Create a PyTorchForceE2EDirect.  The network is defined by  Pytorch and saved
	 * to a .pt or .pth file
	 *
	 * @param file   the path to the file containing the network
	 */
  PyTorchForceE2EDirect(const std::string& file, 
						std::vector<int> particleIndices,
						std::vector<double> signalForceWeights,
						double scale,
						std::vector<torch::Tensor> fixedInputs,
						bool useAttr);
	/**
	 * Get the path to the file containing the graph.
	 */
	const std::string& getFile() const;
	const double getScale() const;
	const std::vector<torch::Tensor> getFixedInputs() const;
	const std::vector<int> getParticleIndices() const;
	const std::vector<double> getSignalForceWeights() const;
	const bool getUseAttr() const;
	void setUsesPeriodicBoundaryConditions(bool periodic);
	/**
	 * Get whether this force makes use of periodic boundary conditions.
	 */
	bool usesPeriodicBoundaryConditions() const;

	/**
	 * Get the number of global parameters that the interaction depends on.
	 */
	int getNumGlobalParameters() const;
	/**
	 * Add a new global parameter that the interaction may depend on.  The default value provided to
	 * this method is the initial value of the parameter in newly created Contexts.  You can change
	 * the value at any time by calling setParameter() on the Context.
	 *
	 * @param name             the name of the parameter
	 * @param defaultValue     the default value of the parameter
	 * @return the index of the parameter that was added
	 */
	int addGlobalParameter(const std::string& name, double defaultValue);
	/**
	 * Get the name of a global parameter.
	 *
	 * @param index     the index of the parameter for which to get the name
	 * @return the parameter name
	 */
	const std::string& getGlobalParameterName(int index) const;
	/**
	 * Set the name of a global parameter.
	 *
	 * @param index          the index of the parameter for which to set the name
	 * @param name           the name of the parameter
	 */
	void setGlobalParameterName(int index, const std::string& name);
	/**
	 * Get the default value of a global parameter.
	 *
	 * @param index     the index of the parameter for which to get the default value
	 * @return the parameter default value
	 */
	double getGlobalParameterDefaultValue(int index) const;
	/**
	 * Set the default value of a global parameter.
	 *
	 * @param index          the index of the parameter for which to set the default value
	 * @param defaultValue   the default value of the parameter
	 */
	void setGlobalParameterDefaultValue(int index, double defaultValue);
protected:
	OpenMM::ForceImpl* createImpl() const;
private:
	class GlobalParameterInfo;
	std::string file;
	std::vector<int> particleIndices;
	std::vector<double> signalForceWeights;
	double scale;
	std::vector<torch::Tensor> fixedInputs;
	bool useAttr;
	bool usePeriodic;
	std::vector<GlobalParameterInfo> globalParameters;
};

/**
 * This is an internal class used to record information about a global parameter.
 * @private
 */
class PyTorchForceE2EDirect::GlobalParameterInfo {
public:
    std::string name;
    double defaultValue;
    GlobalParameterInfo() {
    }
    GlobalParameterInfo(const std::string& name, double defaultValue) : name(name), defaultValue(defaultValue) {
    }
};

  
} // namespace PyTorchPlugin

#endif /*OPENMM_PYTORCHFORCE_H_*/
