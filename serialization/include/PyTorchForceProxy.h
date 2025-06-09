#ifndef OPENMM_PY_TORCH_FORCE_PROXY_H_
#define OPENMM_PY_TORCH_FORCE_PROXY_H_

#include "internal/windowsExportPyTorch.h"
#include "openmm/serialization/SerializationProxy.h"

namespace OpenMM {

/**
 * This is a proxy for serializing PyTorchForce objects.
 */

class OPENMM_EXPORT_PYTORCH PyTorchForceProxy : public SerializationProxy {
public:
    PyTorchForceProxy();
    void serialize(const void* object, SerializationNode& node) const;
    void* deserialize(const SerializationNode& node) const;
};

class OPENMM_EXPORT_PYTORCH PyTorchForceE2EProxy : public SerializationProxy {
public:
    PyTorchForceE2EProxy();
    void serialize(const void* object, SerializationNode& node) const;
    void* deserialize(const SerializationNode& node) const;
};

class OPENMM_EXPORT_PYTORCH PyTorchForceE2EDirectProxy : public SerializationProxy {
public:
    PyTorchForceE2EDirectProxy();
    void serialize(const void* object, SerializationNode& node) const;
    void* deserialize(const SerializationNode& node) const;
};

class OPENMM_EXPORT_PYTORCH PyTorchForceE2EDiffConfProxy : public SerializationProxy {
    public:
        PyTorchForceE2EDiffConfProxy();
        void serialize(const void* object, SerializationNode& node) const;
        void* deserialize(const SerializationNode& node) const;
    };
  
} // namespace OpenMM

#endif /*OPENMM_NEURAL_NETWORK_FORCE_PROXY_H_*/
