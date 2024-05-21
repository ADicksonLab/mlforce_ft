#ifdef WIN32
#include <windows.h>
#include <sstream>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <cstdlib>
#endif

#include "PyTorchForce.h"
#include "PyTorchForceProxy.h"
#include "openmm/serialization/SerializationProxy.h"

#if defined(WIN32)
    #include <windows.h>
    extern "C" OPENMM_EXPORT_PYTORCH void registerPyTorchSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
	if (ul_reason_for_call == DLL_PROCESS_ATTACH)
	    registerPyTorchSerializationProxies();
	return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerPyTorchSerializationProxies();
#endif

using namespace PyTorchPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT_PYTORCH void registerPyTorchSerializationProxies() {
    SerializationProxy::registerProxy(typeid(PyTorchForce), new PyTorchForceProxy());
	SerializationProxy::registerProxy(typeid(PyTorchForceE2E), new PyTorchForceE2EProxy());
}
