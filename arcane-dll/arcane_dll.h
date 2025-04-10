#pragma once

#define ARCANE_DLL_EXPORTS

#ifdef ARCANE_DLL_EXPORTS
#define ARCANE_DLL_API __declspec(dllexport)
#else
#define ARCANE_DLL_API __declspec(dllimport)
#endif

extern "C" {
    ARCANE_DLL_API bool InitializeNetworks(const char* detectionModelPath, const char* classificationModelPath, bool useCuda);
    ARCANE_DLL_API void RunLiveDetection();
    ARCANE_DLL_API void RunDetection(const char* imagePath);
    ARCANE_DLL_API void RunClassification(const char* imagePath);
    ARCANE_DLL_API void Cleanup(); // Add the Cleanup function declaration
}