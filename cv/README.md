# Arcane - High-Performance Computer Vision Inference DLL

Arcane is a high-performance C++ inference dynamic link library (DLL) designed to integrate neural networks for object detection and classification into external graphics engines and host languages (e.g., **Unreal Engine**, **Unity/C#**, and **Python**). 

The module leverages **OpenCV DNN** with optional **NVIDIA CUDA** GPU acceleration.

---

## Key Features & Design Choices

*   **Unreal Engine-Compatible C-API Boundary**: Public headers use strictly C-style parameters (primitive types, raw pointers, and simple structs). There are no C++ standard library (`std::`) or OpenCV (`cv::`) types in the exported interface. This prevents ABI, compiler mismatch, and runtime standard library linkage conflicts when integrating with Unreal Engine.
*   **Security & Stability**: 
    *   No fixed-buffer limits for detections; memory is allocated dynamically based on exact detection counts to prevent buffer overflow vulnerabilities.
    *   All entry points are wrapped in structured `try-catch` blocks, preventing C++ library exceptions from propagating past the DLL boundary and crashing host applications.
*   **Non-Maximum Suppression (NMS)**: Features a robust built-in NMS layer to filter out overlapping bounding boxes returned by the object detector (e.g., YOLO-like models).
*   **Optimal Pipeline**: Image preprocessing (resizing) is handled on the CPU, eliminating expensive CPU-GPU memory copy cycles that are typically slower than CPU execution for simple resizing tasks.
*   **Modern CMake Build System**: Cross-platform configuration which automatically integrates with packages resolved via `vcpkg`.

---

## API Documentation

The DLL exports the following C-style functions:

```cpp
// Initialize detection and classification neural networks
bool InitializeNetworks(const char* detectionModelPath, const char* classificationModelPath, bool useCuda);

// Run object detection & classification on raw image bytes
DetectionResultArray Detect(const unsigned char* imageData, int width, int height, int channels);

// Run classification on raw image bytes
ClassificationResult Classify(const unsigned char* imageData, int width, int height, int channels);

// Safely free the detection results memory allocated by the DLL
void FreeDetectionResults(DetectionResultArray array);

// Unload networks and release resources
void Cleanup();
```

### Data Structures

```cpp
struct DetectionResult {
    int x;             // Left coordinate of bounding box
    int y;             // Top coordinate of bounding box
    int width;         // Width of bounding box
    int height;        // Height of bounding box
    int classId;       // Classified class ID
    float confidence;  // Score between 0.0 and 1.0
};

struct DetectionResultArray {
    DetectionResult* results;  // Pointer to array of results
    int size;                  // Size of array
};
```

---

## Integration Examples

### 1. Unreal Engine (C++ Dynamic DLL Loading)

```cpp
// Fill raw image buffer (e.g., from UTexture2D or Scene Capture)
TArray<uint8> RawBytes; 
int32 Width = 1920, Height = 1080, Channels = 3;

// Define function pointers matching DLL exports
typedef bool(*_InitFunc)(const char*, const char*, bool);
typedef DetectionResultArray(*_DetectFunc)(const unsigned char*, int, int, int);
typedef void(*_FreeFunc)(DetectionResultArray);

// Load the DLL
void* DLLHandle = FPlatformProcess::GetDllHandle(TEXT("arcane_dll.dll"));
if (DLLHandle)
{
    _InitFunc InitNetworks = (_InitFunc)FPlatformProcess::GetDllExport(DLLHandle, TEXT("InitializeNetworks"));
    _DetectFunc Detect = (_DetectFunc)FPlatformProcess::GetDllExport(DLLHandle, TEXT("Detect"));
    _FreeFunc FreeResults = (_FreeFunc)FPlatformProcess::GetDllExport(DLLHandle, TEXT("FreeDetectionResults"));

    if (InitNetworks("yolov8.onnx", "classifier.onnx", true))
    {
        DetectionResultArray OutDetections = Detect(RawBytes.GetData(), Width, Height, Channels);
        
        for (int i = 0; i < OutDetections.size; ++i)
        {
            DetectionResult Object = OutDetections.results[i];
            UE_LOG(LogTemp, Warning, TEXT("Detected Class %d with Confidence %.2f"), Object.classId, Object.confidence);
        }

        // Clean up memory safely using the DLL's exported method
        FreeResults(OutDetections);
    }
    
    FPlatformProcess::FreeDllHandle(DLLHandle);
}
```

### 2. C# / Unity (P/Invoke)

```csharp
using System;
using System.Runtime.InteropServices;

public class ArcaneWrapper
{
    [StructLayout(LayoutKind.Sequential)]
    public struct DetectionResult
    {
        public int x;
        public int y;
        public int width;
        public int height;
        public int classId;
        public float confidence;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DetectionResultArray
    {
        public IntPtr results;
        public int size;
    }

    [DllImport("arcane_dll.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bool InitializeNetworks(string detPath, string classPath, bool useCuda);

    [DllImport("arcane_dll.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern DetectionResultArray Detect(byte[] imageData, int width, int height, int channels);

    [DllImport("arcane_dll.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void FreeDetectionResults(DetectionResultArray array);

    [DllImport("arcane_dll.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void Cleanup();
}
```

---

## Build Instructions

### Prerequisites
*   CMake 3.16+
*   Visual Studio 2022 (MSVC Compiler)
*   OpenCV 4.x (via `vcpkg` recommended)
*   CUDA Toolkit (Optional, version 12.x recommended)

### Compilation
Configure and build using CMake with `vcpkg` integration:

```bash
# Configure build directory
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[VCPKG_ROOT]/scripts/buildsystems/vcpkg.cmake

# Build target
cmake --build build --config Release
```

The compiled library `arcane_dll.dll` will be generated in `build/Release/`.
