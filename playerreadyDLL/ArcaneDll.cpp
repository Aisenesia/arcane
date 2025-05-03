#include "ArcaneDll.h"
#include "PlayerReady.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdexcept>

// -----------------------------
// PlayerReady DLL Wrappers
// -----------------------------

extern "C" {

ARCANE_DLL_API void* CreatePlayerReady(int roiRadius, cv::Point center) {
    try {
        return new PlayerReady(roiRadius, center);
    } catch (...) {
        return nullptr;
    }
}

ARCANE_DLL_API void DestroyPlayerReady(void* instance) {
    if (instance) {
        delete static_cast<PlayerReady*>(instance);
    }
}

ARCANE_DLL_API void PlayerReady_SetFrame(void* instance, const cv::Mat& frame) {
    if (instance) {
        static_cast<PlayerReady*>(instance)->setFrame(frame);
    }
}

ARCANE_DLL_API bool PlayerReady_CheckReady(void* instance, int secondsToCheck) {
    if (instance) {
        return static_cast<PlayerReady*>(instance)->checkReady(secondsToCheck);
    }
    return false;
}

}
