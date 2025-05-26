#include "arcane_dll.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <filesystem> // For checking file existence

#define NOMINMAX

// Network includes (must come before windows.h)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <winhttp.h>

// Windows includes
#include <windows.h>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/dnn.hpp> // For NMSBoxes

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "winhttp.lib")

#define DICE_CLASS 0 // 0 is the class ID for dice, all other detections are different cards.
#define SERVER_IP "localhost"
#define SERVER_PORT 3001 // Example port for server communication

// Window management constants
const char *WINDOW_NAME = "Camera View";
const int WINDOW_PADDING = 50;           // Padding from screen edges
const double DEFAULT_SCALE_FACTOR = 0.8; // Default scale factor for video

// Struct to store display settings
struct DisplaySettings
{
    int screenWidth;
    int screenHeight;
    double maxWindowWidth;
    double maxWindowHeight;
};

// Create and position a window properly
void setupWindow()
{
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, 1600, 720);
}

void applyNonMaximumSuppression(const DetectionResultArray &arr, std::vector<cv::Rect> &filteredBoxes, std::vector<float> &confidences, float nmsThreshold = 0.4)
{
    std::vector<int> indices;
    for (int i = 0; i < arr.size; i++)
    {
        filteredBoxes.push_back(arr.results[i].boundingBox);
        confidences.push_back(arr.results[i].confidence);
    }

    // Apply Non-Maximum Suppression
    cv::dnn::NMSBoxes(filteredBoxes, confidences, 0.5, nmsThreshold, indices);

    // Filter out boxes based on NMS results
    std::vector<cv::Rect> nmsFilteredBoxes;
    for (int idx : indices)
    {
        nmsFilteredBoxes.push_back(filteredBoxes[idx]);
    }

    filteredBoxes = nmsFilteredBoxes;
}

void processReadyCommand(cv::VideoCapture &cap)
{
    setupWindow();

    cv::Point center(100, 100);
    int radius = 50;
    int secondsToWait = 3;

    setReadyProperties(center, radius, secondsToWait);

    while (true)
    {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty())
        {
            std::cerr << "Error: Empty frame received! Exiting 'ready' mode." << std::endl;
            break;
        }

        bool ready = checkPlayerReady(frame);
        if (ready)
        {
            std::cout << "Player is ready!" << std::endl;
        }

        cv::circle(frame, center, radius, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Camera View", frame);

        cv::resizeWindow(WINDOW_NAME, frame.cols, frame.rows);

        if (cv::waitKey(30) == 27)
            break; // Exit on 'ESC' key
    }

    cv::destroyWindow("Camera View");
}

void processDiceCommand(cv::VideoCapture &cap)
{
    const char *detectionModelPath = "dtc.onnx";
    const char *classificationModelPath = "cls.onnx";

    if (!InitializeNetworks(detectionModelPath, classificationModelPath, true))
    {
        std::cerr << "Failed to initialize networks. Please check the model files and paths." << std::endl;
        return;
    }

    setupWindow();

    while (true)
    {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty())
        {
            std::cerr << "Empty frame received!" << std::endl;
            break;
        }

        DetectionResultArray arr = Detect(frame);
        std::vector<cv::Rect> filteredBoxes;
        std::vector<float> confidences;
        applyNonMaximumSuppression(arr, filteredBoxes, confidences);

        for (size_t i = 0; i < filteredBoxes.size(); ++i)
        {
            cv::rectangle(frame, filteredBoxes[i], cv::Scalar(0, 255, 0), 2);
            std::string label = "Class: " + std::to_string(arr.results[i].classId);
            cv::putText(frame, label, cv::Point(filteredBoxes[i].x, filteredBoxes[i].y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        cv::imshow(WINDOW_NAME, frame);

        // Ensure window fits properly on screen
        cv::resizeWindow(WINDOW_NAME, frame.cols, frame.rows);

        if (cv::waitKey(30) == 27)
            break; // Exit on 'ESC' key
    }
}

void processCalibrateCommand(cv::VideoCapture &cap)
{
    std::cout << "Calibration mode activated." << std::endl;
    cv::Point centers[] = {cv::Point(150, 150), cv::Point(1130, 570)};
    int radii[2] = {60, 60};
    setCalibrationProperties(centers, radii);

    setupWindow();

    while (true)
    {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty())
        {
            std::cerr << "Empty frame received!" << std::endl;
            break;
        }

        for (size_t i = 0; i < 2; ++i)
        {
            cv::circle(frame, centers[i], radii[i], cv::Scalar(255, 0, 0), 2);
        }

        bool result = checkCalibrationApi(frame);
        std::string status = result ? "KALIBRASYON TAM" : "KALIBRASYON HATALI";
        cv::putText(frame, status, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                    result ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

        cv::imshow(WINDOW_NAME, frame);

        // Ensure window fits properly on screen
        cv::resizeWindow(WINDOW_NAME, frame.cols, frame.rows);

        if (cv::waitKey(30) == 27)
            break; // Exit on 'ESC' key
    }
}

void listCurrentDirectory()
{
    try
    {
        namespace fs = std::filesystem;
        std::string currentPath = fs::current_path().string();

        // Open file for logging
        std::ofstream logFile("arcane-file-logs.txt", std::ios::app);

        auto writeToConsoleAndFile = [&](const std::string &message)
        {
            std::cout << message;
            if (logFile.is_open())
            {
                logFile << message;
            }
        };

        writeToConsoleAndFile("Current working directory: " + currentPath + "\n");
        writeToConsoleAndFile("Directory contents:\n");

        for (const auto &entry : fs::directory_iterator(currentPath))
        {
            if (entry.is_directory())
            {
                writeToConsoleAndFile("[DIR]  " + entry.path().filename().string() + "\n");
            }
            else if (entry.is_regular_file())
            {
                writeToConsoleAndFile("[FILE] " + entry.path().filename().string() +
                                      " (" + std::to_string(entry.file_size()) + " bytes)\n");
            }
            else
            {
                writeToConsoleAndFile("[OTHER] " + entry.path().filename().string() + "\n");
            }
        }
        writeToConsoleAndFile("\n");

        if (logFile.is_open())
        {
            logFile.close();
        }
    }
    catch (const std::exception &e)
    {
        std::string errorMsg = "Error listing directory: " + std::string(e.what()) + "\n";
        std::cerr << errorMsg;

        std::ofstream logFile("arcane-file-logs.txt", std::ios::app);
        if (logFile.is_open())
        {
            logFile << errorMsg;
            logFile.close();
        }
    }
}

// Struct for user data
struct UserData
{
    std::string token;
    std::string characterId;
};

// Struct for API response
struct APIResponse
{
    std::string sessionId;
    std::string gameStatus;
    std::string currentTurnCharacterId;
    bool success;
    std::string fullResponse;
};

// Function to initialize Winsock
bool initializeWinsock()
{
    WSADATA wsaData;
    int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (result != 0)
    {
        std::cerr << "WSAStartup failed: " << result << std::endl;
        return false;
    }
    return true;
}

// Function to cleanup Winsock
void cleanupWinsock()
{
    WSACleanup();
}

// Function to connect to ESP8266
SOCKET connectToESP8266(const std::string &esp8266_ip, int port)
{
    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET)
    {
        std::cerr << "Socket creation failed" << std::endl;
        return INVALID_SOCKET;
    }

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    inet_pton(AF_INET, esp8266_ip.c_str(), &serverAddr.sin_addr);

    if (connect(sock, (sockaddr *)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR)
    {
        std::cerr << "Connection to ESP8266 failed" << std::endl;
        closesocket(sock);
        return INVALID_SOCKET;
    }

    std::cout << "Connected to ESP8266 at " << esp8266_ip << ":" << port << std::endl;
    return sock;
}

// Function to receive data from ESP8266
std::string receiveFromESP8266(SOCKET sock)
{
    char buffer[1024] = {0};
    int bytesReceived = recv(sock, buffer, 1024, 0);
    if (bytesReceived > 0)
    {
        return std::string(buffer, bytesReceived);
    }
    return "";
}

// Function to parse user data from received string
UserData parseUserData(const std::string &data)
{
    UserData userData;
    size_t tokenPos = data.find("token:");
    size_t charPos = data.find("character:");

    if (tokenPos != std::string::npos && charPos != std::string::npos)
    {
        size_t tokenStart = tokenPos + 6; // length of "token:"
        size_t tokenEnd = data.find(",", tokenStart);
        if (tokenEnd == std::string::npos)
            tokenEnd = data.find("\n", tokenStart);
        if (tokenEnd == std::string::npos)
            tokenEnd = data.length();

        size_t charStart = charPos + 10; // length of "character:"
        size_t charEnd = data.find(",", charStart);
        if (charEnd == std::string::npos)
            charEnd = data.find("\n", charStart);
        if (charEnd == std::string::npos)
            charEnd = data.length();

        userData.token = data.substr(tokenStart, tokenEnd - tokenStart);
        userData.characterId = data.substr(charStart, charEnd - charStart);

        // Trim whitespace
        userData.token.erase(0, userData.token.find_first_not_of(" \t\r\n"));
        userData.token.erase(userData.token.find_last_not_of(" \t\r\n") + 1);
        userData.characterId.erase(0, userData.characterId.find_first_not_of(" \t\r\n"));
        userData.characterId.erase(userData.characterId.find_last_not_of(" \t\r\n") + 1);
    }

    return userData;
}

// Function to extract session ID (_id) from JSON response
std::string extractSessionId(const std::string &jsonResponse)
{
    // Find the top-level "_id" field
    size_t idPos = jsonResponse.find("\"_id\":");
    if (idPos != std::string::npos)
    {
        // Ensure this "_id" is not part of the "users" array
        size_t usersPos = jsonResponse.find("\"users\":");
        if (usersPos != std::string::npos && idPos > usersPos)
        {
            // Skip this "_id" as it belongs to the "users" array
            idPos = jsonResponse.find("\"_id\":", idPos + 1);
        }

        // Extract the value of the top-level "_id"
        if (idPos != std::string::npos)
        {
            size_t idStart = jsonResponse.find("\"", idPos + 5); // Skip "_id": and find the opening quote
            if (idStart != std::string::npos)
            {
                idStart++; // Move past the opening quote
                size_t idEnd = jsonResponse.find("\"", idStart); // Find the closing quote
                if (idEnd != std::string::npos)
                {
                    return jsonResponse.substr(idStart, idEnd - idStart);
                }
            }
        }
    }
    return "";
}

// Function to make API call using WinHTTP
APIResponse makeAPICall(const std::string &token, const std::string &characterId, bool isFirstUser, const std::string &sessionId = "")
{
    APIResponse response;
    response.success = false;

    HINTERNET hSession = WinHttpOpen(L"Arcane DLL Tester/1.0",
                                     WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                     WINHTTP_NO_PROXY_NAME,
                                     WINHTTP_NO_PROXY_BYPASS, 0);

    if (hSession)
    {
        // Convert SERVER_IP to wide string
        std::wstring serverIP(SERVER_IP, SERVER_IP + strlen(SERVER_IP));

        HINTERNET hConnect = WinHttpConnect(hSession, serverIP.c_str(), SERVER_PORT, 0);

        if (hConnect)
        {
            std::wstring endpoint;
            if (isFirstUser)
            {
                endpoint = L"/api/cv/create";
            }
            else
            {
                endpoint = L"/api/cv/" + std::wstring(sessionId.begin(), sessionId.end()) + L"/join";
            }

            HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"POST", endpoint.c_str(),
                                                    NULL, WINHTTP_NO_REFERER,
                                                    WINHTTP_DEFAULT_ACCEPT_TYPES,
                                                    0); // Remove WINHTTP_FLAG_SECURE for localhost

            if (hRequest)
            {
                // Prepare request data (JSON body)
                std::string postData = "{\"characterId\":\"" + characterId + "\"}";

                // Prepare headers
                std::string authHeader = "Authorization: Bearer " + token;
                std::wstring headers = L"Content-Type: application/json\r\n";
                headers += std::wstring(authHeader.begin(), authHeader.end()) + L"\r\n";

                BOOL bResults = WinHttpSendRequest(hRequest, headers.c_str(), -1,
                                                   (LPVOID)postData.c_str(), postData.length(),
                                                   postData.length(), 0);

                if (bResults)
                {
                    bResults = WinHttpReceiveResponse(hRequest, NULL);

                    if (bResults)
                    {
                        DWORD dwSize = 0;
                        DWORD dwDownloaded = 0;
                        LPSTR pszOutBuffer;
                        std::string result;

                        do
                        {
                            dwSize = 0;
                            if (!WinHttpQueryDataAvailable(hRequest, &dwSize))
                            {
                                break;
                            }

                            if (dwSize == 0)
                                break;

                            pszOutBuffer = new char[dwSize + 1];
                            if (!pszOutBuffer)
                            {
                                break;
                            }

                            ZeroMemory(pszOutBuffer, dwSize + 1);

                            if (!WinHttpReadData(hRequest, (LPVOID)pszOutBuffer, dwSize, &dwDownloaded))
                            {
                                delete[] pszOutBuffer;
                                break;
                            }

                            result += std::string(pszOutBuffer, dwDownloaded);
                            delete[] pszOutBuffer;

                        } while (dwSize > 0);

                        response.fullResponse = result;
                        response.success = true;

                        // Extract session ID if this is the first user
                        if (isFirstUser)
                        {
                            response.sessionId = extractSessionId(result);
                            std::cout << "Session ID extracted: " << response.sessionId << std::endl;
                        }

                        // Extract other relevant data
                        size_t gameStatusPos = result.find("\"gameStatus\":");
                        if (gameStatusPos != std::string::npos)
                        {
                            size_t statusStart = result.find("\"", gameStatusPos + 13);
                            if (statusStart != std::string::npos)
                            {
                                statusStart++;
                                size_t statusEnd = result.find("\"", statusStart);
                                if (statusEnd != std::string::npos)
                                {
                                    response.gameStatus = result.substr(statusStart, statusEnd - statusStart);
                                }
                            }
                        }

                        size_t currentTurnPos = result.find("\"currentTurnCharacterId\":");
                        if (currentTurnPos != std::string::npos)
                        {
                            size_t turnStart = result.find("\"", currentTurnPos + 25);
                            if (turnStart != std::string::npos)
                            {
                                turnStart++;
                                size_t turnEnd = result.find("\"", turnStart);
                                if (turnEnd != std::string::npos)
                                {
                                    response.currentTurnCharacterId = result.substr(turnStart, turnEnd - turnStart);
                                }
                            }
                        }
                    }
                }

                WinHttpCloseHandle(hRequest);
            }
            WinHttpCloseHandle(hConnect);
        }
        WinHttpCloseHandle(hSession);
    }

    return response;
}

// Function to check if cards are visible in ready character's area
bool checkCardsInReadyArea(const cv::Mat &frame)
{
    // This should be implemented based on your card detection logic
    // For now, returning a placeholder
    DetectionResultArray arr = Detect(frame);

    for (int i = 0; i < arr.size; i++)
    {
        // Check if detected object is a card and in the ready character's area
        // You'll need to define what constitutes the "ready character's area"
        // and how to detect cards vs other objects
        if (arr.results[i].classId == 1)
        { // Assuming 1 is card class ID
            // Check if card is in the ready area (you'll need to define this region)
            return true;
        }
    }

    return false;
}

// Function to check if dice are visible
bool checkDiceVisible(const cv::Mat &frame)
{
    DetectionResultArray arr = Detect(frame);

    for (int i = 0; i < arr.size; i++)
    {
        if (arr.results[i].classId == 0)
        { // Assuming 0 is dice class ID
            return true;
        }
    }

    return false;
}


void processUnrealCommand(cv::VideoCapture& cap)
{
    const char* detectionModelPath = "dtc.onnx";
    const char* classificationModelPath = "cls.onnx";

    if (!InitializeNetworks(detectionModelPath, classificationModelPath, true))
    {
        std::cerr << "Failed to initialize networks. Please check the model files and paths." << std::endl;
        return;
    }

    // Initialize Winsock
    if (!initializeWinsock())
    {
        std::cerr << "Failed to initialize Winsock" << std::endl;
        return;
    }

    // ESP8266 connection parameters
    std::string esp8266_ip = "192.168.1.100"; // Replace with actual ESP8266 IP
    int esp8266_port = 80;                    // Replace with actual port

    // Connect to ESP8266
    /*
    SOCKET esp_socket = connectToESP8266(esp8266_ip, esp8266_port);
    if (esp_socket == INVALID_SOCKET)
    {
        cleanupWinsock();
        return;
    }*/

    std::cout << "Waiting for two users to connect..." << std::endl;

    UserData users[2];
    APIResponse apiResponses[2];

    // Wait for first user
    std::cout << "Waiting for first user data..." << std::endl;
    bool firstUserReceived = false;
    while (!firstUserReceived)
    {
        // std::string receivedData = receiveFromESP8266(esp_socket);
        std::string receivedData = "token:eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODFhMDY0NjcxNmEzOGI0MDkyZmUzMDciLCJpYXQiOjE3NDgyOTI4MzQsImV4cCI6MTc0ODI5NjQzNH0.DA3gj0Iftu5S0kOekyzzgMyirsm9Qc1fNuKDF0T7Ub0,character:6834cc64a2a5277d477804d1\n"; // Mock data for testing
        if (!receivedData.empty())
        {
            users[0] = parseUserData(receivedData);
            if (!users[0].token.empty() && !users[0].characterId.empty())
            {
                std::cout << "First user received - Token: " << users[0].token
                    << ", Character ID: " << users[0].characterId << std::endl;

                // Make API call for first user (create session)
                apiResponses[0] = makeAPICall(users[0].token, users[0].characterId, true);
                if (apiResponses[0].success)
                {
                    std::cout << "First user API response successful!" << std::endl;
                    std::cout << "Session ID: " << apiResponses[0].sessionId << std::endl;
                    std::cout << "Game Status: " << apiResponses[0].gameStatus << std::endl;
                    std::cout << "Full response: " << apiResponses[0].fullResponse << std::endl;
                    firstUserReceived = true;
                }
                else
                {
                    std::cerr << "First user API call failed!" << std::endl;
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }



    // Wait for second user
    std::cout << "Waiting for second user data..." << std::endl;
    bool secondUserReceived = false;
    while (!secondUserReceived)
    {
        // std::string receivedData = receiveFromESP8266(esp_socket);
        std::string receivedData = "token:eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODFhMDY0NjcxNmEzOGI0MDkyZmUzMDciLCJpYXQiOjE3NDgyOTI4MzQsImV4cCI6MTc0ODI5NjQzNH0.DA3gj0Iftu5S0kOekyzzgMyirsm9Qc1fNuKDF0T7Ub0,character:6834cd22a2a5277d477804df\n"; // Mock data for testing
        if (!receivedData.empty())
        {
            users[1] = parseUserData(receivedData);
            if (!users[1].token.empty() && !users[1].characterId.empty())
            {
                std::cout << "Second user received - Token: " << users[1].token
                    << ", Character ID: " << users[1].characterId << std::endl;

                // Make API call for second user (join session)
                apiResponses[1] = makeAPICall(users[1].token, users[1].characterId, false, apiResponses[0].sessionId);
                if (apiResponses[1].success)
                {
                    std::cout << "Second user API response successful!" << std::endl;
                    std::cout << "Game Status: " << apiResponses[1].gameStatus << std::endl;
                    std::cout << "Full response: " << apiResponses[1].fullResponse << std::endl;
                    secondUserReceived = true;
                }
                else
                {
                    std::cerr << "Second user API call failed!" << std::endl;
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // return; // Exit early for now, as we are not implementing the full game monitoring logic yet
    // give mock data for testing

    // Tested to this point.
    
    std::cout << "Both users connected. Starting game monitoring..." << std::endl;

    // Setup calibration for monitoring
    cv::Point centers[] = { cv::Point(150, 150), cv::Point(1130, 570) };
    int radii[2] = { 60, 60 };
    setCalibrationProperties(centers, radii);

    // Setup ready properties for monitoring
    cv::Point readyCenter(100, 100);
    int readyRadius = 50;
    int secondsToWait = 3;
    setReadyProperties(readyCenter, readyRadius, secondsToWait);

    setupWindow();

    // Main monitoring loop
    while (true)
    {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty())
        {
            std::cerr << "Empty frame received!" << std::endl;
            break;
        }

        // Check all conditions with 2ms intervals
        bool calibrationCorrect = checkCalibrationApi(frame);
        bool playerReady = checkPlayerReady(frame);
        bool cardsVisible = checkCardsInReadyArea(frame);
        bool diceVisible = checkDiceVisible(frame);

        // Visual feedback on frame
        std::string calibrationStatus = calibrationCorrect ? "CALIBRATION: OK" : "CALIBRATION: FAIL";
        std::string readyStatus = playerReady ? "PLAYER: READY" : "PLAYER: NOT READY";
        std::string cardsStatus = cardsVisible ? "CARDS: VISIBLE" : "CARDS: NOT VISIBLE";
        std::string diceStatus = diceVisible ? "DICE: VISIBLE" : "DICE: NOT VISIBLE";

        cv::putText(frame, calibrationStatus, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
            calibrationCorrect ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, readyStatus, cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
            playerReady ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, cardsStatus, cv::Point(30, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7,
            cardsVisible ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, diceStatus, cv::Point(30, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7,
            diceVisible ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

        // Draw calibration circles
        for (size_t i = 0; i < 2; ++i)
        {
            cv::circle(frame, centers[i], radii[i], cv::Scalar(255, 0, 0), 2);
        }

        // Draw ready circle
        cv::circle(frame, readyCenter, readyRadius, cv::Scalar(0, 255, 0), 2);

        // Check if all conditions are met
        if (calibrationCorrect && playerReady && cardsVisible && diceVisible)
        {
            std::string allConditionsMet = "ALL CONDITIONS MET!";
            cv::putText(frame, allConditionsMet, cv::Point(30, 160), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 255, 255), 3);
            std::cout << "All conditions met at: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << "ms" << std::endl;
        }

        cv::imshow(WINDOW_NAME, frame);
        cv::resizeWindow(WINDOW_NAME, frame.cols, frame.rows);

        if (cv::waitKey(2) == 27)
            break; // Exit on 'ESC' key with 2ms interval

        // Sleep for 2ms as requested
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    // Cleanup
    // closesocket(esp_socket);
    cleanupWinsock();
}

int main(int argc, char *argv[])
{
    // Display current directory and its contents at startup
    listCurrentDirectory();

    if (argc > 1)
    {
        std::string command = argv[1];

        // List the available cameras
        std::cout << "Available cameras:" << std::endl;
        for (int i = 0; i < 10; ++i)
        {
            cv::VideoCapture testCap(i);
            if (testCap.isOpened())
            {
                std::cout << "Camera " << i << " is available." << std::endl;
                testCap.release();
            }
        }
        int cameraIndex = 0;
        if (argc == 3)
        {
            cameraIndex = std::stoi(argv[2]);
            std::cout << "Using camera index: " << cameraIndex << std::endl;
        }
        else
        {
            std::cout << "Using default camera index: 0" << std::endl;
        }

        cv::VideoCapture cap(cameraIndex);
        if (!cap.isOpened())
        {
            std::cerr << "Camera could not be opened!" << std::endl;
            return -1;
        }

        // Give camera time to initialize
        std::cout << "Initializing camera..." << std::endl;
        cv::Mat dummy;
        for (int i = 0; i < 10; i++)
        {
            cap >> dummy; // Discard initial frames that might be invalid
            cv::waitKey(100);
        }

        if (command == "--ready")
        {
            processReadyCommand(cap);
        }
        else if (command == "--dice")
        {
            processDiceCommand(cap);
        }
        else if (command == "--calibrate")
        {
            processCalibrateCommand(cap);
        }
        else if (command == "--unreal")
        {
            processUnrealCommand(cap);
        }
        else
        {
            std::cerr << "Invalid command: " << command << std::endl;
        }

        cap.release();
    }

    // Release GPU resources before exiting
    Cleanup();
    cv::destroyAllWindows();

    return 0;
}