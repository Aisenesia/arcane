#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <filesystem> // For checking file existence
#include <cstring>    // Add this for strcpy_s

// half of the height of the cap
int CAP_HEIGHT = 720;

#define USE_NFC 0
#define SKIP_LOGIN 1

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
#include <opencv2/dnn.hpp>       // For NMSBoxes
#include <opencv2/objdetect.hpp> // For QR code detection

// Local includes
#include "detection_types.h"
#include "network_manager.h"
#include "game_components.h"

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "winhttp.lib")

#define DICE_CLASS 0 // 0 is the class ID for dice, all other detections are different cards.
#define SERVER_IP "localhost"
#define SERVER_PORT 3001

// ESP8266 Configuration
#define ESP8266_DEFAULT_IP "192.168.1.184" // Default IP to try first
#define ESP8266_PORT 8080
#define ESP8266_CONNECTION_TIMEOUT 5000 // 5 seconds timeout// Example port for server communication

// Window management constants
const char *WINDOW_NAME = "Camera View";
const int WINDOW_PADDING = 50;           // Padding from screen edges
const double DEFAULT_SCALE_FACTOR = 0.8; // Default scale factor for video

char currentTurnPlayerId[32] = "";
std::string globalSessionId = "";

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
    cv::resizeWindow(WINDOW_NAME, 1600, 900);
}

void applyNonMaximumSuppression(const DetectionResultArray &arr, std::vector<cv::Rect> &filteredBoxes, std::vector<float> &confidences, float nmsThreshold = 0.4)
{
    std::vector<int> indices;
    filteredBoxes.clear();
    confidences.clear();

    for (int i = 0; i < arr.size; i++)
    {
        filteredBoxes.push_back(arr.results[i].boundingBox);
        confidences.push_back(arr.results[i].confidence);
    }

    // Apply Non-Maximum Suppression
    cv::dnn::NMSBoxes(filteredBoxes, confidences, 0.5, nmsThreshold, indices);

    // Filter out boxes based on NMS results
    std::vector<cv::Rect> nmsFilteredBoxes;
    std::vector<float> nmsFilteredConfidences;
    for (int idx : indices)
    {
        nmsFilteredBoxes.push_back(filteredBoxes[idx]);
        nmsFilteredConfidences.push_back(confidences[idx]);
    }

    filteredBoxes = nmsFilteredBoxes;
    confidences = nmsFilteredConfidences;
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
    std::string roomCode;
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

// Function to connect to ESP8266 with improved error handling
SOCKET connectToESP8266(const std::string &esp8266_ip, int port)
{
    std::cout << "Attempting to connect to ESP8266..." << std::endl;
    std::cout << "Target: " << esp8266_ip << ":" << port << std::endl;

    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET)
    {
        std::cerr << "❌ Socket creation failed. Error: " << WSAGetLastError() << std::endl;
        return INVALID_SOCKET;
    }

    // Set socket timeout
    DWORD timeout = ESP8266_CONNECTION_TIMEOUT;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeout, sizeof(timeout));

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);

    int result = inet_pton(AF_INET, esp8266_ip.c_str(), &serverAddr.sin_addr);
    if (result <= 0)
    {
        std::cerr << "❌ Invalid IP address format: " << esp8266_ip << std::endl;
        closesocket(sock);
        return INVALID_SOCKET;
    }

    std::cout << "Connecting..." << std::endl;
    if (connect(sock, (sockaddr *)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR)
    {
        int error = WSAGetLastError();
        std::cerr << "❌ Connection to ESP8266 failed" << std::endl;
        std::cerr << "   Error code: " << error << std::endl;

        switch (error)
        {
        case WSAETIMEDOUT:
            std::cerr << "   Reason: Connection timeout - ESP8266 not responding" << std::endl;
            break;
        case WSAECONNREFUSED:
            std::cerr << "   Reason: Connection refused - Check if ESP8266 server is running" << std::endl;
            break;
        case WSAEHOSTUNREACH:
            std::cerr << "   Reason: Host unreachable - Check IP address and network" << std::endl;
            break;
        default:
            std::cerr << "   Reason: Unknown network error" << std::endl;
        }

        std::cerr << std::endl;
        std::cerr << "Troubleshooting steps:" << std::endl;
        std::cerr << "1. Verify ESP8266 is powered on and connected to WiFi" << std::endl;
        std::cerr << "2. Check the IP address displayed on ESP8266 serial monitor" << std::endl;
        std::cerr << "3. Ensure ESP8266 and PC are on the same network" << std::endl;
        std::cerr << "4. Try pinging the ESP8266: ping " << esp8266_ip << std::endl;

        closesocket(sock);
        return INVALID_SOCKET;
    }

    std::cout << "✅ Connected to ESP8266 at " << esp8266_ip << ":" << port << std::endl;
    return sock;
}

// Function to communicate with ESP8266 (send request and receive data)
std::string receiveFromESP8266(SOCKET sock)
{
    // First, send a request to the ESP8266
    std::string request = "GET_USER_DATA\r\n";
    std::cout << "Sending request to ESP8266..." << std::endl;

    int bytesSent = send(sock, request.c_str(), request.length(), 0);
    if (bytesSent == SOCKET_ERROR)
    {
        std::cout << "❌ Failed to send request to ESP8266. Error: " << WSAGetLastError() << std::endl;
        return "";
    }

    std::cout << "✅ Request sent (" << bytesSent << " bytes)" << std::endl;
    std::cout << "Waiting for response from ESP8266..." << std::endl;

    // Now wait for response
    char buffer[2048] = {0}; // Increased buffer size

    int bytesReceived = recv(sock, buffer, sizeof(buffer) - 1, 0);
    if (bytesReceived > 0)
    {
        buffer[bytesReceived] = '\0'; // Null-terminate
        std::string receivedData(buffer, bytesReceived);

        std::cout << "✅ Received " << bytesReceived << " bytes from ESP8266" << std::endl;
        std::cout << "Response: " << receivedData << std::endl;

        return receivedData;
    }
    else if (bytesReceived == 0)
    {
        std::cout << "⚠️ ESP8266 closed the connection" << std::endl;
    }
    else
    {
        int error = WSAGetLastError();
        if (error == WSAETIMEDOUT)
        {
            std::cout << "⏱️ Receive timeout - ESP8266 not responding" << std::endl;
        }
        else
        {
            std::cout << "❌ Receive error: " << error << std::endl;
        }
    }

    return "";
}

// Function to parse user data from received string
UserData parseUserData(const std::string &data)
{
    UserData userData;

    std::cout << "Parsing received data: " << data << std::endl;

    // Check if data contains both users (ESP8266 format)
    if (data.find("user1:") != std::string::npos && data.find("user2:") != std::string::npos)
    {
        std::cout << "Detected ESP8266 dual-user format" << std::endl;

        // Parse user1 data
        size_t user1Start = data.find("user1:");
        size_t user1End = data.find("|user2:", user1Start);
        if (user1End == std::string::npos)
            user1End = data.length();

        std::string user1Data = data.substr(user1Start, user1End - user1Start);

        // Extract token and character from user1
        size_t tokenPos = user1Data.find("token:");
        size_t charPos = user1Data.find("character:");

        if (tokenPos != std::string::npos && charPos != std::string::npos)
        {
            size_t tokenStart = tokenPos + 6; // length of "token:"
            size_t tokenEnd = user1Data.find(";", tokenStart);
            if (tokenEnd == std::string::npos)
                tokenEnd = user1Data.length();

            size_t charStart = charPos + 10; // length of "character:"
            size_t charEnd = user1Data.find(";", charStart);
            if (charEnd == std::string::npos)
                charEnd = user1Data.length();

            userData.token = user1Data.substr(tokenStart, tokenEnd - tokenStart);
            userData.characterId = user1Data.substr(charStart, charEnd - charStart);

            std::cout << "Successfully parsed User 1:" << std::endl;
            std::cout << "  Token: " << userData.token.substr(0, 50) << "..." << std::endl;
            std::cout << "  Character: " << userData.characterId << std::endl;
        }
    }
    // Legacy format support
    else if (data.find("token:") != std::string::npos && data.find("character:") != std::string::npos)
    {
        std::cout << "Detected legacy single-user format" << std::endl;

        size_t tokenPos = data.find("token:");
        size_t charPos = data.find("character:");

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
    else
    {
        std::cout << "⚠️ Unknown data format received" << std::endl;
    }

    return userData;
}

// Function to parse second user data from ESP8266 dual-user format
UserData parseUser2Data(const std::string &data)
{
    UserData userData;

    std::cout << "Parsing User 2 from dual-user data..." << std::endl;

    // Check if data contains both users (ESP8266 format)
    if (data.find("user1:") != std::string::npos && data.find("user2:") != std::string::npos)
    {
        // Parse user2 data
        size_t user2Start = data.find("|user2:");
        if (user2Start == std::string::npos)
        {
            std::cout << "❌ User2 data not found" << std::endl;
            return userData;
        }

        std::string user2Data = data.substr(user2Start + 1); // Skip the "|"

        // Extract token and character from user2
        size_t tokenPos = user2Data.find("token:");
        size_t charPos = user2Data.find("character:");

        if (tokenPos != std::string::npos && charPos != std::string::npos)
        {
            size_t tokenStart = tokenPos + 6; // length of "token:"
            size_t tokenEnd = user2Data.find(";", tokenStart);
            if (tokenEnd == std::string::npos)
                tokenEnd = user2Data.length();

            size_t charStart = charPos + 10; // length of "character:"
            size_t charEnd = user2Data.find(";", charStart);
            if (charEnd == std::string::npos)
                charEnd = user2Data.length();

            userData.token = user2Data.substr(tokenStart, tokenEnd - tokenStart);
            userData.characterId = user2Data.substr(charStart, charEnd - charStart);

            std::cout << "Successfully parsed User 2:" << std::endl;
            std::cout << "  Token: " << userData.token.substr(0, 50) << "..." << std::endl;
            std::cout << "  Character: " << userData.characterId << std::endl;
        }
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
                idStart++;                                       // Move past the opening quote
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
                std::cout << "First user detected, creating new session..." << std::endl;
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
            { // Prepare request data (JSON body)
                std::string postData = "{\"characterId\":\"" + characterId + "\"}";

                // Prepare headers - WinHTTP requires specific header format
                std::string authHeader = "Authorization: Bearer " + token;
                std::wstring wAuthHeader = std::wstring(authHeader.begin(), authHeader.end());
                std::wstring contentTypeHeader = L"Content-Type: application/json";

                std::cout << "Sending Authorization header: " << authHeader.substr(0, 50) << "..." << std::endl;
                std::cout << "Token length: " << token.length() << " characters" << std::endl;

                // Add Authorization header
                if (!WinHttpAddRequestHeaders(hRequest, wAuthHeader.c_str(), -1, WINHTTP_ADDREQ_FLAG_ADD))
                {
                    std::cerr << "Failed to add Authorization header. Error: " << GetLastError() << std::endl;
                }

                // Add Content-Type header
                if (!WinHttpAddRequestHeaders(hRequest, contentTypeHeader.c_str(), -1, WINHTTP_ADDREQ_FLAG_ADD))
                {
                    std::cerr << "Failed to add Content-Type header. Error: " << GetLastError() << std::endl;
                }
                BOOL bResults = WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                                   (LPVOID)postData.c_str(), postData.length(),
                                                   postData.length(), 0);

                if (bResults)
                {
                    bResults = WinHttpReceiveResponse(hRequest, NULL);

                    if (bResults)
                    {
                        // Check HTTP status code
                        DWORD statusCode = 0;
                        DWORD statusCodeSize = sizeof(statusCode);
                        if (WinHttpQueryHeaders(hRequest,
                                                WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                                                WINHTTP_HEADER_NAME_BY_INDEX,
                                                &statusCode,
                                                &statusCodeSize,
                                                WINHTTP_NO_HEADER_INDEX))
                        {
                            std::cout << "HTTP Status Code: " << statusCode << std::endl;
                        }
                        else
                        {
                            std::cerr << "Failed to query HTTP status code" << std::endl;
                        }

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

                        // Check if the status code indicates success
                        if (statusCode == 200 || statusCode == 201)
                        {
                            response.success = true;
                        }
                        else
                        {
                            response.success = false;
                            std::cerr << "API call failed with HTTP status code: " << statusCode << std::endl;
                            std::cerr << "Expected status code 200 or 201." << std::endl;
                            std::cerr << "Full server response:" << std::endl;
                            std::cerr << result << std::endl;
                        } // Extract session ID if this is the first user and request was successful
                        if (isFirstUser && response.success)
                        {
                            response.sessionId = extractSessionId(result);
                        }

                        // Extract other relevant data only if request was successful
                        if (response.success)
                        {
                            // Extract gameStatus
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

                            // Extract roomCode
                            size_t roomCodePos = result.find("\"roomCode\":");
                            if (roomCodePos != std::string::npos)
                            {
                                size_t codeStart = result.find("\"", roomCodePos + 11);
                                if (codeStart != std::string::npos)
                                {
                                    codeStart++;
                                    size_t codeEnd = result.find("\"", codeStart);
                                    if (codeEnd != std::string::npos)
                                    {
                                        response.roomCode = result.substr(codeStart, codeEnd - codeStart);
                                    }
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
    DetectionResultArray arr = NetworkManager::detect(frame);

    for (int i = 0; i < arr.size; i++)
    {
        if (arr.results[i].classId != DICE_CLASS) // Any non-dice class (cards)
        {
            return true;
        }
    }

    return false;
}

// Function to check if dice are visible
bool checkDiceVisible(const cv::Mat &frame)
{
    DetectionResultArray arr = NetworkManager::detect(frame);

    for (int i = 0; i < arr.size; i++)
    {
        if (arr.results[i].classId == DICE_CLASS) // Dice class ID
        {
            return true;
        }
    }

    return false;
}

// Function to receive user data from QR code using camera
std::string receiveFromQRCode(cv::VideoCapture &cap, int userNumber)
{
    if (userNumber > 0)
    {
        std::cout << "\n🔄 QR Code Mode: Waiting for User " << userNumber << " QR code..." << std::endl;
        std::cout << "User " << userNumber << ": Please show your QR code to the camera" << std::endl;
    }
    else
    {
        std::cout << "\n🔄 QR Code Mode: Waiting for QR code..." << std::endl;
        std::cout << "Please show a QR code with user data to the camera" << std::endl;
    }
    std::cout << "Expected format: token:YOUR_TOKEN,character:YOUR_CHARACTER_ID" << std::endl;
    std::cout << "Press ESC to exit QR code scanning" << std::endl;

    cv::QRCodeDetector qrDetector;

    while (true)
    {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty())
        {
            std::cerr << "Empty frame received!" << std::endl;
            continue;
        }
        // Display the frame with instructions
        cv::Mat displayFrame = frame.clone();
        if (userNumber > 0)
        {
            cv::putText(displayFrame, "QR Code Scanner - User " + std::to_string(userNumber) + " Login",
                        cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }
        else
        {
            cv::putText(displayFrame, "QR Code Scanner - Show QR code to camera",
                        cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }
        cv::putText(displayFrame, "Press ESC to exit",
                    cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
        // Try to detect and decode QR code
        std::string decodedText;
        std::vector<cv::Point> points;

        decodedText = qrDetector.detectAndDecode(frame, points);
        bool qrDetected = !decodedText.empty();

        if (qrDetected && !decodedText.empty())
        {
            std::cout << "✅ QR Code detected!" << std::endl;
            std::cout << "Decoded text: " << decodedText << std::endl;

            // Draw QR code boundary
            if (points.size() == 4)
            {
                for (int i = 0; i < 4; i++)
                {
                    cv::line(displayFrame, points[i], points[(i + 1) % 4], cv::Scalar(0, 255, 0), 3);
                }
            }            cv::putText(displayFrame, "QR Code Found!",
                        cv::Point(30, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

            cv::imshow(WINDOW_NAME, displayFrame);
            
            // Check and save window size before showing result
            // checkAndSaveWindowSize();
            
            cv::waitKey(2000); // Show result for 2 seconds

            return decodedText;
        }
        else
        {
            cv::putText(displayFrame, "Scanning for QR code...",
                        cv::Point(30, 90), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
        }        cv::imshow(WINDOW_NAME, displayFrame);
        
        // Check and save window size if changed (periodically)
        // checkAndSaveWindowSize();

        // Check for ESC key
        if (cv::waitKey(30) == 27)
        {
            std::cout << "QR code scanning cancelled by user" << std::endl;
            return "";
        }
    }
}

void processUnrealCommand(cv::VideoCapture &cap)
{

    std::cout << "\n"
              << std::string(60, '=') << std::endl;
    if (USE_NFC)
    {
        std::cout << "           UNREAL COMMAND - ESP8266 MODE" << std::endl;
    }
    else
    {
        std::cout << "           UNREAL COMMAND - QR CODE MODE" << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;

    const char *detectionModelPath = "dtc.onnx";
    const char *classificationModelPath = "cls.onnx";

    if (!NetworkManager::initializeNetworks(detectionModelPath, classificationModelPath, true))
    {
        std::cerr << "❌ Failed to initialize networks. Please check the model files and paths." << std::endl;
        return;
    }

    if (!SKIP_LOGIN)
    {

        // ESP8266 setup only if using NFC
        SOCKET esp_socket = INVALID_SOCKET;
        if (USE_NFC)
        {
            // Initialize Winsock
            if (!initializeWinsock())
            {
                std::cerr << "❌ Failed to initialize Winsock" << std::endl;
                return;
            }

            // ESP8266 connection parameters with user input
            std::string esp8266_ip;
            std::cout << "\nESP8266 Connection Setup:" << std::endl;
            std::cout << "Please check your ESP8266 serial monitor for the actual IP address." << std::endl;
            std::cout << "Enter ESP8266 IP address (or press Enter for default " << ESP8266_DEFAULT_IP << "): ";

            std::getline(std::cin, esp8266_ip);
            if (esp8266_ip.empty())
            {
                esp8266_ip = ESP8266_DEFAULT_IP;
                std::cout << "Using default IP: " << esp8266_ip << std::endl;
            }

            int esp8266_port = ESP8266_PORT;
            std::cout << "Using port: " << esp8266_port << std::endl;

            // Connect to ESP8266 with retry logic
            int connectionAttempts = 0;
            const int maxAttempts = 3;

            while (esp_socket == INVALID_SOCKET && connectionAttempts < maxAttempts)
            {
                connectionAttempts++;
                std::cout << "\n--- Connection Attempt " << connectionAttempts << "/" << maxAttempts << " ---" << std::endl;

                esp_socket = connectToESP8266(esp8266_ip, esp8266_port);

                if (esp_socket == INVALID_SOCKET)
                {
                    if (connectionAttempts < maxAttempts)
                    {
                        std::cout << "Retrying in 3 seconds..." << std::endl;
                        Sleep(3000);
                    }
                }
            }

            if (esp_socket == INVALID_SOCKET)
            {
                std::cout << "\n❌ Failed to connect after " << maxAttempts << " attempts." << std::endl;
                std::cout << "Please check your ESP8266 setup and try again." << std::endl;
                cleanupWinsock();
                return;
            }

            std::cout << "\n🔄 Waiting for user data from ESP8266..." << std::endl;
            std::cout << "The ESP8266 will generate mock data automatically after 5 seconds." << std::endl;
        }
        else
        {
            std::cout << "\n🔄 QR Code Mode: Ready to scan for user data..." << std::endl;
            std::cout << "Camera will be used to scan QR codes containing user tokens and character IDs." << std::endl;
        }
        UserData users[2];
        APIResponse apiResponses[2];

        // Sequential user login implementation
        bool user1LoggedIn = false;
        bool user2LoggedIn = false;

        // Step 1: First user login
        std::cout << "\n"
                  << std::string(50, '=') << std::endl;
        std::cout << "STEP 1: FIRST USER LOGIN" << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        if (USE_NFC)
        {
            std::cout << "Waiting for first user data from ESP8266..." << std::endl;
        }
        else
        {
            std::cout << "First user: Please scan your QR code to log in..." << std::endl;
        }
        while (!user1LoggedIn)
        {
            std::string receivedData;
            if (USE_NFC)
            {
                receivedData = receiveFromESP8266(esp_socket);
            }
            else
            {
                receivedData = receiveFromQRCode(cap, 1); // User 1
            }

            if (!receivedData.empty())
            {
                // Check if ESP8266 is still waiting for users
                if (receivedData.find("WAITING:") == 0)
                {
                    std::cout << "ESP8266 status: " << receivedData << std::endl;
                    std::cout << "Waiting for mock data generation..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    continue;
                }

                // Parse user data - for single user QR codes, use parseUserData only
                UserData user1Data = parseUserData(receivedData);

                // Check if we got valid user data
                bool user1Valid = !user1Data.token.empty() && !user1Data.characterId.empty();
                if (user1Valid)
                {
                    std::cout << "\n✅ First user data received!" << std::endl;
                    std::cout << "User 1 - Token: " << user1Data.token.substr(0, 50) << "..., Character: " << user1Data.characterId << std::endl;
                    std::cout << "Full token length: " << user1Data.token.length() << " characters" << std::endl;

                    // Make API call for first user (create session)
                    std::cout << "\n🌐 Making API call for User 1 (create session)..." << std::endl;
                    apiResponses[0] = makeAPICall(user1Data.token, user1Data.characterId, true);
                    if (apiResponses[0].success)
                    {
                        std::cout << "✅ User 1 API response successful!" << std::endl;
                        std::cout << "Session ID: " << apiResponses[0].sessionId << std::endl;

                        std::cout << "Game Status: " << apiResponses[0].gameStatus << std::endl;
                        strcpy_s(currentTurnPlayerId, apiResponses[0].currentTurnCharacterId.c_str());
                        globalSessionId = apiResponses[0].sessionId;
                        // User 1 is now officially logged in
                        users[0] = user1Data;
                        user1LoggedIn = true;
                        std::cout << "\n🎉 User 1 successfully logged in!" << std::endl;
                    }
                    else
                    {
                        std::cerr << "❌ User 1 API call failed! Please try again." << std::endl;
                        std::cerr << "Only users with successful API responses (200/201) are considered logged in." << std::endl;
                        if (!apiResponses[0].fullResponse.empty())
                        {
                            std::cerr << "Full server response:" << std::endl;
                            std::cerr << apiResponses[0].fullResponse << std::endl;
                        }
                    }
                }
                else
                {
                    std::cout << "⚠️ Invalid user data received. Please scan a valid QR code..." << std::endl;
                }
            }

            if (!user1LoggedIn)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }

        // Step 2: Second user login (only after first user is logged in)
        std::cout << "\n"
                  << std::string(50, '=') << std::endl;
        std::cout << "STEP 2: SECOND USER LOGIN" << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        if (USE_NFC)
        {
            std::cout << "Waiting for second user data from ESP8266..." << std::endl;
            std::cout << "The ESP8266 will generate second user data automatically." << std::endl;
        }
        else
        {
            std::cout << "Second user: Please scan your QR code to join the session..." << std::endl;
            std::cout << "Expected format: token:YOUR_TOKEN,character:YOUR_CHARACTER_ID" << std::endl;
        }
        while (!user2LoggedIn)
        {
            std::string receivedData;
            if (USE_NFC)
            {
                receivedData = receiveFromESP8266(esp_socket);
            }
            else
            {
                receivedData = receiveFromQRCode(cap, 2); // User 2
            }

            if (!receivedData.empty())
            {
                // Check if ESP8266 is still waiting for users
                if (receivedData.find("WAITING:") == 0)
                {
                    
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    continue;
                }

                // Parse user data - for single user QR codes, use parseUserData only
                UserData user2Data = parseUserData(receivedData);

                // Check if we got valid user data
                bool user2Valid = !user2Data.token.empty() && !user2Data.characterId.empty();

                if (user2Valid)
                {
                    std::cout << "\n✅ Second user data received!" << std::endl;
                    std::cout << "User 2 - Token: " << user2Data.token.substr(0, 50) << "..., Character: " << user2Data.characterId << std::endl;

                    // Make API call for second user (join session)
                    std::cout << "\n🌐 Making API call for User 2 (join session)..." << std::endl;
                    apiResponses[1] = makeAPICall(user2Data.token, user2Data.characterId, false, apiResponses[0].sessionId);
                    if (apiResponses[1].success)
                    {
                        std::cout << "✅ User 2 API response successful!" << std::endl;
                        std::cout << "Game Status: " << apiResponses[1].gameStatus << std::endl;

                        // User 2 is now officially logged in
                        users[1] = user2Data;
                        user2LoggedIn = true;
                        std::cout << "\n🎉 User 2 successfully logged in!" << std::endl;
                        std::cout << "\n🎮 Both users successfully connected to game session!" << std::endl;
                    }
                    else
                    {
                        std::cerr << "❌ User 2 API call failed! Please try again." << std::endl;
                        std::cerr << "Only users with successful API responses (200/201) are considered logged in." << std::endl;
                        if (!apiResponses[1].fullResponse.empty())
                        {
                            std::cerr << "Full server response:" << std::endl;
                            std::cerr << apiResponses[1].fullResponse << std::endl;
                        }
                    }
                }
                else
                {
                    std::cout << "⚠️ Invalid user data received. Please scan a valid QR code..." << std::endl;
                }
            }

            if (!user2LoggedIn)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }

        // return; // Exit early for now, as we are not implementing the full game monitoring logic yet
        // give mock data for testing
    }
    // Tested to this point.

    std::cout << "Both users connected. Starting game monitoring..." << std::endl;
    std::cout << "Session ID: " << globalSessionId << std::endl;

    // Setup calibration for monitoring
    cv::Point centers[] = {cv::Point(135, 120), cv::Point(1783, 938)};
    int radii[2] = {72, 72};

    // Create calibration checker instance
    std::vector<cv::Point> centerVec = {centers[0], centers[1]};
    std::vector<int> radiiVec = {radii[0], radii[1]};
    CalibrationChecker calibrationChecker(centerVec, radiiVec);

    // Setup ready properties for monitoring. centered at the mıddle of the screen
    cv::Point readyCenter(800, 120);
    int readyRadius = 100;
    int secondsToWait = 3;

    // Create player ready checker instance
    PlayerReady playerReadyChecker(readyRadius, readyCenter, secondsToWait);

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
        } // Check all conditions with 2ms intervals
        calibrationChecker.setFrame(frame);
        playerReadyChecker.setFrame(frame);

        bool calibrationCorrect = calibrationChecker.checkCalibration();
        bool playerReady = playerReadyChecker.checkReady();       

        cv::Mat resizedFrame = frame.clone();
        // Cut the frame centered at the middle: 500px left + 500px right = 1000px total width, full height
        int cropWidth = 1000;  // 500px left + 500px right from center
        int cropHeight = frame.rows;  // Keep full height (900px)
        int xOffset = (frame.cols - cropWidth) / 2;  // Center horizontally
        int yOffset = 0;  // No vertical offset, keep full height
        cv::Rect cropRegion(xOffset, yOffset, cropWidth, cropHeight);
        resizedFrame = resizedFrame(cropRegion);

        // draw the crop region on the frame
        cv::rectangle(frame, cropRegion, cv::Scalar(0, 255, 255), 2);

        DetectionResultArray detectionResults = NetworkManager::detect(resizedFrame);

        // Separate dice and card detections
        std::vector<DetectionResult> cardDetections;
        std::vector<DetectionResult> diceDetections;
        
        for (int i = 0; i < detectionResults.size; i++)
        {
            // This is a dice detection - crop and classify
            cv::Rect diceBox = detectionResults.results[i].boundingBox;

            diceBox.x += xOffset; // Add crop region's starting x position
            diceBox.y += yOffset; // Add crop region's starting y position (though yOffset is 0 in this case)


            if (detectionResults.results[i].classId == DICE_CLASS)
            {
                
                
                
                cv::Mat diceCrop = frame(diceBox);
                
                // Classify the dice face
                ClassificationResult classResult = NetworkManager::classify(diceCrop);
                
                // Apply confidence filtering: only accept dice face classifications > 0.9 confidence
                if (classResult.confidence > 0.9)
                {
                    // Create new detection result with classification class
                    DetectionResult diceResult;
                    diceResult.boundingBox = diceBox;
                    diceResult.classId = classResult.classId;
                    diceResult.confidence = classResult.confidence;
                    
                    diceDetections.push_back(diceResult);
                    
                    
                }
            }
            else
            {
                // This is a card detection - keep detection class
                cardDetections.push_back(detectionResults.results[i]);
            }
        }

        // Visual feedback on frame
        std::string calibrationStatus = calibrationCorrect ? "CALIBRATION: OK" : "CALIBRATION: FAIL";
        std::string readyStatus = playerReady ? "PLAYER: READY" : "PLAYER: NOT READY";

        cv::putText(frame, calibrationStatus, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    calibrationCorrect ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, readyStatus, cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    playerReady ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

        if(calibrationCorrect)
        {
            // dump all detections and classifications to console in one line
            std::string readyStatus = playerReady ? "READY" : "NOT READY";
            std::cout << readyStatus << "Detections: ";
            for (const auto& card : cardDetections)
            {
                std::cout << "CARD:" << card.classId << " (" << std::fixed << std::setprecision(2) << card.confidence << "), ";
            }
            for (const auto& dice : diceDetections)
            {
                std::cout << "DICE:" << dice.classId << " (" << std::fixed << std::setprecision(2) << dice.confidence << "), ";
            }
        }

        // Draw calibration circles
        for (size_t i = 0; i < 2; ++i)
        {
            cv::circle(frame, centers[i], radii[i], cv::Scalar(255, 0, 0), 2);
        }        // Draw ready circle
        cv::circle(frame, readyCenter, readyRadius, cv::Scalar(0, 255, 0), 2);

        // Draw card detections (using detection class)
        for (const auto& card : cardDetections)
        {
            cv::Scalar cardColor = cv::Scalar(255, 0, 0); // Red for cards
            cv::rectangle(frame, card.boundingBox, cardColor, 2);
            
            std::string cardLabel = "CARD ID:" + std::to_string(card.classId) + " (" + std::to_string(card.confidence).substr(0, 4) + ")";
            cv::putText(frame, cardLabel, cv::Point(card.boundingBox.x, card.boundingBox.y - 10), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cardColor, 2);
        }
        
        // Draw dice detections (using classification class)
        for (const auto& dice : diceDetections)
        {
            cv::Scalar diceColor = cv::Scalar(0, 255, 0); // Green for dice
            cv::rectangle(frame, dice.boundingBox, diceColor, 2);
            
            std::string diceLabel = "DICE FACE:" + std::to_string(dice.classId) + " (" + std::to_string(dice.confidence).substr(0, 4) + ")";
            cv::putText(frame, diceLabel, cv::Point(dice.boundingBox.x, dice.boundingBox.y - 10), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, diceColor, 2);
        }

        // Check if all conditions are met
        if (calibrationCorrect && playerReady)
        {
            std::string allConditionsMet = "ALL CONDITIONS MET!";
            cv::putText(frame, allConditionsMet, cv::Point(30, 160), cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(0, 255, 255), 3);
            std::cout << "All conditions met at: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << "ms" << std::endl;
        }        cv::imshow(WINDOW_NAME, frame);
        
        // Check and save window size if changed (periodically)
        // checkAndSaveWindowSize();

        if (cv::waitKey(2) == 27)
            break; // Exit on 'ESC' key with 2ms interval

        // Sleep for 2ms as requested
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    } // Cleanup
    if (USE_NFC)
    {
        // closesocket(esp_socket);
        cleanupWinsock();
    }
}

// Function declarations
void setupWindow();
void processUnrealCommand(cv::VideoCapture &cap);
bool checkCardsInReadyArea(const cv::Mat &frame);
bool checkDiceVisible(const cv::Mat &frame);
std::string receiveFromQRCode(cv::VideoCapture &cap, int userNumber = 0);

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
            // processReadyCommand(cap);
        }
        else if (command == "--dice")
        {
            // processDiceCommand(cap);
        }
        else if (command == "--calibrate")
        {
            // processCalibrateCommand(cap);
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
    }    // Save final window size before exit

    
    // Release GPU resources before exiting
    NetworkManager::cleanup();
    cv::destroyAllWindows();

    return 0;
}