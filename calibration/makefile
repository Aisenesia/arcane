CC = g++
CFLAGS = -std=c++17 `pkg-config --cflags --libs opencv4`
TARGET = calibration_checker
SOURCES = main.cpp CalibrationChecker.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)