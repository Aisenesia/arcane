# Derleyici
CXX = g++

# Derleyici seçenekleri
CXXFLAGS = -std=c++17 -Wall

# OpenCV bayrakları
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

# Hedef dosya
TARGET = card_app

# Kaynak dosyalar
SRCS = main.cpp CardHandler.cpp CardClassifier.cpp

# Nesne dosyaları
OBJS = $(SRCS:.cpp=.o)

# Varsayılan hedef
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_FLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< $(OPENCV_FLAGS)

clean:
	rm -f $(OBJS) $(TARGET)
