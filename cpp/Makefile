# Makefile for Darknet Weights Extractor

CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra -I./src
LDFLAGS = 

# Target executable
TARGET = weights_extractor

# Source files
SOURCES = src/main.cpp \
          src/cfg_parser.cpp \
          src/weights_reader.cpp \
          src/batch_norm_folder.cpp \
          src/int16_quantizer.cpp \
          src/activation_tracker.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Compile source files
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(OBJECTS) $(TARGET)
	@echo "Clean complete"

# Install (copy to parent directory for easy access)
install: $(TARGET)
	@echo "Installing $(TARGET)..."
	cp $(TARGET) ../$(TARGET)
	@echo "Installed to ../$(TARGET)"

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: clean $(TARGET)

# Show help
help:
	@echo "Available targets:"
	@echo "  all     - Build the project (default)"
	@echo "  clean   - Remove build artifacts"
	@echo "  debug   - Build with debug symbols"
	@echo "  install - Install executable to parent directory"
	@echo "  help    - Show this help message"

.PHONY: all clean install debug help

