CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter -Wno-deprecated-enum-enum-conversion
APP := planet

RAYLIB_CFLAGS := $(shell pkg-config --cflags raylib 2>/dev/null)
RAYLIB_LIBS := $(shell pkg-config --libs raylib 2>/dev/null)

IMGUI_DIR   := vendor/imgui
IMPLOT_DIR  := vendor/implot
RLIMGUI_DIR := vendor/rlImGui

INCLUDES := -I$(IMGUI_DIR) -I$(IMPLOT_DIR) -I$(RLIMGUI_DIR)
DEFINES  := -DNO_FONT_AWESOME

IMGUI_SRC := \
	$(IMGUI_DIR)/imgui.cpp \
	$(IMGUI_DIR)/imgui_draw.cpp \
	$(IMGUI_DIR)/imgui_widgets.cpp \
	$(IMGUI_DIR)/imgui_tables.cpp \
	$(IMGUI_DIR)/imgui_demo.cpp \
	$(IMPLOT_DIR)/implot.cpp \
	$(IMPLOT_DIR)/implot_items.cpp \
	$(IMPLOT_DIR)/implot_demo.cpp \
	$(RLIMGUI_DIR)/rlImGui.cpp

SOURCES := main.cpp $(IMGUI_SRC)

FRAMEWORKS :=
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
FRAMEWORKS := -framework Cocoa -framework IOKit -framework OpenGL
endif

all: $(APP)

$(APP): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(DEFINES) $(RAYLIB_CFLAGS) $(INCLUDES) $(SOURCES) -o $(APP) $(RAYLIB_LIBS) $(FRAMEWORKS) -lm

clean:
	rm -f $(APP)

.PHONY: all clean
