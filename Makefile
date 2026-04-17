CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra
APP := planet

RAYLIB_CFLAGS := $(shell pkg-config --cflags raylib 2>/dev/null)
RAYLIB_LIBS := $(shell pkg-config --libs raylib 2>/dev/null)

all: $(APP)

$(APP): main.c
	$(CC) $(CFLAGS) $(RAYLIB_CFLAGS) main.c -o $(APP) $(RAYLIB_LIBS) -lm

clean:
	rm -f $(APP)

.PHONY: all clean
