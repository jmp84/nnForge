USE_BOOST=yes
USE_OPENCV=yes
USE_OPENMP=yes
USE_CUDA=yes
USE_NNFORGE=yes

include ../../Settings.mk
BOOST_LIBS+=-lboost_program_options -lboost_log -lboost_log_setup -lboost_thread -lboost_filesystem -lboost_system
include ../../Main.mk
LDFLAGS+=-lboost_unit_test_framework

# TODO
#GENERIC_CXXFLAGS+=-pedantic -pedantic-errors -Wall -Wextra -Werror -Wconversion

APP_NAME=$(shell basename `pwd`)
TARGET=../../bin/$(APP_NAME)
TARGET_TEST=$(TARGET)_test
CONFIG_FILE=$(TARGET).cfg
SOURCES+=$(wildcard *.cpp)
TEST_DIR=test
SOURCES_TEST=$(SOURCES) $(wildcard $(TEST_DIR)/*.cpp)
SOURCES_TEST:=$(filter-out $(APP_NAME).cpp, $(SOURCES_TEST))
OBJECTS+=$(patsubst %.cpp,%.o,$(wildcard *.cpp))
OBJECTS_TEST=$(OBJECTS) $(patsubst %.cpp,%.o,$(wildcard $(TEST_DIR)/*.cpp))
OBJECTS_TEST:=$(filter-out $(APP_NAME).o, $(OBJECTS_TEST))

all: $(TARGET) $(CONFIG_FILE)

$(OBJECTS): $(SOURCES)

$(TARGET): $(OBJECTS) $(LDLIBSDEPEND)
	$(CXX) -o $(TARGET) $(OBJECTS) $(LDLIBSDEPEND) $(LDFLAGS)

$(CONFIG_FILE): config.cfg
	$(RM) $(CONFIG_FILE)
	echo 'input_data_folder=$(NNFORGE_INPUT_DATA_PATH)/$(APP_NAME)' >> $(CONFIG_FILE)
	echo 'working_data_folder=$(NNFORGE_WORKING_DATA_PATH)/$(APP_NAME)' >> $(CONFIG_FILE)
	cat config.cfg >> $(CONFIG_FILE)

test: $(TARGET_TEST)

$(OBJECTS_TEST): $(SOURCES_TEST)

$(TARGET_TEST): $(OBJECTS_TEST) $(LDLIBSDEPEND)
	$(CXX) -o $(TARGET_TEST) $(OBJECTS_TEST) $(LDLIBSDEPEND) $(LDFLAGS)

clean:
	$(RM) $(OBJECTS) $(OBJECTS_TEST) $(TARGET) $(TARGET_TEST) $(CONFIG_FILE)

.PHONY: all clean test
