
$(foreach setting, $(shell ../GET_PROJECT_SETTINGS.sh --makefile --id_prefix IFX_), $(eval $(setting)))
  
SWERV_ISS_HOME:=$(IFX_TOOLSPREFIX)/swerviss-$(IFX_SWERVISS_VERSION))
ETISS_HOME:=$(IFX_TOOLSPREFIX)/etiss-$(IFX_ETISS_VERSION)
PREFIX:=$(IFX_TOOLSPREFIX)/tflite_u-${IFX_TFLITE_MICRO_VERSION}

# Here's the real Makefile....
include tensorflow/lite/micro/tools/make/Makefile


show_project_targets:
	@echo $(ALL_PROJECT_TARGETS)
	
test_executables: $(MICROLITE_BUILD_TARGETS)

TESTING_TEMPLS:=$(wildcard $(TESTING_DIR)/*.sh.tpl)

TEST_SCRIPTS:=$(patsubst $(TESTING_DIR)/%.sh.tpl,$(MAKEFILE_DIR)/gen/%,$(TESTING_TEMPLS))

# Note that third-party download libraries are copied into place
# as they appear in a sub-directory of tools/make
install:  installed_settings $(TEST_SCRIPTS)
	mkdir -p $(PREFIX)
	cp --parent $(MICROLITE_CC_SRCS) $(MICROLITE_CC_HDRS)  $(PREFIX)
	cp -r --parent tensorflow/lite/micro/testing $(PREFIX)
	cd $(PROJECT_DIR) ; cp -r --parent tools/make $(PREFIX)

clear_installation:
	rm -rf $(PREFIX)
	mkdir -p $(PREFIX)

installed_settings: 
	mkdir -p $(PREFIX)/tools/make
	echo 'TFLITE_U_PATH := $$(abspath $$(here)../..)/' >  $(PREFIX)/tools/make/installed_settings.inc
	echo 'PULPINO_CRT ?= $(ETISS_HOME)/crt_cmake' >> $(PREFIX)/tools/make/installed_settings.inc


 $(MAKEFILE_DIR)/gen/%.sh: $(TESTING_DIR)/%.sh.tpl
	mkdir -p  $(MAKEFILE_DIR)/gen
	sed -e'1,$$s^@SWERV_ISS_HOME@^$(SWERV_ISS_HOME)^' -e'1,$$s^@ETISS_HOME@^$(ETISS_HOME)^' "$<" > "$@"
	chmod +x "$@"


