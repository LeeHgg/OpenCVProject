# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.30.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.30.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/ggang/Documents/opencv-setup

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ggang/Documents/opencv-setup/build

# Include any dependencies generated for this target.
include CMakeFiles/MyProject.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/MyProject.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/MyProject.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MyProject.dir/flags.make

CMakeFiles/MyProject.dir/tutorial.cpp.o: CMakeFiles/MyProject.dir/flags.make
CMakeFiles/MyProject.dir/tutorial.cpp.o: /Users/ggang/Documents/opencv-setup/tutorial.cpp
CMakeFiles/MyProject.dir/tutorial.cpp.o: CMakeFiles/MyProject.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ggang/Documents/opencv-setup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MyProject.dir/tutorial.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/MyProject.dir/tutorial.cpp.o -MF CMakeFiles/MyProject.dir/tutorial.cpp.o.d -o CMakeFiles/MyProject.dir/tutorial.cpp.o -c /Users/ggang/Documents/opencv-setup/tutorial.cpp

CMakeFiles/MyProject.dir/tutorial.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/MyProject.dir/tutorial.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ggang/Documents/opencv-setup/tutorial.cpp > CMakeFiles/MyProject.dir/tutorial.cpp.i

CMakeFiles/MyProject.dir/tutorial.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/MyProject.dir/tutorial.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ggang/Documents/opencv-setup/tutorial.cpp -o CMakeFiles/MyProject.dir/tutorial.cpp.s

# Object files for target MyProject
MyProject_OBJECTS = \
"CMakeFiles/MyProject.dir/tutorial.cpp.o"

# External object files for target MyProject
MyProject_EXTERNAL_OBJECTS =

MyProject: CMakeFiles/MyProject.dir/tutorial.cpp.o
MyProject: CMakeFiles/MyProject.dir/build.make
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_gapi.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_stitching.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_alphamat.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_aruco.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_bgsegm.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_bioinspired.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_ccalib.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_dnn_objdetect.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_dnn_superres.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_dpm.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_face.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_freetype.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_fuzzy.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_hfs.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_img_hash.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_intensity_transform.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_line_descriptor.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_mcc.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_quality.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_rapid.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_reg.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_rgbd.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_saliency.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_sfm.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_signal.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_stereo.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_structured_light.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_superres.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_surface_matching.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_tracking.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_videostab.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_viz.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_wechat_qrcode.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_xfeatures2d.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_xobjdetect.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_xphoto.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_shape.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_highgui.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_datasets.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_plot.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_text.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_ml.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_phase_unwrapping.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_optflow.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_ximgproc.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_video.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_videoio.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_imgcodecs.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_objdetect.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_calib3d.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_dnn.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_features2d.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_flann.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_photo.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_imgproc.4.10.0.dylib
MyProject: /usr/local/Cellar/opencv/4.10.0_4/lib/libopencv_core.4.10.0.dylib
MyProject: CMakeFiles/MyProject.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/ggang/Documents/opencv-setup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable MyProject"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MyProject.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MyProject.dir/build: MyProject
.PHONY : CMakeFiles/MyProject.dir/build

CMakeFiles/MyProject.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MyProject.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MyProject.dir/clean

CMakeFiles/MyProject.dir/depend:
	cd /Users/ggang/Documents/opencv-setup/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ggang/Documents/opencv-setup /Users/ggang/Documents/opencv-setup /Users/ggang/Documents/opencv-setup/build /Users/ggang/Documents/opencv-setup/build /Users/ggang/Documents/opencv-setup/build/CMakeFiles/MyProject.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/MyProject.dir/depend

