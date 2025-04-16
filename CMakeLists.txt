cmake_minimum_required(VERSION 3.15)
project(slm_qa_system)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/ggml
)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(nlohmann_json 3.0 REQUIRED)

# Check for Poppler only if building genQA
option(BUILD_GENQA "Build genQA executable" ON)
if(BUILD_GENQA)
    pkg_check_modules(POPPLER REQUIRED poppler-cpp)
endif()

# Link directories (assuming llama.cpp is built in a subdirectory)
link_directories(
    ${CMAKE_SOURCE_DIR}/lib
)

# Add executables
add_executable(fineTuneSLM
    src/fineTuneSLM.cpp
)

add_executable(querySLM
    src/querySLM.cpp
)

if(BUILD_GENQA)
    add_executable(genQA
        src/genQA.cpp
    )
endif()

# Common link libraries for all targets
foreach(target IN ITEMS fineTuneSLM querySLM genQA)
    if(TARGET ${target})
        target_link_libraries(${target}
            ggml
            llama
            nlohmann_json::nlohmann_json
            stdc++fs
        )
        
        # Add Poppler only for genQA
        if(target STREQUAL "genQA" AND BUILD_GENQA)
            target_link_libraries(genQA ${POPPLER_LIBRARIES})
        endif()
        
        # Copy model files after build
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${target}>/models/
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_SOURCE_DIR}/models/t5-small-qa-qg-hl.gguf
                $<TARGET_FILE_DIR:${target}>/models/
        )
    endif()
endforeach()

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native)
endif()
