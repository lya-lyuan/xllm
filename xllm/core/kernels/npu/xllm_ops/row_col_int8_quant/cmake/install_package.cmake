# Copyright 2026 The xLLM Authors. All Rights Reserved.

cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED BUILD_DIR OR NOT DEFINED INSTALL_PREFIX)
  message(FATAL_ERROR "BUILD_DIR and INSTALL_PREFIX are required")
endif()

set(package_config "${BUILD_DIR}/CPackConfig.cmake")
if(NOT EXISTS "${package_config}")
  message(FATAL_ERROR "Missing package configuration: ${package_config}")
endif()
include("${package_config}")
if(NOT DEFINED CPACK_PACKAGE_FILE_NAME OR CPACK_PACKAGE_FILE_NAME STREQUAL "")
  message(FATAL_ERROR "CPack did not define CPACK_PACKAGE_FILE_NAME")
endif()

set(package_file "${BUILD_DIR}/${CPACK_PACKAGE_FILE_NAME}")
if(NOT EXISTS "${package_file}" OR IS_DIRECTORY "${package_file}")
  message(FATAL_ERROR "Missing generated operator package: ${package_file}")
endif()
execute_process(
  COMMAND bash "${package_file}" --quiet "--install-path=${INSTALL_PREFIX}"
  RESULT_VARIABLE install_status
)
if(NOT "${install_status}" STREQUAL "0")
  message(FATAL_ERROR "Operator package installation failed: ${install_status}")
endif()
