# Star Wave Rave by Team210 - 64k Demo at Vortex III 2k18
# Copyright (C) 2018  Alexander Kraus <nr4@z10.info>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 2.8)

project(fontdemo)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(WIN32)
    find_program(SHADER_MINIFIER shader_minifier.exe HINTS ${PROJECT_BINARY_DIR})
    find_program(KKRUNCHY_K7 kkrunchy_k7.exe HINTS ${PROJECT_BINARY_DIR})
    
    add_custom_target(minification
        COMMAND ${SHADER_MINIFIER} "${PROJECT_SOURCE_DIR}/gfx.frag" -o "${PROJECT_SOURCE_DIR}/gfx.h"  --preserve-all-globals --no-renaming --preserve-externals
        COMMAND ${SHADER_MINIFIER}  -o "${PROJECT_SOURCE_DIR}/sfx.h" "${PROJECT_SOURCE_DIR}/sfx.frag" --preserve-externals
    )
    
    add_definitions(/Gy /O1 /fp:fast /GR- /GS- /MT)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:demo /SUBSYSTEM:Windows  /OPT:REF  /OPT:ICF=10 /VERBOSE /NODEFAULTLIB") 
    add_executable(swr swr.c)
    
    # for debug output: User32.lib ucrt.lib msvcrt.lib
    target_link_libraries(swr opengl32.lib Winmm.lib User32.lib ucrt.lib msvcrt.lib)
    
    add_dependencies(swr minification)
    add_custom_command(TARGET swr COMMAND ${KKRUNCHY_K7} --best ${CMAKE_CURRENT_BINARY_DIR}/Release/swr.exe --out ${CMAKE_CURRENT_BINARY_DIR}/Release/swrc.exe POST_BUILD)
endif()

