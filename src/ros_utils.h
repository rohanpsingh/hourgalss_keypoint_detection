#ifndef ROS_UTILS_H
#define ROS_UTILS_H

#include "ros_headers.h"
#include "lua_utils.h"

void setDynParams(detection_hg::paramConfig &config, int level) {

    std::cout << "set dynamic params..." << std::endl;
    min_hm_thresh = config.hm_thresh;
    vis_kp_ind = config.vis_kp_ind;
    circle_rad = config.circle_rad;
    font_thick = config.font_thick;
    font_scale = config.font_scale;

    lua_pushnumber(L, min_hm_thresh);
    lua_setglobal(L, "min_hm_thresh");

    return;
}

bool serviceCallback(roseus::StringString::Request& request, roseus::StringString::Response& response){

    std::string filename = request.str;
    lua_getglobal(L, "loadModel");
    lua_pushstring(L, filename.c_str());
    int model_load = lua_pcall(L, 1, 0 ,0);
    if (model_load) {
        fprintf(stderr, "Failed to load model: %s\n", lua_tostring(L, -1));
        model_load_success = false;
	return false;
    }
    else {
        std::cout << "------model load success----- " << std::endl;
	model_load_success = true;
    }

    return true;
}

#endif // ROS_UTILS_H
