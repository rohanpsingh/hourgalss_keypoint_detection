#include <ros/ros.h>
#include <iostream>
#include <boost/thread/thread.hpp>
#include <Eigen/Dense>

#include <darknet_ros_msgs/BoundingBoxes.h>
#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

//#include <detection-hg/sol.hpp>
//#include <detection-hg/assert.hpp>

extern "C" {
    #include <lua.h>
    #include <lualib.h>
    #include <lauxlib.h>
}
#include <luaT.h>
//#include <TH/THStorage.h>
//#include <TH/THTensor.h>
#include <TH/TH.h>

//sol::state lua;
lua_State *L;

void msgCallback(const sensor_msgs::ImageConstPtr& img, const darknet_ros_msgs::BoundingBoxesConstPtr& box){

    cv::Mat read_image, image;
    read_image = cv::imread("/home/rohan/object3d/demo/yellow_tool1/images/frame00001.jpg", cv::IMREAD_UNCHANGED);
    read_image.convertTo(image, CV_32FC3);
    cv::Mat img_bgr[3];
    cv::split(image, img_bgr);


    return;
}


int main (int argc, char** argv){

    ros::init(argc, argv, "detect");
    ros::NodeHandle priv_nh("~");

    std::cout << "------running lua----- " << std::endl;
//    lua.open_libraries(sol::lib::base, sol::lib::package);
//    auto result = lua.script_file("lua/init.lua");
//    c_assert(!result.valid());
//    sol::load_result script1 = lua.load_file("lua/init.lua");
//    sol::protected_function_result res = script1();
//    if(res.valid()) {
//        std::cout << "valid" << std::endl;
//    }
//    else {
//        std::cout << "invalid" << std::endl;
//    }
//    luaL_dostring(lua, "require 'paths'");
//    luaL_dofile(lua, "lua/img.lua");
//    luaL_dofile(lua, "lua/util.lua");
//    luaL_dostring(lua, "paths.dofile('util.lua')");


    L = luaL_newstate();
    luaL_openlibs(L);                                    /* Load Lua libraries */
    int status;
    status = luaL_loadfile(L, "lua/init.lua");
    int result = lua_pcall(L, 0, LUA_MULTRET, 0);
    if (result) {
        fprintf(stderr, "Failed to run script: %s\n", lua_tostring(L, -1));
        exit(1);
    }

    lua_getglobal(L, "loadModel");
    lua_pushliteral(L, "model.t7");
    int model_load = lua_pcall(L, 1, 0 ,0);
    if (model_load) {
        fprintf(stderr, "Failed to load model: %s\n", lua_tostring(L, -1));
        exit(1);
    }


    const int height = read_image.rows;
    const int width = read_image.cols;
    const int channs = read_image.channels();
    const int size = channs*height*width;


//    std::cout << "=================" << std::endl;
//    std::cout << (*imgData) << std::endl;
//    cv::Mat dest(height, width, CV_32FC1, imgData);
//    std::cout << dest << std::endl;
//    cv::imwrite("/home/rohan/out.jpg", dest);
//    std::cout << "=================" << std::endl;
//    float *tensorData = (float*)malloc(sizeof(float)*size);



    lua_getglobal(L, "inImage_c1");
    float* imgData = img_bgr[0].ptr<float>();
    THFloatStorage* imgStorage = THFloatStorage_newWithData(imgData, size);
    THFloatTensor* imgTensor = THFloatTensor_newWithStorage2d(imgStorage, 0,
        height, width,   // size 1, stride 1
        width, 1);        // size 2, stride 2
    luaT_pushudata(L, (void*)imgTensor, "torch.FloatTensor");
    lua_setglobal(L, "inImage_c1");

    lua_getglobal(L, "inImage_c2");
    float* imgData1 = img_bgr[1].ptr<float>();
    THFloatStorage* imgStorage1 = THFloatStorage_newWithData(imgData1, size);
    THFloatTensor* imgTensor1 = THFloatTensor_newWithStorage2d(imgStorage1, 0,
        height, width,   // size 1, stride 1
        width, 1);        // size 2, stride 2
    luaT_pushudata(L, (void*)imgTensor1, "torch.FloatTensor");
    lua_setglobal(L, "inImage_c2");

    lua_getglobal(L, "inImage_c3");
    float* imgData2 = img_bgr[2].ptr<float>();
    THFloatStorage* imgStorage2 = THFloatStorage_newWithData(imgData2, size);
    THFloatTensor* imgTensor2 = THFloatTensor_newWithStorage2d(imgStorage2, 0,
        height, width,   // size 1, stride 1
        width, 1);        // size 2, stride 2
    luaT_pushudata(L, (void*)imgTensor2, "torch.FloatTensor");
    lua_setglobal(L, "inImage_c3");



//    THFloatTensor_free(imgTensor);
//    THFloatStorage_free(imgStorage);
//    lua_gc(L, LUA_GCCOLLECT, 0);
    lua_getglobal(L, "loadImage");
    lua_pcall(L,0,0,0);

    int cx = 337;
    int cy = 293;
    float scale = 1.47;

    lua_getglobal(L, "evaluate");
    lua_pushinteger(L,cx);
    lua_pushinteger(L,cy);
    lua_pushnumber(L,scale);
    lua_pcall(L,3,0,1);

    lua_getglobal(L, "keypoint_locs");
    THFloatTensor* keypointTensor = (THFloatTensor*)luaT_toudata(L, -1, "torch.FloatTensor");
    float* kpts = THFloatTensor_data(keypointTensor);

    int num_of_keypoint = THFloatTensor_size(keypointTensor,0);
    for (unsigned int i = 0; i < num_of_keypoint*2; i++) {
        std::cout << *(kpts+i) << std::endl;
    }


    std::cout << "done" << std::endl;


    message_filters::Subscriber<sensor_msgs::Image> img_sub(priv_nh, "input_image", 1);
    message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> box_sub(priv_nh, "input_bbox", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(100), img_sub, box_sub);
    sync.registerCallback(boost::bind(&msgCallback, _1, _2));

    ros::spin();
    
    return 0;
}
