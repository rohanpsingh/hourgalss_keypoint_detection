#include <ros/ros.h>
#include <iostream>
#include <chrono>
#include <boost/thread/thread.hpp>
#include <Eigen/Dense>

#include <darknet_ros_msgs/BoundingBoxes.h>
#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv_apps/Point2DArray.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

extern "C" {
    #include <lua.h>
    #include <lualib.h>
    #include <lauxlib.h>
    #include <luaT.h>
    #include <TH/TH.h>
}


image_transport::Publisher image_keypoints;
ros::Publisher keypoint_pos;
lua_State *L;



void msgCallback(const sensor_msgs::ImageConstPtr& img, const darknet_ros_msgs::BoundingBoxesConstPtr& box){

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    cv::Mat read_image, image;
    try{
        read_image = cv_bridge::toCvShare(img, "bgr8")->image;
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("Could not convert from '%s' to 'bgr8'.", img->encoding.c_str());
    }
    read_image.convertTo(image, CV_32FC3);

    cv::Mat img_bgr[3];
    cv::split(image, img_bgr);


    const int height = read_image.rows;
    const int width = read_image.cols;
    const int channs = read_image.channels();
    const int size = channs*height*width;
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

    lua_getglobal(L, "loadImage");
    lua_pcall(L,0,0,0);



    const int xmin = box->bounding_boxes[0].xmin;
    const int xmax = box->bounding_boxes[0].xmax;
    const int ymin = box->bounding_boxes[0].ymin;
    const int ymax = box->bounding_boxes[0].ymax;
    int cx = abs(xmax+xmin)/2;
    int cy = abs(ymax+ymin)/2;
    float scale = std::max(xmax-xmin, ymax-ymin);
    scale /= 200.0f;

    lua_getglobal(L, "evaluate");
    lua_pushinteger(L,cx);
    lua_pushinteger(L,cy);
    lua_pushnumber(L,scale);
    lua_pcall(L,3,0,1);

    lua_getglobal(L, "keypoint_locs");

    THFloatTensor* keypointTensor = (THFloatTensor*)luaT_toudata(L, -1, "torch.FloatTensor");
    float* kpts = THFloatTensor_data(keypointTensor);
    int num_of_keypoint = THFloatTensor_size(keypointTensor,0);

    opencv_apps::Point2DArray pt_array_msg;
    cv::Mat keypoint_img = read_image.clone();
    for (unsigned int i = 0; i < num_of_keypoint*2; i++) {
        cv::Point pt;
        pt.x = *(kpts+i);
        pt.y = *(kpts+i+1);
        float rd = 5;
        if (pt != cv::Point(-1,-1)) {
            cv::circle(keypoint_img, pt, rd, cv::Scalar(0,255,0), -1);
            cv::putText(keypoint_img, std::to_string(i/2), pt, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 3);
        }

        opencv_apps::Point2D op;
        op.x = pt.x;
        op.y = pt.y;
        pt_array_msg.points.push_back(op);
        i++;
    }

    keypoint_pos.publish(pt_array_msg);

    sensor_msgs::ImagePtr img_pub_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", keypoint_img).toImageMsg();
    image_keypoints.publish(img_pub_msg);

    lua_gc(L, LUA_GCCOLLECT, 0);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "callBack time: " << duration/1000 << " ms" << std::endl;
    return;
}


int main (int argc, char** argv){

    ros::init(argc, argv, "detect");
    ros::NodeHandle priv_nh("~");


    L = luaL_newstate();
    std::cout << "------lua loading libraries----- " << std::endl;
    luaL_openlibs(L);
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

    image_transport::ImageTransport it(priv_nh);
    image_keypoints = it.advertise("keypoints",1);

    keypoint_pos = priv_nh.advertise<opencv_apps::Point2DArray>("keypoint_pos", 1);

    message_filters::Subscriber<sensor_msgs::Image> img_sub(priv_nh, "input_image", 1);
    message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> box_sub(priv_nh, "input_bbox", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(100), img_sub, box_sub);
    sync.registerCallback(boost::bind(&msgCallback, _1, _2));

    ros::spin();
    
    return 0;
}
