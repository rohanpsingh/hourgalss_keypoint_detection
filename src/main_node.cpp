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
#include <opencv_apps/Point2DArrayStamped.h>
#include <std_msgs/Float32MultiArray.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <dynamic_reconfigure/server.h>
#include <detection_hg/paramConfig.h>

extern "C" {
    #include <lua.h>
    #include <lualib.h>
    #include <lauxlib.h>
    #include <luaT.h>
    #include <TH/TH.h>
}

//ros pubs
image_transport::Publisher image_keypoints;
ros::Publisher keypoint_hms;
ros::Publisher keypoint_pos;

//lua state
lua_State *L;

//ros dynamic params
float min_hm_thresh;
float font_scale;
int font_thick;
int circle_rad;
int vis_kp_ind;

//rosparams
bool vis_out;
bool pub_hms;
int max_kps;

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
    int num_of_keypoint_vis = THFloatTensor_size(keypointTensor,0);

    lua_getglobal(L, "heatmap_peaks");
    THFloatTensor* hmpeaksTensor = (THFloatTensor*)luaT_toudata(L, -1, "torch.FloatTensor");
    float* hmps = THFloatTensor_data(hmpeaksTensor);
    if (THFloatTensor_size(hmpeaksTensor,0) != max_kps) {
        ROS_ERROR("something terrible has happened!!");
        return;
    }
    std::vector<float> hm_peaks_vec;
    for (unsigned int i = 0; i < max_kps; i++)
      hm_peaks_vec.push_back(*(hmps+i));
    
    opencv_apps::Point2DArrayStamped pt_array_msg;
    cv::Mat keypoint_img = read_image.clone();
    for (unsigned int i = 0; i < num_of_keypoint_vis*2; i++) {
        if (vis_kp_ind != -1)
            if ((i/2) != vis_kp_ind)
                continue;
        cv::Point pt;
        pt.x = *(kpts+i);
        pt.y = *(kpts+i+1);
        float rd = circle_rad;
        if (pt != cv::Point(-1,-1)) {
            cv::circle(keypoint_img, pt, rd, cv::Scalar(0,255,0), -1);
            cv::putText(keypoint_img, std::to_string(i/2), pt, cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255,0,0), font_thick);
        }

        opencv_apps::Point2D op;
        op.x = pt.x;
        op.y = pt.y;
        pt_array_msg.points.push_back(op);
        i++;
    }

    pt_array_msg.header.stamp = img->header.stamp;
    keypoint_pos.publish(pt_array_msg);

    if (vis_out) {
        sensor_msgs::ImagePtr img_pub_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", keypoint_img).toImageMsg();
        image_keypoints.publish(img_pub_msg);
    }
    if (pub_hms) {
        std_msgs::Float32MultiArray hm_msg;
        hm_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
      	hm_msg.layout.dim[0].label = "kp_peaks";
        hm_msg.layout.dim[0].size = max_kps;
        hm_msg.layout.dim[0].stride = 1;
        hm_msg.layout.data_offset = 0;
        hm_msg.data = hm_peaks_vec;
        keypoint_hms.publish(hm_msg);
    }

    lua_gc(L, LUA_GCCOLLECT, 0);
    lua_settop(L, 0);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "callBack time: " << duration/1000 << " ms" << std::endl;
    return;
}

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

int main (int argc, char** argv){

    ros::init(argc, argv, "detect");
    ros::NodeHandle priv_nh("~");

    std::string nn_model_weights;
    std::string pkg_dir;

    priv_nh.param("max_keypoints", max_kps, int(20));
    priv_nh.param("nn_model_weights", nn_model_weights, std::string("model.t7"));
    priv_nh.param("pkg_dir", pkg_dir, std::string("~/detection_hg"));    
    priv_nh.param("vis_out", vis_out, bool(false));
    priv_nh.param("pub_hms", pub_hms, bool(false));

    L = luaL_newstate();
    std::cout << "------lua loading libraries----- " << std::endl;
    luaL_openlibs(L);
    int status;
    status = luaL_loadfile(L, std::string(pkg_dir + "lua/init.lua").c_str());
    int result = lua_pcall(L, 0, LUA_MULTRET, 0);
    if (result) {
        fprintf(stderr, "Failed to run script: %s\n", lua_tostring(L, -1));
        exit(1);
    }
    
    lua_getglobal(L, "loadModel");
    lua_pushstring(L, (pkg_dir + nn_model_weights).c_str());
    int model_load = lua_pcall(L, 1, 0 ,0);
    if (model_load) {
        fprintf(stderr, "Failed to load model: %s\n", lua_tostring(L, -1));
        exit(1);
    }
    else
        std::cout << "------model load success----- " << std::endl;

    lua_pushinteger(L, max_kps);
    lua_setglobal(L, "num_keypoints");



    image_keypoints = image_transport::ImageTransport(priv_nh).advertise("keypoints",1);
    keypoint_pos = priv_nh.advertise<opencv_apps::Point2DArrayStamped>("keypoint_pos", 1);
    keypoint_hms = priv_nh.advertise<std_msgs::Float32MultiArray>("keypoint_hms", 1);

    message_filters::Subscriber<sensor_msgs::Image> img_sub(priv_nh, "input_image", 1);
    message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> box_sub(priv_nh, "input_bbox", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(100), img_sub, box_sub);
    sync.registerCallback(boost::bind(&msgCallback, _1, _2));

    dynamic_reconfigure::Server<detection_hg::paramConfig> server;
    dynamic_reconfigure::Server<detection_hg::paramConfig>::CallbackType f;
    f = boost::bind(&setDynParams, _1, _2);
    server.setCallback(f);
    
    ros::spin();
    
    return 0;
}
