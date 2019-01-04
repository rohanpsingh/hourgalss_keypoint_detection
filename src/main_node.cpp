#include "common_headers.h"
#include "ros_headers.h"
#include "lua_utils.h"
#include "ros_utils.h"
#include "get_peaks.h"

//ros pubs
image_transport::Publisher image_keypoints;
ros::Publisher keypoint_pub;

//rosparams
int max_kps;
std::string nn_model_weights;
std::string pkg_dir;


void msgCallback(const sensor_msgs::ImageConstPtr& img, const darknet_ros_msgs::BoundingBoxesConstPtr& box){

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    if (!model_load_success){
      std::cout << "model not loaded successfully" << std::endl;
      return;
    }

    cv::Mat read_image;
    try{
        read_image = cv_bridge::toCvShare(img, "bgr8")->image;
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("Could not convert from '%s' to 'bgr8'.", img->encoding.c_str());
    }
    loadImage(read_image);

    const int xmin = box->bounding_boxes[0].xmin;
    const int xmax = box->bounding_boxes[0].xmax;
    const int ymin = box->bounding_boxes[0].ymin;
    const int ymax = box->bounding_boxes[0].ymax;
    loadBBox(xmin, xmax, ymin, ymax);


    const char* locations = "keypoint_locs";
    const char* peakvals = "heatmap_peaks";
    const char* heatmaps = "heatmaps";
    THFloatTensor *keypointTensor, *hmpeaksTensor, *hmapsTensor;
    float *kpts, *hmps, *hmaps;

    getUdata(locations, &keypointTensor, &kpts);
    int num_of_keypoint_vis = THFloatTensor_size(keypointTensor,0);
    
    getUdata(peakvals, &hmpeaksTensor, &hmps);
    if (THFloatTensor_size(hmpeaksTensor,0) != max_kps){
        ROS_ERROR("something terrible has happened!!");
        return;
    }

    getUdata(heatmaps, &hmapsTensor, &hmaps);
    std::vector<float> peaks = getPeakCoordinates(hmapsTensor, hmaps);

    
    object_keypoint_msgs::ObjectKeyPoints obj_kps;
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

        object_keypoint_msgs::KeyPoint kp;
        kp.position.x = pt.x;
	kp.position.y = pt.y;
	kp.confidence = (*(hmps+i/2));
	obj_kps.keypoint.push_back(kp);
        i++;
    }

    object_keypoint_msgs::ObjectKeyPointArray pt_array_msg;
    pt_array_msg.header = img->header;
    pt_array_msg.object.push_back(obj_kps);
    keypoint_pub.publish(pt_array_msg);

    sensor_msgs::ImagePtr img_pub_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", keypoint_img).toImageMsg();
    image_keypoints.publish(img_pub_msg);

    lua_gc(L, LUA_GCCOLLECT, 0);
    lua_settop(L, 0);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "callBack time: " << duration/1000 << " ms" << std::endl;
    return;
}

int main (int argc, char** argv){

    ros::init(argc, argv, "detect");
    ros::NodeHandle priv_nh("~");

    priv_nh.param("max_keypoints", max_kps, int(20));
    priv_nh.param("nn_model_weights", nn_model_weights, std::string("model.t7"));
    priv_nh.param("pkg_dir", pkg_dir, std::string("~/detection_hg"));    

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
    
    lua_pushinteger(L, max_kps);
    lua_setglobal(L, "num_keypoints");


    ros::ServiceServer service = priv_nh.advertiseService("load_trained_model", serviceCallback);

    image_keypoints = image_transport::ImageTransport(priv_nh).advertise("keypoints_img",1);
    keypoint_pub = priv_nh.advertise<object_keypoint_msgs::ObjectKeyPointArray>("keypoints_dat", 1);

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
