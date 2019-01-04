#ifndef ROS_HEADERS_H
#define ROS_HEADERS_H

#include <ros/ros.h>
#include <boost/thread/thread.hpp>

#include <darknet_ros_msgs/BoundingBoxes.h>
#include <object_keypoint_msgs/ObjectKeyPointArray.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <dynamic_reconfigure/server.h>
#include <detection_hg/paramConfig.h>

#include <roseus/StringString.h>

//ros dynamic params
float min_hm_thresh;
float font_scale;
int font_thick;
int circle_rad;
int vis_kp_ind;

bool model_load_success = false;

#endif // ROS_HEADERS_H
