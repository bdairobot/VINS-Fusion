/*******************************************************
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Dai Bo (bdairobot@gmail.com)
 *******************************************************/

#include "ros/ros.h"
#include <ros/console.h>
#include "globalOptTest.h"
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/FluidPressure.h>
#include <std_msgs/Header.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <stdio.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <fstream>
#include <queue>
#include <mutex>
#include "LocalCartesian.hpp"
using namespace std;

GlobalOptimization globalEstimator;
ros::Publisher pub_global_odometry, pub_global_path, pub_car, pub_gps_baro_pos;
nav_msgs::Path *global_path;
nav_msgs::Path gps_path,gps_baro_path;
double last_vio_t = -1;
// gps information: time, x, y, z, xy_var, z_var
queue<pair<double, vector<double>>> tmpGPSQueue;
mutex m_buf, map_buf;
string OUTPUT_PATH;

GeographicLib::LocalCartesian geoConverter;
int gps_init = 0;
ros::Publisher pub_gps_path;
ros::Publisher pub_gps_baro_path;
ros::Publisher pub_gps_info;
static int GPS_MARKER_NUM = 1;
// baro information: time, z, z_var
queue<pair<double, vector<double>>> tmpBaroQueue;
ros::Publisher pub_baro_pose;

// init flag: 1:baro, 3: baro+gps, 7:baro+gps+vio
int init_flag = 0;
int error_flag = 0;

// static const float BETA_TABLE[5] = {0,
// 				    3.84,
// 				    5.99,
// 				    7.81,
//                     9.49
// 				   };

static const float BETA_TABLE[5] = {0,
				    2.71,
				    4.61,
				    6.25,
                    7.78
				   };

// queue pair betweeen GPS and VIO, baro and VIO
// info: t, vio_x, vio_y, vio_var, gps_x, gps_y, gps_var

// aligned data
static map<double, vector<double>> map_GPS; // x,y,z direction
// info: t, vio_z, vio_var, baro_z, baro_var
static map<double, vector<double>> map_Baro; // z direction
static map<double, vector<double>> map_Mag // mag

void publish_car_model(double t, Eigen::Vector3d t_w_car, Eigen::Quaterniond q_w_car)
{
    visualization_msgs::MarkerArray markerArray_msg;
    visualization_msgs::Marker car_mesh;
    car_mesh.header.stamp = ros::Time(t);
    car_mesh.header.frame_id = "world";
    car_mesh.type = visualization_msgs::Marker::MESH_RESOURCE;
    car_mesh.action = visualization_msgs::Marker::ADD;
    car_mesh.id = 0;

    car_mesh.mesh_resource = "package://global_fusion/models/car.dae";

    Eigen::Matrix3d rot;
    rot << 0, 0, -1, 0, -1, 0, -1, 0, 0;
    
    Eigen::Quaterniond Q;
    Q = q_w_car * rot; 
    car_mesh.pose.position.x    = t_w_car.x();
    car_mesh.pose.position.y    = t_w_car.y();
    car_mesh.pose.position.z    = t_w_car.z();
    car_mesh.pose.orientation.w = Q.w();
    car_mesh.pose.orientation.x = Q.x();
    car_mesh.pose.orientation.y = Q.y();
    car_mesh.pose.orientation.z = Q.z();

    car_mesh.color.a = 1.0;
    car_mesh.color.r = 1.0;
    car_mesh.color.g = 0.0;
    car_mesh.color.b = 0.0;

    float major_scale = 0.5;

    car_mesh.scale.x = major_scale;
    car_mesh.scale.y = major_scale;
    car_mesh.scale.z = major_scale;
    markerArray_msg.markers.push_back(car_mesh);
    pub_car.publish(markerArray_msg);
}

void baro_callback(const sensor_msgs::FluidPressureConstPtr &baro_msg)
{
    static double CONSTANTS_ABSOLUTE_NULL_CELSIUS = -273.15;
    static double CONSTANTS_AIR_GAS_CONST = 287.1f;
    static double BARO_MSL = 101.325; /* current pressure at MSL in kPa */
    static double CONSTANTS_ONE_G = 9.80665f;
    static double T1 = 15.0 - CONSTANTS_ABSOLUTE_NULL_CELSIUS;	/* temperature at base height in Kelvin */
    static double a  = -6.5 / 1000.0;	/* temperature gradient in degrees per metre */

    /* measured pressure in kPa */
    const double p = baro_msg->fluid_pressure * 0.001;
    /*
        * Solve:
        *
        *     /               -(aR / g)    \
        *    | (p / BARO_MSL)          . T1 | - T1
        *     \                            /
        * h = -------------------------------  + h1
        *                   a
        */
    double height = (((pow((p / BARO_MSL), (-(a * CONSTANTS_AIR_GAS_CONST) / CONSTANTS_ONE_G))) * T1) - T1) / a;
    static double init_height = 0.0;
    static double last_height = height;
    double filter_factor = 0.5;
    double filter_height = filter_factor*height + (1-filter_factor) * last_height;
    last_height = filter_height;
    m_buf.lock();
    vector<double> baro_info = {filter_height - init_height, 0.3*0.3};
    tmpBaroQueue.push(make_pair(baro_msg->header.stamp.toSec(),baro_info));
    m_buf.unlock();

    // store 3s data
    if (tmpBaroQueue.size() > 150)
        tmpBaroQueue.pop();
    static unsigned int init_size = 100;
    if (!(init_flag & 1<<0)){        
        if(tmpBaroQueue.size() == init_size){
            double sum_x = 0.0;
            double sum_var = 0.0;
            for(int i = 1; !tmpBaroQueue.empty(); i++){
                sum_x += tmpBaroQueue.front().second[0];
                sum_var += tmpBaroQueue.front().second[0] * tmpBaroQueue.front().second[0];
                tmpBaroQueue.pop();
            }
            double mean_x = sum_x / init_size;
            double dev = sqrt(sum_var / init_size - mean_x * mean_x);
            
            init_flag |= 1<<0;
            init_height = mean_x;
            ROS_INFO("baro init: init height : %f, deviation reference: %f", mean_x,dev);
        }
    }
    if (init_flag & 1<<0){
        geometry_msgs::PoseStamped baro_height;
        baro_height.header = baro_msg->header;
        baro_height.pose.position.x = height - init_height;
        baro_height.pose.position.y = filter_height - init_height;
        pub_baro_pose.publish(baro_height);
    }
}

void GPS_callback(const sensor_msgs::NavSatFixConstPtr &GPS_msg)
{
 
    double gps_t = GPS_msg->header.stamp.toSec();
    if (GPS_msg->status.status==-1){
        error_flag |= 1<<0;
        return;
    }
    static int count = 0;
    if (!(init_flag & 1<<1)){
        // position accuracy less than 2 meters or after 2 seconds
        if (GPS_msg->position_covariance[0] < 4.0 || count == 10){
            geoConverter.Reset(GPS_msg->latitude, GPS_msg->longitude, GPS_msg->altitude);
            init_flag |= 1<<1;
            ROS_INFO("geo init: lat %12.8f,lon %12.8f, alt %12.8f, cov: %8.4f", GPS_msg->latitude,GPS_msg->longitude, GPS_msg->altitude, GPS_msg->position_covariance[0]);
        }
        count++;
    }
    if (init_flag & 1<<1) {
        
        geometry_msgs::PoseStamped pose_gps;
        pose_gps.header.stamp = ros::Time(gps_t);
        pose_gps.header.frame_id = "world";
        double xyz[3];
        geoConverter.Forward(GPS_msg->latitude, GPS_msg->longitude,GPS_msg->altitude, xyz[0],xyz[1],xyz[2]);
           //printf("gps_callback! ");
        vector<double> gps_info = {xyz[0],xyz[1],xyz[2], GPS_msg->position_covariance[0], GPS_msg->position_covariance[8]};
        m_buf.lock();
        tmpGPSQueue.push(make_pair(gps_t, gps_info));
        // store 3s data
        m_buf.unlock();
        if (tmpGPSQueue.size() > 15)
            tmpGPSQueue.pop();
        pose_gps.pose.position.x = xyz[0];
        pose_gps.pose.position.y = xyz[1];
        // if (init_flag & 1<<0)
            // pose_gps.pose.position.z = tmpBaroQueue.back().second[0];
        pose_gps.pose.orientation.w = 1.0;
        pose_gps.pose.orientation.x = 0.0;
        pose_gps.pose.orientation.y = 0.0;
        pose_gps.pose.orientation.z = 0.0;
        gps_path.header = pose_gps.header;
        gps_path.poses.push_back(pose_gps);
        pub_gps_path.publish(gps_path);

        if (init_flag & 1<<0){
            pose_gps.pose.position.z = sqrt(GPS_msg->position_covariance[0]);
            gps_baro_path.poses.push_back(pose_gps);
            pub_gps_baro_path.publish(gps_baro_path);
            pub_gps_baro_pos.publish(pose_gps);

            visualization_msgs::Marker gps_marker;
            gps_marker.header = GPS_msg->header;
            gps_marker.header.frame_id = "world";
            gps_marker.ns = "gps_info";
            gps_marker.type = visualization_msgs::Marker::CYLINDER;
            gps_marker.action = visualization_msgs::Marker::ADD;
            gps_marker.pose.position = pose_gps.pose.position;
            gps_marker.pose.orientation.w = 1.0;
            gps_marker.lifetime = ros::Duration();
            gps_marker.id = GPS_msg->header.seq;

            static queue<int> gps_id;
            gps_id.push(gps_marker.id);

            gps_marker.scale.x = sqrt(GPS_msg->position_covariance[0]);
            gps_marker.scale.y = sqrt(GPS_msg->position_covariance[4]);
            gps_marker.scale.z = 0.14;
            gps_marker.color.r = 1.0;
            gps_marker.color.g = 0.2;
            gps_marker.color.b = 0.2;
            gps_marker.color.a = 0.5;
            pub_gps_info.publish(gps_marker);
            if (gps_id.size() > uint(GPS_MARKER_NUM)){
                visualization_msgs::Marker delete_marker;
                delete_marker.ns = "gps_info";
                delete_marker.action = visualization_msgs::Marker::DELETE;
                delete_marker.id = gps_id.front();
                gps_id.pop();
                pub_gps_info.publish(delete_marker);
            }
        }
    }
}

void keyframe_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    static uint sub_keyframe_number = 0;
    if (!(error_flag & 1<<2)){
        double kf_t = pose_msg->header.stamp.toSec();
       Eigen::Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
        Eigen::Quaterniond vio_q;
        vio_q.w() = pose_msg->pose.pose.orientation.w;
        vio_q.x() = pose_msg->pose.pose.orientation.x;
        vio_q.y() = pose_msg->pose.pose.orientation.y;
        vio_q.z() = pose_msg->pose.pose.orientation.z;
    
        globalEstimator.inputKeyframe(t, vio_t, vio_q, sqrt(pose_msg->pose.covariance[0]), sqrt(pose_msg->pose.covariance[28]));

        if(map_GPS.find(kf_t) != map_GPS.end())
            globalEstimator.inputGPS(kf_t, map_GPS[kf_t]);
        if(map_Baro.find(kf_t) != map_Baro.end())
            globalEstimator.inputBaro(kf_t, map_Baro[kf_t]);
        if(map_Mag.find(kf_t) != map_Mag.end())
            globalEstimator.inputMag(kf_t, map_Mag[kf_t]);
    }
}

void vio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    //printf("vio_callback! ");
    if(pose_msg->pose.covariance[0] < 0){
        error_flag |= 1<<2;
        init_flag &= 0<<2;
        pub_vins_restart.publish(true);
        return;
    }
    error_flag &= 0<<2;
    double t = pose_msg->header.stamp.toSec();
    last_vio_t = t;
    Eigen::Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    Eigen::Quaterniond vio_q;
    vio_q.w() = pose_msg->pose.pose.orientation.w;
    vio_q.x() = pose_msg->pose.pose.orientation.x;
    vio_q.y() = pose_msg->pose.pose.orientation.y;
    vio_q.z() = pose_msg->pose.pose.orientation.z;
    
    globalEstimator.inputOdom(t, vio_t, vio_q);

    Eigen::Vector3d global_t;
    Eigen:: Quaterniond global_q;
    globalEstimator.getGlobalOdom(global_t, global_q);
    { // check with baro
        static uint duration = 4, step = 10;
        if (init_flag & 1<<0){
            while(!tmpBaroQueue.empty()){
                bool oldest_baro = (tmpBaroQueue.size() == 150);
                double baro_t = tmpBaroQueue.front().first;
                if (t < baro_t - 0.02 && oldest_baro){
                    ROS_ERROR("VIO is away behind baro information! ");
                    assert(0);
                } else if (t <= baro_t + 0.02 && t >= baro_t - 0.02){
                    m_buf.lock();
                    if (map_Baro.size() == duration*step){
                        for (uint i = 0; i<duration*step-1; i++)
                            map_Baro[i] = map_Baro[i+1];
                    } else ROS_INFO("map_Baro.size(): %lu ", map_Baro.size());
                    int index = (map_Baro.size() == duration*step) ? map_Baro.size() - 1 : map_Baro.size();
                    map_Baro[index] = make_pair(vector<double> {global_t(2),pose_msg->pose.covariance[0]}, tmpBaroQueue.front().second);
                    tmpBaroQueue.pop();
                    m_buf.unlock();
                    break;
                } else
                    tmpBaroQueue.pop();
            }
        }

        if(map_Baro.size() == duration*step){   
            int error_count = 0;
            for (int i =  0; i < 10; i++){
                double vio_move = map_Baro[duration*step-10+i].first[0] - map_Baro[0].first[0];
                double vio_var = map_Baro[duration*step-10+i].first[1] + map_Baro[0].first[1];
                double baro_move = map_Baro[duration*step-10+i].second[0] - map_Baro[0].second[0];
                double baro_var = map_Baro[duration*step-10+i].second[1] + map_Baro[0].second[1];
                // chi distribution check
                // ROS_INFO("VIO Baro BETA info: %8.4f, vio_dev: %8.4f, baro_dev:%8.4f ", (vio_move - baro_move)*(vio_move - baro_move)/(vio_var + baro_var), sqrt(vio_var/2), sqrt(baro_var/2) );
                if ((vio_move - baro_move)*(vio_move - baro_move)/(vio_var + baro_var) > BETA_TABLE[1])
                    error_count++;
            }
            static int highly_error = 0;
            if (error_count >= 6){
                if (++highly_error > 3){
                    ROS_ERROR("********** VIO Error, VIO-Baro Drifted. Please Restart VIO! ");
                    // assert(0);
                }
                ROS_WARN("Warning, VIO-Baro May Drifted! ");
                // assert(0);
            } else highly_error = 0;
        }
    }

    { // check with gps
        static uint duration = 4, step = 5;
        if (init_flag & 1<<1){
            while(!tmpGPSQueue.empty()){
                // two cases: first, optimization initialized, and not
                pair<double, vector<double>> GPS_info = tmpGPSQueue.front();
                double gps_t = GPS_info.first;
                Eigen::Vector3d gps_pose(GPS_info.second[0],GPS_info.second[1],GPS_info.second[2]);
                pair<double, vector<double>> last_pop;
                static bool pop_flag = false;
                m_buf.lock();
                if (t < gps_t - 0.02 && (tmpGPSQueue.size()==15)){
                    ROS_ERROR("VIO is away behind GPS information! ");
                    assert(0);
                } else if (t <= gps_t + 0.02 && t >= gps_t - 0.02){
                    if (map_GPS.size() == duration*step){
                        for (uint i = 0; i<duration*step-1; i++)
                            map_GPS[i] = map_GPS[i+1];
                    }else ROS_INFO("map_GPS.size(): %lu ", map_GPS.size());
                    int index = (map_GPS.size() == duration*step) ? map_GPS.size() - 1 : map_GPS.size();
                    map_GPS[index] = make_pair(vector<double> {global_t(0), global_t(1),pose_msg->pose.covariance[0]} , vector<double>{gps_pose(0), gps_pose(1), GPS_info.second[3]});
                    last_pop = GPS_info;
                    tmpGPSQueue.pop();
                    pop_flag = true;
                    break;
                } else { // insert
                    if(!pop_flag || t > gps_t + 0.02){
                        last_pop = GPS_info;
                        tmpGPSQueue.pop();
                        pop_flag = true;
                    } else {
                        Eigen::Vector3d insert_gps_pose = gps_pose - (gps_t - t)/(gps_t - last_pop.first)*(gps_pose - Eigen::Vector3d{last_pop.second[1],last_pop.second[1],last_pop.second[2]});

                        if (map_GPS.size() == duration*step){
                        for (uint i = 0; i<duration*step-1; i++)
                            map_GPS[i] = map_GPS[i+1];
                        }else ROS_INFO("map_GPS.size(): %lu ", map_GPS.size());
                        int index = (map_GPS.size() == duration*step) ? map_GPS.size() - 1 : map_GPS.size();
                        map_GPS[index] = make_pair(vector<double> {global_t(0), global_t(1),pose_msg->pose.covariance[0]} , vector<double> {insert_gps_pose(0), insert_gps_pose(1), GPS_info.second[3]});
                        last_pop = GPS_info;
                        tmpGPSQueue.pop();
                        pop_flag = true;
                        break;
                    }
                }
                m_buf.unlock();
            }
        }
        // if(map_GPS.size() == duration*step){
        //     int error_count = 0;
        //     for (int i =  0; i < 10; i++){
        //         if (/* globalEstimator.initialized */ false){

        //         }else{
        //             Eigen::Vector2d vio_move(map_GPS[duration*step-10+i].first[0] - map_GPS[0].first[0], map_GPS[duration*step-10+i].first[1] - map_GPS[0].first[1]);
        //             double vio_var = map_GPS[duration*step-10+i].first[2] + map_GPS[0].first[2];
        //             Eigen::Vector2d gps_move(map_GPS[duration*step-10+i].second[0] - map_GPS[0].second[0],
        //             map_GPS[duration*step-10+i].second[1] - map_GPS[0].second[1]);
        //             double gps_var = map_GPS[duration*step-10+i].second[2] + map_GPS[i].second[2]; 
        //             Eigen::Vector2d error_move = vio_move - gps_move.norm()*vio_move.normalized();
        //             ROS_INFO("VIO GPS BETA info: %8.4f, vio_dev: %8.4f, baro_dev:%8.4f ",(double)(error_move.transpose()/(vio_var + gps_var)*error_move), sqrt(vio_var/2), sqrt(gps_var/2) );

        //             // chi distribution x^2 + y^2
        //             if ((error_move.transpose()/(vio_var + gps_var)*error_move) > BETA_TABLE[2]){
        //                 error_count++;
        //             }
        //         }
        //     }
        //     static int highly_error = 0;
        //     if (error_count >= 6){
        //         if (++highly_error > 3){
        //             ROS_ERROR("VIO Error, VIO-GPS Drifted. Please Restart VIO! ");
        //             assert(0);
        //         }
        //         ROS_WARN("Warning, VIO-GPS May Drifted! ");
        //         // assert(0);
        //     } else highly_error = 0;
        // }
    }



    nav_msgs::Odometry odometry;
    odometry.header = pose_msg->header;
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "world";
    odometry.pose.pose.position.x = global_t.x();
    odometry.pose.pose.position.y = global_t.y();
    odometry.pose.pose.position.z = global_t.z();
    odometry.pose.pose.orientation.x = global_q.x();
    odometry.pose.pose.orientation.y = global_q.y();
    odometry.pose.pose.orientation.z = global_q.z();
    odometry.pose.pose.orientation.w = global_q.w();
    pub_global_odometry.publish(odometry);
    pub_global_path.publish(*global_path);
    publish_car_model(t, global_t, global_q);


    // write result to file
    ofstream foutC(OUTPUT_PATH+"/vio_global.csv", ios::app);
    foutC.setf(ios::fixed, ios::floatfield);
    foutC.precision(0);
    foutC << pose_msg->header.stamp.toSec() * 1e9 << ",";
    foutC.precision(5);
    foutC << global_t.x() << ","
            << global_t.y() << ","
            << global_t.z() << ","
            << global_q.w() << ","
            << global_q.x() << ","
            << global_q.y() << ","
            << global_q.z() << endl;
    foutC.close();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "globalEstimator");
    ros::NodeHandle n("~");

    global_path = &globalEstimator.global_path;

    string GPS_TOPIC;
    n.param<string>("output_path", OUTPUT_PATH, "~/output/");
    n.param<string>("gps_topic", GPS_TOPIC, "/gps");
    n.param<int>("gps_marker_num", GPS_MARKER_NUM, 1);
    ros::Subscriber sub_GPS = n.subscribe(GPS_TOPIC, 100, GPS_callback);
    ros::Subscriber sub_vio = n.subscribe("/vins_estimator/odometry", 100, vio_callback);
    ros::Subscriber sub_baro = n.subscribe("/mavros/imu/static_pressure",100, baro_callback);
    ros::Subscriber sub_keyframe = n.subscribe("/vins_estimator/keyframe_pose", 100, keyframe_callback);

    pub_global_path = n.advertise<nav_msgs::Path>("global_path", 100);
    pub_global_odometry = n.advertise<nav_msgs::Odometry>("global_odometry", 100);
    pub_car = n.advertise<visualization_msgs::MarkerArray>("car_model", 1000);
    pub_gps_path = n.advertise<nav_msgs::Path>("gps_path", 100);
    pub_gps_baro_path = n.advertise<nav_msgs::Path>("gps_baro_path",100);
    pub_gps_info = n.advertise<visualization_msgs::Marker>("gps_info_marker", 10);
    pub_gps_baro_pos = n.advertise<geometry_msgs::PoseStamped>("gps_baro_pos", 5);
    pub_baro_pose = n.advertise<geometry_msgs::PoseStamped>("baro_pose", 100);
    pub_vins_restart = n.advertise<std_msgs::Bool>("/vins_restart", 5);
    
    // clear file
    ofstream fout(OUTPUT_PATH+"/vio_global.csv", ios::out);
    fout.close();
    
    ros::spin();
    return 0;
}
