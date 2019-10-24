#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#elif USE_SOCKET_PUBLISHER
#include "socket_publisher/publisher.h"
#endif

#include "openvslam/system.h"
#include "openvslam/config.h"

#include <iostream>
#include <chrono>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

#include <openni2/OpenNI.h>

class Xtion {
public:
    Xtion() {
        openni::Status rc;
        rc = openni::OpenNI::initialize();
        if (rc != openni::STATUS_OK) {
            std::cerr << "OpenNI Initial Error: " << openni::OpenNI::getExtendedError() << std::endl;
            openni::OpenNI::shutdown();
            throw std::string("error in constructor");
        }


        const char *deviceURI = openni::ANY_DEVICE;
        rc = device.open(deviceURI);
        if (rc != openni::STATUS_OK) {
            std::cerr << "ERROR: Can't Open Device: " << openni::OpenNI::getExtendedError() << std::endl;
            openni::OpenNI::shutdown();
            throw std::string("error in constructor");
        }


        rc = colorStream.create(device, openni::SENSOR_COLOR);
        if (rc == openni::STATUS_OK) {
            // setting
            openni::VideoMode colorMode;
            colorMode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);
            colorMode.setFps(30);
            colorMode.setResolution(640, 480);
            colorStream.setVideoMode(colorMode);

            height = colorMode.getResolutionY();
            width = colorMode.getResolutionX();
            std::cout << "Color = (" << width << "," << height << ")" << std::endl;
            // Set exposure if needed
            colorStream.getCameraSettings()->setAutoWhiteBalanceEnabled(false);
            int exposure = colorStream.getCameraSettings()->getExposure();
            int delta = 100;
            colorStream.getCameraSettings()->setExposure(exposure + delta);

            rc = colorStream.start();
            if (rc != openni::STATUS_OK) {
                std::cerr << "ERROR: Can't start color stream on device: " << openni::OpenNI::getExtendedError()
                          << std::endl;
                colorStream.destroy();
                throw std::string("error in constructor");
            }
        } else {
            std::cerr << "ERROR: This device does not have color sensor" << std::endl;
            openni::OpenNI::shutdown();
            throw std::string("error in constructor");
        }

        rc = depthStream.create(device, openni::SENSOR_DEPTH);
        if (rc == openni::STATUS_OK) {
            // setting
            openni::VideoMode depthMode;
            depthMode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);
            depthMode.setFps(30);
            depthMode.setResolution(640, 480);
            depthStream.setVideoMode(depthMode);
            height = depthMode.getResolutionY();
            width = depthMode.getResolutionX();
            std::cout << "Depth = (" << width << "," << height << ")" << std::endl;

            rc = depthStream.start();
            if (rc != openni::STATUS_OK) {
                std::cerr << "ERROR: Can't start depth stream on device: " << openni::OpenNI::getExtendedError()
                          << std::endl;
                depthStream.destroy();
                throw std::string("error in constructor");
            }
        } else {
            std::cerr << "ERROR: This device does not have depth sensor" << std::endl;
            openni::OpenNI::shutdown();
            throw std::string("error in constructor");
        }



//    const openni::SensorInfo* sinfo = device.getSensorInfo(openni::SENSOR_COLOR); // select index=4 640x480, 30 fps, 1mm
//    const openni::Array< openni::VideoMode>& modesColor = sinfo->getSupportedVideoModes();
//    for (int i = 0; i<modesColor.getSize(); i++) {
//        printf("%i: %ix%i, %i fps, %i format\n", i, modesColor[i].getResolutionX(), modesColor[i].getResolutionY(),
//               modesColor[i].getFps(), modesColor[i].getPixelFormat()); //PIXEL_FORMAT_DEPTH_1_MM = 100, PIXEL_FORMAT_DEPTH_100_UM
//    }

        rc = device.setDepthColorSyncEnabled(true);
        if (rc != openni::STATUS_OK) {
            std::cerr << "ERROR: Can't enable depth color sync: " << openni::OpenNI::getExtendedError() << std::endl;
            openni::OpenNI::shutdown();
            throw std::string("error in constructor");
        }
        rc = device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
        if (rc != openni::STATUS_OK) {
            std::cerr << "ERROR: Can't register depth to color: " << openni::OpenNI::getExtendedError() << std::endl;
            openni::OpenNI::shutdown();
            throw std::string("error in constructor");
        }

    }

    int getFrame(cv::Mat &img, cv::Mat &depth) {
        bool isFlip = true;
        colorStream.readFrame(&colorFrame);
        if (colorFrame.isValid()) {
            cv::Mat img_(height, width, CV_8UC3, (void *) colorFrame.getData());
            cv::cvtColor(img_, img, cv::COLOR_RGB2BGR);
            if (isFlip == true)
                cv::flip(img, img, 1);
        } else {
            std::cerr << "ERROR: Cannot read color frame from color stream. Quitting..." << std::endl;
            return false;
        }
        depthStream.readFrame(&depthFrame);
        if (depthFrame.isValid())
        {
            cv::Mat depth_(height, width, CV_16UC1, (void*)depthFrame.getData());
            depth_.convertTo(depth, CV_16UC1, depth_scale);
            if (isFlip == true)
                cv::flip(depth, depth, 1);
        }
        else
        {
            std::cerr << "ERROR: Cannot read depth frame from depth stream. Quitting..." << std::endl;
            return false;
        }
        return true;
    }

    void setDepthScale(double scale){
        depth_scale = scale;
    }

    ~Xtion() {
        colorStream.stop();
        colorStream.destroy();
        depthStream.stop();
        depthStream.destroy();
        device.close();
        openni::OpenNI::shutdown();
    }

private:
    openni::Device device;
    openni::VideoStream colorStream;
    openni::VideoFrameRef colorFrame;
    openni::VideoStream depthStream;
    openni::VideoFrameRef depthFrame;
    int height;
    int width;
    double depth_scale = 5.0;
};

void mono_tracking(const std::shared_ptr<openvslam::config>& cfg,
                   const std::string& vocab_file_path, const unsigned int cam_num, const std::string& mask_img_path,
                   const float scale, const std::string& map_db_path) {
    // load the mask image
    const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // startup the SLAM process
    SLAM.startup();

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

    Xtion video;
//    if (!video.isOpened()) {
//        spdlog::critical("cannot open a camera {}", cam_num);
//        SLAM.shutdown();
//        return;
//    }

    cv::Mat frame, depth;
    double timestamp = 0.0;
    std::vector<double> track_times;

    unsigned int num_frame = 0;

    bool is_not_end = true;
    // run the SLAM in another thread
    std::thread thread([&]() {
        while (is_not_end) {
            // check if the termination of SLAM system is requested or not
            if (SLAM.terminate_is_requested()) {
                break;
            }

            is_not_end = video.getFrame(frame, depth);

            if (frame.empty()) {
                continue;
            }
            if (scale != 1.0) {
                cv::resize(frame, frame, cv::Size(), scale, scale, cv::INTER_LINEAR);
            }

            const auto tp_1 = std::chrono::steady_clock::now();

            // input the current frame and estimate the camera pose
            SLAM.feed_monocular_frame(frame, timestamp, mask);

            const auto tp_2 = std::chrono::steady_clock::now();

            const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
            track_times.push_back(track_time);

            timestamp += 1.0 / cfg->camera_->fps_;
            ++num_frame;
        }

        // wait until the loop BA is finished
        while (SLAM.loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
    });

    // run the viewer in the current thread
#ifdef USE_PANGOLIN_VIEWER
    viewer.run();
#elif USE_SOCKET_PUBLISHER
    publisher.run();
#endif

    thread.join();

    // shutdown the SLAM process
    SLAM.shutdown();

    if (!map_db_path.empty()) {
        // output the map database
        SLAM.save_map_database(map_db_path);
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
}

void rgbd_tracking(const std::shared_ptr<openvslam::config>& cfg,
                   const std::string& vocab_file_path, const std::string& mask_img_path,
                    const std::string& map_db_path) {
    // load the mask image
    const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // startup the SLAM process
    SLAM.startup();

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

    Xtion video;
    video.setDepthScale(1.0);
//    if (!video.isOpened()) {
//        spdlog::critical("cannot open a camera {}", cam_num);
//        SLAM.shutdown();
//        return;
//    }

    cv::Mat frame, depth;
    double timestamp = 0.0;
    std::vector<double> track_times;

    unsigned int num_frame = 0;

    bool is_not_end = true;
    // run the SLAM in another thread
    std::thread thread([&]() {
        while (is_not_end) {
            // check if the termination of SLAM system is requested or not
            if (SLAM.terminate_is_requested()) {
                break;
            }

            is_not_end = video.getFrame(frame, depth);
            if (frame.empty()) {
                continue;
            }

            const auto tp_1 = std::chrono::steady_clock::now();

            if (!frame.empty() && !depth.empty()) {
                // input the current frame and estimate the camera pose
                SLAM.feed_RGBD_frame(frame, depth, timestamp);
            }

            const auto tp_2 = std::chrono::steady_clock::now();

            const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
            track_times.push_back(track_time);

            timestamp += 1.0 / cfg->camera_->fps_;
            ++num_frame;
        }

        // wait until the loop BA is finished
        while (SLAM.loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
    });

    // run the viewer in the current thread
#ifdef USE_PANGOLIN_VIEWER
    viewer.run();
#elif USE_SOCKET_PUBLISHER
    publisher.run();
#endif

    thread.join();

    // shutdown the SLAM process
    SLAM.shutdown();

    if (!map_db_path.empty()) {
        // output the map database
        SLAM.save_map_database(map_db_path);
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
}


int main(int argc, char* argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif

    // create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
    auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
    auto mask_img_path = op.add<popl::Value<std::string>>("", "mask", "mask image path", "");
    auto scale = op.add<popl::Value<float>>("s", "scale", "scaling ratio of images", 1.0);
    auto map_db_path = op.add<popl::Value<std::string>>("p", "map-db", "store a map database at this path after SLAM", "");
    auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!vocab_file_path->is_set() || !config_file_path->is_set()) {
        std::cerr << "invalid arguments" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    if (debug_mode->is_set()) {
        spdlog::set_level(spdlog::level::debug);
    }
    else {
        spdlog::set_level(spdlog::level::info);
    }

    // load configuration
    std::shared_ptr<openvslam::config> cfg;
    try {
        cfg = std::make_shared<openvslam::config>(config_file_path->value());
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStart("slam.prof");
#endif

    // run tracking
    if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Monocular) {
        mono_tracking(cfg, vocab_file_path->value(), 0, mask_img_path->value(),
                      scale->value(), map_db_path->value());
    }else if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::RGBD) {
        rgbd_tracking(cfg, vocab_file_path->value(),mask_img_path->value(), map_db_path->value());
    }
    else {
        throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif

    return EXIT_SUCCESS;
}
