// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <mujoco/mujoco.h>
#include "glfw_adapter.h"
#include "simulate.h"
#include "array_safety.h"

#include <lcm/lcm-cpp.hpp>
#include <glog/logging.h>

#include "joint_state_lcm.hpp"
#include "imu_state_lcm.hpp"
#include "joint_control_lcm.hpp"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

extern "C" {
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}

std::shared_ptr<lcm::LCM> joint_state_lcm_ptr_;
std::shared_ptr<lcm::LCM> joint_control_lcm_ptr_;
std::shared_ptr<lcm::LCM> imu_state_lcm_ptr_;

std::shared_ptr<joint_state_lcm> joint_state_ptr_;
std::shared_ptr<imu_state_lcm> imu_state_ptr_;
std::shared_ptr<joint_control_lcm> joint_control_ptr_;

Eigen::Quaterniond imu_quat_;
Eigen::Vector3d imu_pos_;
Eigen::Vector3d imu_acc_;
Eigen::Vector3d imu_gyro_;

Eigen::Vector3d kp_{300, 180, 70};
Eigen::Vector3d kd_{15, 8, 3};

Eigen::Vector3d lf_pos_;
Eigen::Vector3d rf_pos_;
Eigen::Vector3d lb_pos_;
Eigen::Vector3d rb_pos_;
Eigen::Vector3d lf_vel_;
Eigen::Vector3d rf_vel_;
Eigen::Vector3d lb_vel_;
Eigen::Vector3d rb_vel_;
Eigen::Vector3d lf_pos_des_(0.0, 0.67, -1.3);
Eigen::Vector3d rf_pos_des_(-0.0, 0.67, -1.3);
Eigen::Vector3d lb_pos_des_(0.0, 0.67, -1.3);
Eigen::Vector3d rb_pos_des_(-0.0, 0.67, -1.3);
Eigen::Vector3d lf_tau_;
Eigen::Vector3d rf_tau_;
Eigen::Vector3d lb_tau_;
Eigen::Vector3d rb_tau_;

uint64_t delay_cnt = 0;


//void pub_joint_state();
//void pub_imu_state();
//void update_state();
//void joint_control_pd();

namespace {
    namespace mj = ::mujoco;
    namespace mju = ::mujoco::sample_util;

// constants
    const double syncMisalign = 0.1;        // maximum mis-alignment before re-sync (simulation seconds)
    const double simRefreshFraction = 0.7;  // fraction of refresh available for simulation
    const int kErrorLength = 1024;          // load error string length

// model and data
    mjModel *m = nullptr;
    mjData *d = nullptr;

// control noise variables
    mjtNum *ctrlnoise = nullptr;

    using Seconds = std::chrono::duration<double>;


//---------------------------------------- plugin handling -----------------------------------------

// return the path to the directory containing the current executable
// used to determine the location of auto-loaded plugin libraries
    std::string getExecutableDir() {
#if defined(_WIN32) || defined(__CYGWIN__)
        constexpr char kPathSep = '\\';
        std::string realpath = [&]() -> std::string {
          std::unique_ptr<char[]> realpath(nullptr);
          DWORD buf_size = 128;
          bool success = false;
          while (!success) {
            realpath.reset(new(std::nothrow) char[buf_size]);
            if (!realpath) {
              std::cerr << "cannot allocate memory to store executable path\n";
              return "";
            }

            DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
            if (written < buf_size) {
              success = true;
            } else if (written == buf_size) {
              // realpath is too small, grow and retry
              buf_size *=2;
            } else {
              std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
              return "";
            }
          }
          return realpath.get();
        }();
#else
        constexpr char kPathSep = '/';
#if defined(__APPLE__)
        std::unique_ptr<char[]> buf(nullptr);
        {
          std::uint32_t buf_size = 0;
          _NSGetExecutablePath(nullptr, &buf_size);
          buf.reset(new char[buf_size]);
          if (!buf) {
            std::cerr << "cannot allocate memory to store executable path\n";
            return "";
          }
          if (_NSGetExecutablePath(buf.get(), &buf_size)) {
            std::cerr << "unexpected error from _NSGetExecutablePath\n";
          }
        }
        const char* path = buf.get();
#else
        const char *path = "/proc/self/exe";
#endif
        std::string realpath = [&]() -> std::string {
            std::unique_ptr<char[]> realpath(nullptr);
            std::uint32_t buf_size = 128;
            bool success = false;
            while (!success) {
                realpath.reset(new(std::nothrow) char[buf_size]);
                if (!realpath) {
                    std::cerr << "cannot allocate memory to store executable path\n";
                    return "";
                }

                std::size_t written = readlink(path, realpath.get(), buf_size);
                if (written < buf_size) {
                    realpath.get()[written] = '\0';
                    success = true;
                } else if (written == -1) {
                    if (errno == EINVAL) {
                        // path is already not a symlink, just use it
                        return path;
                    }

                    std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
                    return "";
                } else {
                    // realpath is too small, grow and retry
                    buf_size *= 2;
                }
            }
            return realpath.get();
        }();
#endif

        if (realpath.empty()) {
            return "";
        }

        for (std::size_t i = realpath.size() - 1; i > 0; --i) {
            if (realpath.c_str()[i] == kPathSep) {
                return realpath.substr(0, i);
            }
        }

        // don't scan through the entire file system's root
        return "";
    }


// scan for libraries in the plugin directory to load additional plugins
    void scanPluginLibraries() {
        // check and print plugins that are linked directly into the executable
        int nplugin = mjp_pluginCount();
        if (nplugin) {
            std::printf("Built-in plugins:\n");
            for (int i = 0; i < nplugin; ++i) {
                std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
            }
        }

        // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
        const std::string sep = "\\";
#else
        const std::string sep = "/";
#endif


        // try to open the ${EXECDIR}/plugin directory
        // ${EXECDIR} is the directory containing the simulate binary itself
        const std::string executable_dir = getExecutableDir();
        if (executable_dir.empty()) {
            return;
        }

        const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
        mj_loadAllPluginLibraries(
                plugin_dir.c_str(), +[](const char *filename, int first, int count) {
                    std::printf("Plugins registered by library '%s':\n", filename);
                    for (int i = first; i < first + count; ++i) {
                        std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
                    }
                });
    }


//------------------------------------------- simulation -------------------------------------------


    mjModel *LoadModel(const char *file, mj::Simulate &sim) {
        // this copy is needed so that the mju::strlen call below compiles
        char filename[mj::Simulate::kMaxFilenameLength];
        mju::strcpy_arr(filename, file);

        // make sure filename is not empty
        if (!filename[0]) {
            return nullptr;
        }

        // load and compile
        char loadError[kErrorLength] = "";
        mjModel *mnew = 0;
        if (mju::strlen_arr(filename) > 4 &&
            !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                          mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4)) {
            mnew = mj_loadModel(filename, nullptr);
            if (!mnew) {
                mju::strcpy_arr(loadError, "could not load binary model");
            }
        } else {
            mnew = mj_loadXML(filename, nullptr, loadError, mj::Simulate::kMaxFilenameLength);
            // remove trailing newline character from loadError
            if (loadError[0]) {
                int error_length = mju::strlen_arr(loadError);
                if (loadError[error_length - 1] == '\n') {
                    loadError[error_length - 1] = '\0';
                }
            }
        }

        mju::strcpy_arr(sim.load_error, loadError);

        if (!mnew) {
            std::printf("%s\n", loadError);
            return nullptr;
        }

        // compiler warning: print and pause
        if (loadError[0]) {
            // mj_forward() below will print the warning message
            std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
            sim.run = 0;
        }

        return mnew;
    }

// simulate in background thread (while rendering in main thread)
    void PhysicsLoop(mj::Simulate &sim) {
        // cpu-sim syncronization point
        std::chrono::time_point<mj::Simulate::Clock> syncCPU;
        mjtNum syncSim = 0;

        // run until asked to exit
        while (!sim.exitrequest.load()) {
            if (sim.droploadrequest.load()) {
                mjModel *mnew = LoadModel(sim.dropfilename, sim);
                sim.droploadrequest.store(false);

                mjData *dnew = nullptr;
                if (mnew) dnew = mj_makeData(mnew);
                if (dnew) {
                    sim.Load(mnew, dnew, sim.dropfilename);

                    mj_deleteData(d);
                    mj_deleteModel(m);

                    m = mnew;
                    d = dnew;
                    mj_forward(m, d);

                    // allocate ctrlnoise
                    free(ctrlnoise);
                    ctrlnoise = (mjtNum *) malloc(sizeof(mjtNum) * m->nu);
                    mju_zero(ctrlnoise, m->nu);
                }
            }

            // 此处应该是请求导入一个新的XML文件
            if (sim.uiloadrequest.load()) {
                sim.uiloadrequest.fetch_sub(1);
                mjModel *mnew = LoadModel(sim.filename, sim);
                mjData *dnew = nullptr;
                if (mnew) dnew = mj_makeData(mnew);
                if (dnew) {
                    sim.Load(mnew, dnew, sim.filename);

                    mj_deleteData(d);
                    mj_deleteModel(m);

                    m = mnew;
                    d = dnew;
                    mj_forward(m, d);

                    // allocate ctrlnoise
                    free(ctrlnoise);
                    ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
                    mju_zero(ctrlnoise, m->nu);
                }
            }

            // sleep for 1 ms or yield, to let main thread run
            //  yield results in busy wait - which has better timing but kills battery life
            if (sim.run && sim.busywait) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

//    std::cout << "physics loop is running" << std::endl;

            {
                // lock the sim mutex
                const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

                // run only if model is present
                if (m) {
                    // running
                    if (sim.run) {
                        // record cpu time at start of iteration
                        const auto startCPU = mj::Simulate::Clock::now();

                        // elapsed CPU and simulation time since last sync
                        const auto elapsedCPU = startCPU - syncCPU;
                        double elapsedSim = d->time - syncSim;

                        // inject noise
                        if (sim.ctrl_noise_std) {
                            // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
                            mjtNum rate = mju_exp(-m->opt.timestep / mju_max(sim.ctrl_noise_rate, mjMINVAL));
                            mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

                            for (int i = 0; i < m->nu; i++) {
                                // update noise
                                ctrlnoise[i] = rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

                                // apply noise
                                d->ctrl[i] = ctrlnoise[i];
                            }
                        }

                        // requested slow-down factor
                        double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

                        // misalignment condition: distance from target sim time is bigger than syncmisalign
                        bool misaligned =
                                mju_abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

                        // out-of-sync (for any reason): reset sync times, step
                        if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
                            misaligned || sim.speed_changed) {
                            // re-sync
                            syncCPU = startCPU;
                            syncSim = d->time;
                            sim.speed_changed = false;

                            // run single step, let next iteration deal with timing
                            mj_step(m, d);
                        }

                            // in-sync: step until ahead of cpu
                        else {
                            bool measured = false;
                            mjtNum prevSim = d->time;

                            double refreshTime = simRefreshFraction / sim.refresh_rate;

                            // step while sim lags behind cpu and within refreshTime
                            while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                                   mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime)) {
                                // measure slowdown before first step
                                if (!measured && elapsedSim) {
                                    sim.measured_slowdown =
                                            std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                                    measured = true;
                                }

                                // 此处对机器人进行控制

                                // call mj_step
                                mj_step(m, d);

                                // break if reset
                                if (d->time < prevSim) {
                                    break;
                                }
                            }
                        }
                    }

                        // paused
                    else {
                        // run mj_forward, to update rendering and joint sliders
                        mj_forward(m, d);
                    }
                }
            }  // release std::lock_guard<std::mutex>
        }
    }
}  // namespace

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj::Simulate *sim, const char *filename) {
    // request loadmodel if file given (otherwise drag-and-drop)
    if (filename != nullptr) {
        m = LoadModel(filename, *sim);
        if (m) d = mj_makeData(m);
        if (d) {
            sim->Load(m, d, filename);
            mj_forward(m, d);

            // allocate ctrlnoise
            free(ctrlnoise);
            ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
            mju_zero(ctrlnoise, m->nu);
        }
    }

    PhysicsLoop(*sim);

    // delete everything we allocated
    free(ctrlnoise);
    mj_deleteData(d);
    mj_deleteModel(m);
}

//--------------------------------------- algorithm ------------------------------------------------
void pub_joint_state() {
    joint_state_ptr_->rf_pos[0] = d->sensordata[0];
    joint_state_ptr_->rf_pos[1] = d->sensordata[1];
    joint_state_ptr_->rf_pos[2] = d->sensordata[2];

    joint_state_ptr_->lf_pos[0] = d->sensordata[3];
    joint_state_ptr_->lf_pos[1] = d->sensordata[4];
    joint_state_ptr_->lf_pos[2] = d->sensordata[5];

    joint_state_ptr_->rb_pos[0] = d->sensordata[6];
    joint_state_ptr_->rb_pos[1] = d->sensordata[7];
    joint_state_ptr_->rb_pos[2] = d->sensordata[8];

    joint_state_ptr_->lb_pos[0] = d->sensordata[9];
    joint_state_ptr_->lb_pos[1] = d->sensordata[10];
    joint_state_ptr_->lb_pos[2] = d->sensordata[11];

    joint_state_ptr_->rf_vel[0] = d->sensordata[12];
    joint_state_ptr_->rf_vel[1] = d->sensordata[13];
    joint_state_ptr_->rf_vel[2] = d->sensordata[14];

    joint_state_ptr_->lf_vel[0] = d->sensordata[15];
    joint_state_ptr_->lf_vel[1] = d->sensordata[16];
    joint_state_ptr_->lf_vel[2] = d->sensordata[17];

    joint_state_ptr_->rb_vel[0] = d->sensordata[18];
    joint_state_ptr_->rb_vel[1] = d->sensordata[19];
    joint_state_ptr_->rb_vel[2] = d->sensordata[20];

    joint_state_ptr_->lb_vel[0] = d->sensordata[21];
    joint_state_ptr_->lb_vel[1] = d->sensordata[22];
    joint_state_ptr_->lb_vel[2] = d->sensordata[23];

    joint_state_lcm_ptr_->publish("joint_state", joint_state_ptr_.get());
}

void pub_imu_state() {
    imu_state_ptr_->acc[0] =  d->sensordata[24];
    imu_state_ptr_->acc[1] =  d->sensordata[25];
    imu_state_ptr_->acc[2] =  d->sensordata[26];

    imu_state_ptr_->gyro[0] =  d->sensordata[27];
    imu_state_ptr_->gyro[1] =  d->sensordata[28];
    imu_state_ptr_->gyro[2] =  d->sensordata[29];

    imu_state_ptr_->pos[0] = d->sensordata[30];
    imu_state_ptr_->pos[1] = d->sensordata[31];
    imu_state_ptr_->pos[2] = d->sensordata[32];

    imu_state_ptr_->quat[0] = d->sensordata[33];
    imu_state_ptr_->quat[1] = d->sensordata[34];
    imu_state_ptr_->quat[2] = d->sensordata[35];
    imu_state_ptr_->quat[3] = d->sensordata[36];

    imu_state_lcm_ptr_->publish("imu_state", imu_state_ptr_.get());
}

void update_state() {
    imu_acc_ << imu_state_ptr_->acc[0], imu_state_ptr_->acc[1], imu_state_ptr_->acc[2];
    imu_gyro_ << imu_state_ptr_->gyro[0], imu_state_ptr_->gyro[1], imu_state_ptr_->gyro[2];
    imu_quat_  = Eigen::Quaterniond (imu_state_ptr_->quat[0], imu_state_ptr_->quat[1], imu_state_ptr_->quat[2], imu_state_ptr_->quat[3]);
    imu_pos_ << imu_state_ptr_->pos[0], imu_state_ptr_->pos[1], imu_state_ptr_->pos[2];

    for(int i=0; i<3; i++) {
        lf_pos_[i] = joint_state_ptr_->lf_pos[i];
        rf_pos_[i] = joint_state_ptr_->rf_pos[i];
        lb_pos_[i] = joint_state_ptr_->lb_pos[i];
        rb_pos_[i] = joint_state_ptr_->rb_pos[i];

        lf_vel_[i] = joint_state_ptr_->lf_vel[i];
        rf_vel_[i] = joint_state_ptr_->rf_vel[i];
        lb_vel_[i] = joint_state_ptr_->lb_vel[i];
        rb_vel_[i] = joint_state_ptr_->rb_vel[i];
    }
}

void joint_control_pd() {
    for(int i=0; i<3; i++) {
        joint_control_ptr_->lf_tau[i] = lf_tau_[i] = kp_[i] * (lf_pos_des_[i] - lf_pos_[i]) + kd_[i]*(0 - lf_vel_[i]);
        joint_control_ptr_->rf_tau[i] = rf_tau_[i] = kp_[i] * (rf_pos_des_[i] - rf_pos_[i]) + kd_[i]*(0 - rf_vel_[i]);
        joint_control_ptr_->lb_tau[i] = lb_tau_[i] = kp_[i] * (lb_pos_des_[i] - lb_pos_[i]) + kd_[i]*(0 - lb_vel_[i]);
        joint_control_ptr_->rb_tau[i] = rb_tau_[i] = kp_[i] * (rb_pos_des_[i] - rb_pos_[i]) + kd_[i]*(0 - rb_vel_[i]);
    }

    joint_control_lcm_ptr_->publish("joint_control", joint_control_ptr_.get());

    d->ctrl[0] = rf_tau_[0];
    d->ctrl[1] = rf_tau_[1];
    d->ctrl[2] = rf_tau_[2];

    d->ctrl[3] = lf_tau_[0];
    d->ctrl[4] = lf_tau_[1];
    d->ctrl[5] = lf_tau_[2];

    d->ctrl[6] = rb_tau_[0];
    d->ctrl[7] = rb_tau_[1];
    d->ctrl[8] = rb_tau_[2];

    d->ctrl[9] = lb_tau_[0];
    d->ctrl[10] = lb_tau_[1];
    d->ctrl[11] = lb_tau_[2];
}

void ControlThread(mj::Simulate *sim) {
    while (!sim->exitrequest.load()) {
        if (sim->run && sim->busywait) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if(delay_cnt < 0) {
            delay_cnt++;
        } else {
            if(sim->run && m != nullptr) {
//            LOG(INFO) << "qpos size: " << m->nu;
                // 更新机器人状态, 此处先使用仿真直接提供的Body状态
                pub_joint_state();
                pub_imu_state();
                update_state();

                joint_control_pd();
            }
        }
    }
}

//------------------------------------------ main --------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char* title, const char* msg);
static const char* rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char* msg) {
  rosetta_error_msg = msg;
}
#endif

// run event loop
int main(int argc, const char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::SetStderrLogging(google::INFO);

    joint_state_lcm_ptr_ = std::make_shared<lcm::LCM>();
    if (!joint_state_lcm_ptr_->good()) {
        LOG(FATAL) << "joint state lcm is not good";
    }
    imu_state_lcm_ptr_ = std::make_shared<lcm::LCM>();
    if (!imu_state_lcm_ptr_->good()) {
        LOG(FATAL) << "imu state lcm is not good";
    }
    joint_control_lcm_ptr_ = std::make_shared<lcm::LCM>();
    if (!joint_control_lcm_ptr_->good()) {
        LOG(FATAL) << "imu state lcm is not good";
    }

    joint_state_ptr_ = std::make_shared<joint_state_lcm>();
    imu_state_ptr_ = std::make_shared<imu_state_lcm>();
    joint_control_ptr_ = std::make_shared<joint_control_lcm>();

    // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
    if (rosetta_error_msg) {
      DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
      std::exit(1);
    }
#endif

    // print version, check compatibility
    std::printf("MuJoCo version %s\n", mj_versionString());
    if (mjVERSION_HEADER != mj_version()) {
        mju_error("Headers and library have different versions");
    }

    // scan for libraries in the plugin directory to load additional plugins
    scanPluginLibraries();

    mjvScene scn;
    mjv_defaultScene(&scn);

    mjvCamera cam;
    mjv_defaultCamera(&cam);

    mjvOption opt;
    mjv_defaultOption(&opt);

    mjvPerturb pert;
    mjv_defaultPerturb(&pert);

    // simulate object encapsulates the UI
    auto sim = std::make_unique<mj::Simulate>(
            std::make_unique<mj::GlfwAdapter>(),
            &scn, &cam, &opt, &pert, /* fully_managed = */ true
    );

    const char *filename = nullptr;
    if (argc > 1) {
        filename = argv[1];
    }

    filename = "/home/huangyong1/Software/mujoco_sim/quad_mujoco_sim/third_party/mujoco-2.3.5/model/quadruped/aliengo/xml/aliengo.xml";
    std::cout << "filename: " << filename << std::endl;

    // start physics thread
    std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename);
    std::thread controlthreadhandle(&ControlThread, sim.get());

    // start simulation UI loop (blocking call)
    sim->RenderLoop();
    physicsthreadhandle.join();
    controlthreadhandle.join();

    google::ShutdownGoogleLogging();

    return 0;
}
