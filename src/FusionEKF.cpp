#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);

    // Laser - Measurement covariance matrix
    R_laser_ << 0.0225, 0,
            0, 0.0225;

    // Radar - Measurement covariance matrix
    R_radar_ << 0.09, 0, 0,
            0, 0.0009, 0,
            0, 0, 0.09;

    /*
     * Initialize the FusionEKF.
     * Set the process and measurement noises
     */
    // Measurement function matrix
    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;

    // State Covariance Matrix
    MatrixXd P_ = MatrixXd(4, 4);
    P_ << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1000, 0,
          0, 0, 0, 1000;

    // State Transition Matrix
    MatrixXd F_ = MatrixXd(4, 4);
    F_ << 1, 0, 1, 0,
          0, 1, 0, 1,
          0, 0, 1, 0,
          0, 0, 0, 1;

    // New covariance matrix - based on noise vector
    MatrixXd Q_ = MatrixXd(4, 4);
    Q_ << 1, 0, 1, 0,
          0, 1, 0, 1,
          1, 0, 1, 0,
          0, 1, 0, 1;

    VectorXd x_ = VectorXd(4);
    x_ << 1, 1, 1, 1;
    ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        /**
          * Initialize the state ekf_.x_ with the first measurement.
          * Create the covariance matrix.
          * Remember: you'll need to convert radar from polar to cartesian coordinates.
        */
        // Initial Measurement
        cout << "EKF: " << endl;
        ekf_.x_ = VectorXd(4);
        ekf_.x_ << 1, 1, 1, 1;

        // Initialize Position and Velocity
        float px = 0.0;
        float py = 0.0;
        float vx = 0.0;
        float vy = 0.0;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            // Convert radar from polar to cartesian coordinates and initialize state.
            float rho = measurement_pack.raw_measurements_[0];
            float phi = measurement_pack.raw_measurements_[1];
            float rho_dot = measurement_pack.raw_measurements_[2];

            px = rho * cos(phi);
            py = rho * sin(phi);
        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            px = measurement_pack.raw_measurements_[0];
            py = measurement_pack.raw_measurements_[1];
        }

        ekf_.x_ << px, py, vx, vy;
        previous_timestamp_ = measurement_pack.timestamp_;

        // Initializing
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    /**
       * Update the state transition matrix F according to the new elapsed time.
        - Time is measured in seconds.
       * Update the process noise covariance matrix.
       * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
     */

    // Calculate time elapsed
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;

    // Update State Transition Matrix F
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    float dt_2 = dt * dt;
    float dt_3 = dt_2 * dt / 2.0;
    float dt_4 = dt_3 * dt / 2.0;

    // Process Co-variance Matrix Q
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << dt_4 * noise_ax, 0, dt_3 * noise_ax, 0,
               0, dt_4 * noise_ay, 0, dt_3 * noise_ay,
               dt_3 * noise_ax, 0, dt_2 * noise_ax, 0,
               0, dt_3 * noise_ay, 0, dt_2 * noise_ay;

    ekf_.Predict();

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    /**
       * Use the sensor type to perform the update step.
       * Update the state and covariance matrices.
     */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar
        try {
            ekf_.R_ = R_radar_;
            ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
            ekf_.UpdateEKF(measurement_pack.raw_measurements_);
        } catch (...) {
            return;
        }
    } else {
        // Laser
        ekf_.R_ = R_laser_;
        ekf_.H_ = H_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // Output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}