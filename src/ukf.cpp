#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.8;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // filter is initialized after the first measurement
  is_initialized_ = false;

  // CSTR state dimension (px, py, v, yaw, yawd)
  n_x_ = 5;

  // augumented state dimension, additional (nu_a, nu_yawdd)
  // as acceleration and yaw acceleration
  n_aug_ = n_x_ + 2;

  // number of sigma points - mean and another 2 points for each dimension
  n_sig_ = n_aug_ * 2 + 1;

  // lambda controls range of sigma point
  lambda_ = 3 - n_aug_;

  // weights for lambda points 
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_); // first weight for mean

  // sigma points for prediction, they will be used to 
  // estimate a Gaussian approximation to the state distribution of next step.
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  Process each measurement (Radar or Lidar), update internal state and its
  covariance.
  */
  if (! is_initialized_) { /* initialize with first measurement */
    
    double px = 0, py = 0;
    auto m = meas_package.raw_measurements_;
    
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      px = m(0); py = m(1);
    } else { /*meas_package.sensor_type_ == MeasurementPackage::RADAR*/
      auto rho = m(0), phi = m(1);
      px = rho * cos(phi); py = rho * sin(phi);
    }
    
    x_ << px, py, /*v=*/0, /*yaw=*/0, /*yawd*/0;
    P_ <<  0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
    
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
  
  } else {                 /* prediction-update loop after first measurement */
    
    double dt/*sec*/ = (meas_package.timestamp_ - time_us_)/*millisec*/ / 1000000.0;

    if (use_radar_ and meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      Prediction(dt);
      UpdateRadar(meas_package);
      time_us_ = meas_package.timestamp_;
    }

    if (use_laser_ and meas_package.sensor_type_ == MeasurementPackage::LASER) {
      Prediction(dt);
      UpdateLidar(meas_package);
      time_us_ = meas_package.timestamp_;
    }
  
  }
  
} /*end of UKF::ProcessMeasurement*/

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double dt) {
  
  /** Predict step1. Generate augumented sigma points based on current state and covariance 
  */

  // augmentetd state and cov: 
  // (px, py, v, yaw, yawd) += (a, yawdd)
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  x_aug << x_, /*a=*/0, /*yawdd=*/0;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(n_aug_-n_x_, n_aug_-n_x_).diagonal() << std_a_*std_a_, /*acceleration noise variance*/
                                                          std_yawdd_*std_yawdd_; /*yaw accel noise variance*/

  // each col of Xsig_aug is an augumented sigma point
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);

  MatrixXd A = P_aug.llt().matrixL(); // sqrt of augmented covariance
  A *= sqrt(lambda_ + n_aug_);

  Xsig_aug << x_aug,               /*first sigma point is the mean*/ 
              A.colwise() + x_aug, /*followed by mean +/- sqrt of cov*/
              (-A).colwise() + x_aug;

  /** Predict step2. Transform augumented sigma points at time t to state sigma points at t+1
  */
  for (auto i = 0; i < n_sig_; i++) { /* for each sigma point */

    auto px      = Xsig_aug(0, i),
         py      = Xsig_aug(1, i),
         v       = Xsig_aug(2, i),
         yaw     = Xsig_aug(3, i),
         yawd    = Xsig_aug(4, i),
         acc     = Xsig_aug(5, i),
         yawdd   = Xsig_aug(6, i);
    
    auto dt2 = 0.5 * dt * dt;

    VectorXd order1st(n_x_); 
    if (fabs(yawd) >= GT_ZERO) { /* first order kinetics for curved path*/

      auto new_yaw = yaw + yawd * dt;
      auto ratio = v / yawd;
      order1st << ratio * (sin(new_yaw) - sin(yaw)), /*=px*/
                  ratio * (-cos(new_yaw) + cos(yaw)),/*=py*/
                  0,                                 /*=v*/
                  yawd * dt,                         /*=yaw*/
                  0;                                 /*=yawd*/
    } else {                       /* first order kinetics for straight path*/
      order1st << v * cos(yaw) * dt,
                  v * sin(yaw) * dt,
                  0,
                  yawd * dt,
                  0;
    }

    VectorXd order2nd(n_x_);      /* second order kinetics */
    order2nd <<   dt2 * cos(yaw) * acc,
                  dt2 * sin(yaw) * acc,
                  dt * acc,
                  dt2 * yawdd,
                  dt * yawdd;

    Xsig_pred_.col(i) = Xsig_aug.col(i).head(n_x_) + order1st + order2nd;  

  }

  /** Predict step3. Estimate Gaussian approximation of state distribution at t+1 from new sigma points
  */

  // new state mean is the weighted sum of new sigma points
  x_ = Xsig_pred_ * weights_; 
  // new state covariance as weighted correlation
  MatrixXd diff = Xsig_pred_.colwise() - x_;
  MatrixXd wdiff = diff.array().rowwise() * weights_.transpose().array();
  P_ = wdiff * diff.transpose();

} /*end of UKF::Prediction*/

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) { 
  
  /** Update step1. transform state sigma points to measurement sigma points.
  It will be different for LASER and RADAR.
  */

  auto n_z = 2; /*dimension of observation (px, py)*/

  /*observation (px, py) is the first 2 components of state (px, py, v, yaw, yawd)*/
  MatrixXd Zsig = Xsig_pred_.topRows(n_z);

  // define measurement noise R
  MatrixXd R(n_z, n_z);
  R.fill(0);
  R.diagonal() << std_laspx_ * std_laspx_, 
                  std_laspy_ * std_laspy_;

  /** Update step2. compare predicted measurement sigma points and real measurement,
  update filter states based on the difference.
  It is common for LASER and RADAR.
  */
  Update_common_steps(Zsig, R, meas_package);

} /*end of UKF::UpdateLidar*/

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  
  /** Update step1. transform state sigma points to measurement sigma points.
  It will be different for LASER and RADAR.
  */

  auto n_z = 3; /*dimension of observation (rho, phi, rhod)*/

  using Eigen::ArrayXd;
  ArrayXd px = Xsig_pred_.row(0).array(),
          py = Xsig_pred_.row(1).array(),
          v  = Xsig_pred_.row(2).array(),
          psi = Xsig_pred_.row(3).array();
  for (auto i = 0; i < px.size(); ++i) { /*avoid divided by zero*/
    px(i) = (fabs(px(i)) >= GT_ZERO) ? px(i) : GT_ZERO;
  }
  
  // each col of Zsig is a sigma point of dim n_z 
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);

  Zsig.row(0)/*rho*/ = (px.square() + py.square()).sqrt();
  Zsig.row(1)/*phi*/ = (py / px).unaryExpr([](double r){return atan(r);});
  Zsig.row(2)/*rhod*/ = ( px*v*psi.cos() + py*v*psi.sin() );
  Zsig.row(2) = Zsig.row(2).array() / Zsig.row(0).array();

  // define measurement noise R
  
  MatrixXd R(n_z, n_z);
  R.fill(0);
  R.diagonal() << std_radr_ * std_radr_,
                  std_radphi_ * std_radphi_,
                  std_radrd_ * std_radrd_;

  /** Update step2. compare predicted measurement sigma points and real measurement,
  update filter states based on the difference.
  It is common for LASER and RADAR.
  */
  Update_common_steps(Zsig, R, meas_package);

} /*end of UKF::UpdateRadar*/

/** Common steps for updating both LASER and RADAR
*/
void UKF::Update_common_steps(const MatrixXd & Zsig,
                              const MatrixXd & R, 
                              const MeasurementPackage & meas_package) {
  /**Update filter state with sigma points for measurements
  */

  /**Gaussian approximation based on measurement sigma points Zsig,
  */
  // mean is the weighted sum
  VectorXd z_pred = Zsig * weights_; 

  // covariance is the weighted correlation
  MatrixXd diff = Zsig.colwise() - z_pred;
  MatrixXd wdiff = diff.array().rowwise() * weights_.transpose().array();
  MatrixXd S = wdiff * diff.transpose();
  S = S + R;

  /** Get the real measurement value
  */
  VectorXd z = meas_package.raw_measurements_;

  /** Estimate kalman gain based on state/measurement sigma points
  */
  MatrixXd xdiff = Xsig_pred_.colwise() - x_;
  MatrixXd zdiff = Zsig.colwise() - z_pred;
  MatrixXd wxdiff = xdiff.array().rowwise() * weights_.transpose().array();
  MatrixXd K = (wxdiff * zdiff.transpose()) * S.inverse();

  /** Update state by the difference between real measurement and its prediction
  */
  VectorXd residual = z - z_pred;
  // angle normalization - introduced in the class, not I found it unnecessary
  // for the given test data.
  // while (residual(1)> M_PI) residual(1) -= 2*M_PI;
  // while (residual(1)<-M_PI) residual(1) += 2*M_PI;

  x_ = x_ + (K * residual);

  P_ = P_ - (K * S * K.transpose());
  NIS_radar_ = residual.transpose() * S.inverse() * residual;
} /*end of Update_common_steps*/
